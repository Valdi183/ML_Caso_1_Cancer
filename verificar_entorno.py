#!/usr/bin/env python3
"""
Comprueba que esta maquina puede entrenar el caso BreastDCEDL.

Lanzalo despues de configurar un equipo nuevo, o cuando algo deje de funcionar.
No modifica nada: solo mira y dice que falta.

    python verificar_entorno.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

OK, MAL, AVISO = "  OK  ", " FALLA", "  !   "
problemas: list[str] = []


def comprobar(etiqueta: str, funcion):
    """Ejecuta una comprobacion y la imprime sin dejar que una excepcion corte."""
    try:
        detalle = funcion()
        print(f"{OK}  {etiqueta}: {detalle}")
        return True
    except Exception as e:
        print(f"{MAL}  {etiqueta}: {e}")
        problemas.append(etiqueta)
        return False


# --------------------------------------------------------------------------- #
print(f"\n=== Entorno =================================================")
print(f"       maquina: {platform.node()}  ({platform.system()} {platform.release()})")
print(f"       python : {sys.version.split()[0]}  en  {sys.prefix}")

if sys.version_info < (3, 10):
    problemas.append("python demasiado antiguo")
    print(f"{MAL}  python: hace falta 3.10 o superior")

if Path(sys.prefix) == Path(sys.base_prefix):
    print(f"{AVISO}  no estas dentro del entorno virtual (.venv)")


# --------------------------------------------------------------------------- #
print(f"\n=== Paquetes ================================================")

def version_de(modulo: str):
    def _f():
        import importlib
        return importlib.import_module(modulo).__version__
    return _f

for nombre, modulo in [("numpy", "numpy"), ("pandas", "pandas"), ("pillow", "PIL"),
                       ("matplotlib", "matplotlib"), ("scikit-learn", "sklearn")]:
    comprobar(nombre, version_de(modulo))


# --------------------------------------------------------------------------- #
print(f"\n=== PyTorch =================================================")
hay_gpu = False
try:
    import torch
    print(f"{OK}  torch: {torch.__version__}")
    hay_gpu = torch.cuda.is_available()
    if hay_gpu:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"{OK}  GPU {i}: {props.name}  "
                  f"{props.total_memory / 1024**3:.1f} GB  cuda {torch.version.cuda}")
    else:
        print(f"{AVISO}  sin GPU: entrenar aqui sera lento. Solo para explorar y depurar.")
except ImportError as e:
    print(f"{MAL}  torch: {e}")
    problemas.append("torch")


# --------------------------------------------------------------------------- #
print(f"\n=== Dataset =================================================")
raiz = None
try:
    import config
    raiz = config.raiz_datos()
    print(f"{OK}  carpeta: {raiz}")
except Exception as e:
    print(f"{MAL}  {e}")
    problemas.append("dataset")

if raiz is not None:
    def cuenta_imagenes():
        n = sum(1 for _ in (raiz / "dataset").rglob("*.png"))
        if n != 38109:
            raise RuntimeError(f"{n} de 38109 PNG. Relanza descargar_datos.py, es reanudable.")
        return f"{n} PNG"

    comprobar("imagenes", cuenta_imagenes)

    def metadatos():
        uc = config.utils()
        samples = uc.cargar_samples()
        if len(samples) != 12703:
            raise RuntimeError(f"{len(samples)} cortes, esperaba 12703")
        tr, va = uc.particion(samples, fold_val=0)
        solape = set(tr.patient_id) & set(va.patient_id)
        if solape:
            raise RuntimeError(f"fuga por paciente: {len(solape)} pacientes en ambos lados")
        return (f"{len(samples)} cortes, {samples.patient_id.nunique()} pacientes, "
                f"pos_weight {uc.pos_weight(tr):.3f}")

    comprobar("metadatos y particion", metadatos)

    def fases_correctas():
        uc = config.utils()
        samples = uc.cargar_samples()
        malas = 0
        for _, fila in samples.sample(20, random_state=0).iterrows():
            x = uc.cargar_imagen(fila)
            tejido = x[0] > 0.1
            pre, early, _ = [float(c[tejido].mean()) for c in x]
            if not pre < early:
                malas += 1
        if malas:
            raise RuntimeError(f"PRE >= EARLY en {malas}/20: las fases no se leen bien")
        return "PRE < EARLY en 20/20 muestras, el realce se lee bien"

    comprobar("orden de fases PRE/EARLY/LATE", fases_correctas)


# --------------------------------------------------------------------------- #
if raiz is not None and "torch" not in problemas:
    print(f"\n=== Velocidad ===============================================")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        uc = config.utils()
        tr, _ = uc.particion(uc.cargar_samples(), fold_val=0)
        dispositivo = "cuda" if hay_gpu else "cpu"

        capas = []
        for a, b in ((3, 16), (16, 32), (32, 64), (64, 128)):
            capas += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b),
                      nn.ReLU(), nn.MaxPool2d(2)]
        modelo = nn.Sequential(*capas, nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(), nn.Linear(128, 1)).to(dispositivo)
        opt = torch.optim.Adam(modelo.parameters(), 1e-3)
        crit = nn.BCEWithLogitsLoss()

        dl = DataLoader(uc.BreastDCEDataset(tr), batch_size=32, shuffle=True)
        inicio, lotes = time.perf_counter(), 0
        for x, y in dl:
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad()
            crit(modelo(x).squeeze(1), y).backward()
            opt.step()
            lotes += 1
            if lotes == 6:
                break
        if hay_gpu:
            torch.cuda.synchronize()

        seg = (time.perf_counter() - inicio) / lotes
        total = (len(tr) + 31) // 32
        print(f"{OK}  {dispositivo}: {seg:.2f} s/lote (batch 32, CNN de 98k parametros)")
        print(f"       una epoca = {total} lotes = {seg * total / 60:.1f} min")
    except Exception as e:
        print(f"{AVISO}  no se pudo medir: {e}")


# --------------------------------------------------------------------------- #
print(f"\n=============================================================")
if problemas:
    print(f"{len(problemas)} problema(s): {', '.join(problemas)}")
    sys.exit(1)
print("Entorno listo.")
