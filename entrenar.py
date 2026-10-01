#!/usr/bin/env python3
"""Entrena la CNN de pCR con un fold de validacion por paciente.

    # Prueba rapida en el portatil (CPU): pocas pacientes, 2 epocas, minutos
    python entrenar.py --rapido

    # Entrenamiento real en el laboratorio (GPU)
    python entrenar.py                          # base, perdida normal
    python entrenar.py --ponderada              # base, perdida ponderada
    python entrenar.py --ponderada --aumentado --realce

Cada ejecucion guarda en resultados/<nombre>/ la configuracion, el historial por
epoca, las metricas de validacion, las probabilidades por corte y las curvas; y
los pesos del mejor modelo en modelos/<nombre>.pt. El nombre sale de las
opciones si no se da con --nombre.

Test NO se usa aqui. Se evalua una sola vez, al final, con el modelo ya elegido.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # antes que matplotlib: al reves, en Windows chocan sus DLL
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from tqdm import tqdm

import config
from modelo import CNNpCR, guardar, parametros

AQUI = Path(__file__).resolve().parent
uc = config.utils()


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", help="nombre del experimento (por defecto, sale de las opciones)")
    p.add_argument("--fold", type=int, default=0, help="fold de validacion, 0-4")
    p.add_argument("--ponderada", action="store_true", help="BCE con pos_weight = N0/N1")
    p.add_argument("--realce", action="store_true", help="anade EARLY-PRE y LATE-EARLY como canales")
    p.add_argument("--sin-late", action="store_true", help="descarta la fase LATE")
    p.add_argument("--aumentado", action="store_true", help="volteos y rotaciones de hasta 15 grados")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--paciencia", type=int, default=8, help="epocas sin mejorar el AUC antes de parar")
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay de AdamW")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--rapido", action="store_true",
                   help="prueba del circuito: 40 pacientes de train, 20 de validacion, 2 epocas")
    a = p.parse_args()

    if a.rapido:
        a.epocas = 2
    if a.nombre is None:
        partes = ["ponderada" if a.ponderada else "normal"]
        partes += [n for n, activa in (("realce", a.realce), ("sinlate", a.sin_late),
                                       ("aum", a.aumentado)) if activa]
        partes.append(f"f{a.fold}")
        a.nombre = ("rapido_" if a.rapido else "") + "_".join(partes)
    return a


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# Datos: todas las imagenes en memoria como uint8
# --------------------------------------------------------------------------- #

def submuestra(filas: pd.DataFrame, pacientes: int, semilla: int) -> pd.DataFrame:
    """Para --rapido: unas pocas pacientes, manteniendo la proporcion de pCR."""
    por_paciente = filas.groupby("patient_id").pCR.first()
    elegidas = (por_paciente.groupby(por_paciente, group_keys=False)
                            .apply(lambda g: g.sample(max(1, round(pacientes * len(g) / len(por_paciente))),
                                                      random_state=semilla)))
    return filas[filas.patient_id.isin(elegidas.index)]


def cargar_en_memoria(filas: pd.DataFrame, etiqueta: str) -> torch.Tensor:
    """(N, 3, 256, 256) uint8. Leer los PNG en cada epoca haria esperar a la GPU.

    Se usa `uc.cargar_imagen` (la misma funcion que la app) y se vuelve a 0-255:
    es exacto, porque las imagenes son PNG de 8 bits divididos entre 255.
    """
    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as hilos:
        imagenes = list(hilos.map(lambda f: np.rint(uc.cargar_imagen(f) * 255).astype(np.uint8),
                                  filas.itertuples()))
    tensor = torch.from_numpy(np.stack(imagenes))
    print(f"  {etiqueta}: {len(filas)} cortes, {filas.patient_id.nunique()} pacientes, "
          f"{tensor.numel() / 1024**3:.2f} GB, {time.perf_counter() - inicio:.0f} s")
    return tensor


def aumentar(x: torch.Tensor) -> torch.Tensor:
    """Volteos y rotacion de hasta 15 grados, iguales para los canales de un corte.

    Se transforma el tensor completo de cada corte, nunca canal a canal: desalinear
    PRE, EARLY y LATE destruiria el realce.
    """
    n = x.shape[0]
    volteo_h = torch.rand(n, device=x.device) < 0.5
    volteo_v = torch.rand(n, device=x.device) < 0.5
    x = torch.where(volteo_h[:, None, None, None], x.flip(3), x)
    x = torch.where(volteo_v[:, None, None, None], x.flip(2), x)

    angulo = (torch.rand(n, device=x.device) * 2 - 1) * np.deg2rad(15)
    cos, sen = torch.cos(angulo), torch.sin(angulo)
    ceros = torch.zeros_like(angulo)
    theta = torch.stack([torch.stack([cos, -sen, ceros], 1), torch.stack([sen, cos, ceros], 1)], 1)
    rejilla = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, rejilla, align_corners=False, padding_mode="zeros")


# --------------------------------------------------------------------------- #
# Entrenamiento y evaluacion
# --------------------------------------------------------------------------- #

def lotes(n: int, tamano: int, desordenar: bool):
    orden = torch.randperm(n) if desordenar else torch.arange(n)
    for i in range(0, n, tamano):
        yield orden[i:i + tamano]


def una_epoca(red, imagenes, etiquetas, criterio, optimizador, args, dispositivo, epoca) -> float:
    red.train()
    total, vistos = 0.0, 0
    barra = tqdm(lotes(len(imagenes), args.lote, desordenar=True), total=-(-len(imagenes) // args.lote),
                 desc=f"  epoca {epoca:2d}", leave=False, ncols=90)
    for indices in barra:
        x = imagenes[indices].to(dispositivo).float() / 255
        y = etiquetas[indices].to(dispositivo)
        if args.aumentado:
            x = aumentar(x)
        optimizador.zero_grad()
        perdida = criterio(red(x), y)
        perdida.backward()
        optimizador.step()
        total += perdida.item() * len(indices)
        vistos += len(indices)
        barra.set_postfix(perdida=f"{total / vistos:.4f}")
    return total / vistos


@torch.no_grad()
def predecir(red, imagenes, dispositivo) -> np.ndarray:
    """Probabilidad por corte, en el mismo orden que las filas (sin desordenar)."""
    red.eval()
    salidas = [torch.sigmoid(red(imagenes[i].to(dispositivo).float() / 255)).cpu()
               for i in lotes(len(imagenes), 128, desordenar=False)]
    return torch.cat(salidas).numpy()


def perdida_media(probs: np.ndarray, etiquetas: torch.Tensor, criterio) -> float:
    logits = torch.logit(torch.from_numpy(probs).clamp(1e-6, 1 - 1e-6))
    return criterio(logits, etiquetas).item()


def umbral_youden(filas: pd.DataFrame, probs: np.ndarray) -> float:
    """Umbral que maximiza sensibilidad + especificidad, sobre la media por paciente."""
    por_paciente = uc.agregar_por_paciente(filas.patient_id.values, probs).join(
        filas.groupby("patient_id").pCR.first())
    fpr, tpr, umbrales = roc_curve(por_paciente.pCR, por_paciente.prob)
    return float(umbrales[np.argmax(tpr - fpr)])


def dibujar_curvas(historial: pd.DataFrame, ruta: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].plot(historial.epoca, historial.perdida_train, label="train")
    ejes[0].plot(historial.epoca, historial.perdida_val, label="validacion")
    ejes[0].set(title="Perdida", xlabel="epoca")
    ejes[0].legend()
    ejes[1].plot(historial.epoca, historial.auc_val, color="#55A868")
    ejes[1].axhline(0.69, color="gray", ls="--", lw=1, label="referencia clinica (0,69)")
    ejes[1].axhline(0.5, color="#C44E52", ls=":", lw=1, label="azar")
    ejes[1].set(title="AUC por paciente (validacion)", xlabel="epoca", ylim=(0.3, 1))
    ejes[1].legend(fontsize=8)
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(ruta, dpi=120, facecolor="white")
    plt.close(figura)


def main() -> None:
    args = argumentos()
    fijar_semilla(args.semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carpeta = AQUI / "resultados" / args.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pesos = AQUI / "modelos" / f"{args.nombre}.pt"

    print(f"Experimento {args.nombre} en {dispositivo}"
          + (f" ({torch.cuda.get_device_name()})" if dispositivo.type == "cuda" else ""))

    entreno, valida = uc.particion(uc.cargar_samples(), fold_val=args.fold)
    if args.rapido:
        entreno = submuestra(entreno, 40, args.semilla)
        valida = submuestra(valida, 20, args.semilla)
    entreno, valida = entreno.reset_index(drop=True), valida.reset_index(drop=True)

    print("Cargando imagenes en memoria:")
    img_entreno = cargar_en_memoria(entreno, "train")
    img_valida = cargar_en_memoria(valida, "validacion")
    y_entreno = torch.tensor(entreno.pCR.values, dtype=torch.float32)
    y_valida = torch.tensor(valida.pCR.values, dtype=torch.float32)

    red = CNNpCR(realce=args.realce, sin_late=args.sin_late).to(dispositivo)
    peso_positivos = uc.pos_weight(entreno)
    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(peso_positivos, device=dispositivo) if args.ponderada else None)
    criterio_cpu = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso_positivos) if args.ponderada else None)
    optimizador = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=args.wd)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max", factor=0.5, patience=3)

    configuracion = {**vars(args), "dispositivo": str(dispositivo), "parametros": parametros(red),
                     "pos_weight": peso_positivos, "arquitectura": red.config,
                     "cortes_train": len(entreno), "cortes_val": len(valida),
                     "pacientes_train": entreno.patient_id.nunique(),
                     "pacientes_val": valida.patient_id.nunique()}
    (carpeta / "config.json").write_text(json.dumps(configuracion, indent=2), encoding="utf-8")
    n = f"{parametros(red):,}".replace(",", ".")
    print(f"Red: {n} parametros, {red.preprocesado.canales} canales de entrada")

    historial, mejor_auc, sin_mejorar = [], -1.0, 0
    for epoca in range(1, args.epocas + 1):
        inicio = time.perf_counter()
        perdida_train = una_epoca(red, img_entreno, y_entreno, criterio, optimizador, args, dispositivo, epoca)
        probs = predecir(red, img_valida, dispositivo)
        metricas = uc.evaluar_por_paciente(probs, valida, umbral=0.5, metodo="mean")
        auc = metricas["auc"]
        planificador.step(auc)

        historial.append({"epoca": epoca, "perdida_train": perdida_train,
                          "perdida_val": perdida_media(probs, y_valida, criterio_cpu),
                          "auc_val": auc, "sensibilidad_05": metricas["sensibilidad"],
                          "especificidad_05": metricas["especificidad"],
                          "lr": optimizador.param_groups[0]["lr"],
                          "segundos": time.perf_counter() - inicio})
        mejora = auc > mejor_auc
        print(f"  epoca {epoca:2d}  perdida {perdida_train:.4f}  AUC val {auc:.3f}"
              f"  ({historial[-1]['segundos']:.0f} s){'  *' if mejora else ''}")

        if mejora:
            mejor_auc, sin_mejorar = auc, 0
            guardar(red, ruta_pesos, epoca=epoca, auc_val=auc)
        else:
            sin_mejorar += 1
            if sin_mejorar >= args.paciencia:
                print(f"  Parada temprana: {args.paciencia} epocas sin mejorar")
                break

    historial = pd.DataFrame(historial)
    historial.to_csv(carpeta / "historial.csv", index=False)
    dibujar_curvas(historial, carpeta / "curvas.png", args.nombre)

    # ---- Evaluacion del mejor modelo en validacion ---------------------------
    paquete = torch.load(ruta_pesos, map_location=dispositivo, weights_only=True)
    red.load_state_dict(paquete["pesos"])
    probs = predecir(red, img_valida, dispositivo)
    valida.assign(prob=probs)[["sample_id", "patient_id", "pCR", "prob"]].to_csv(
        carpeta / "predicciones_val.csv", index=False)

    umbral = umbral_youden(valida, probs)
    agregaciones = {m: uc.evaluar_por_paciente(probs, valida, metodo=m)["auc"] for m in uc.METODOS_AGREGACION}
    cohorte = valida.patient_id.map(uc.cargar_patients().set_index("pid").dataset)
    por_cohorte = {}
    for c in sorted(cohorte.unique()):
        mascara = (cohorte == c).values
        if valida[mascara].pCR.nunique() == 2:
            por_cohorte[c] = uc.evaluar_por_paciente(probs[mascara], valida[mascara], metodo="mean")["auc"]

    resultado = {
        "mejor_epoca": int(paquete["info"]["epoca"]),
        "auc_val_media": mejor_auc,
        "auc_por_agregacion": agregaciones,
        "auc_por_cohorte": por_cohorte,
        "umbral_05": uc.evaluar_por_paciente(probs, valida, umbral=0.5, metodo="mean"),
        "umbral_youden": uc.evaluar_por_paciente(probs, valida, umbral=umbral, metodo="mean"),
    }
    (carpeta / "metricas.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")

    # Los pesos llevan lo que la app tiene que mostrar
    guardar(red, ruta_pesos, epoca=resultado["mejor_epoca"], auc_val=mejor_auc, umbral=umbral,
            metodo_agregacion="mean", entrenamiento={k: configuracion[k] for k in (
                "fold", "ponderada", "aumentado", "epocas", "lote", "lr", "wd", "semilla", "pos_weight")})

    y = resultado["umbral_youden"]
    print(f"\nMejor epoca {resultado['mejor_epoca']}: AUC por paciente {mejor_auc:.3f}")
    print(f"Umbral Youden {umbral:.3f}: sensibilidad {y['sensibilidad']:.2f}, "
          f"especificidad {y['especificidad']:.2f}, matriz {y['matriz_confusion']}")
    print(f"AUC por cohorte: { {c: round(v, 3) for c, v in por_cohorte.items()} }")
    print(f"Guardado en {carpeta.relative_to(AQUI)} y {ruta_pesos.relative_to(AQUI)}")


if __name__ == "__main__":
    main()
