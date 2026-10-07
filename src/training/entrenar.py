#!/usr/bin/env python3
"""Entrena la CNN de pCR con un fold de validacion por paciente.

Se lanza como modulo desde la raiz del repositorio:

    # Prueba rapida en el portatil (CPU): pocas pacientes, 2 epocas, minutos
    python -m src.training.entrenar --rapido

    # Entrenamiento real en el laboratorio (GPU)
    python -m src.training.entrenar                          # base, perdida normal
    python -m src.training.entrenar --ponderada              # base, perdida ponderada
    python -m src.training.entrenar --ponderada --aumentado --realce
    python -m src.training.entrenar --ponderada --aumentado --normalizar --pooling atencion

Cada ejecucion guarda en results/<nombre>/ la configuracion, el historial por
epoca, las metricas de validacion, las probabilidades por corte y las curvas; y
los pesos del mejor modelo en models/<nombre>.pt. El nombre sale de las
opciones si no se da con --nombre. El historial y las curvas se reescriben al
terminar cada epoca: si se corta, se conserva lo hecho hasta ahi.

Los valores por defecto (semilla, epocas, lote, lr...) estan en `src/config.py`.

Test NO se usa aqui. Se evalua una sola vez, al final, con el modelo ya elegido.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch  # antes que matplotlib: al reves, en Windows chocan sus DLL
import torch.nn as nn
from tqdm import tqdm

from src import config
from src.data.carga import aumentar, cargar_en_memoria, lotes, submuestra
from src.models.cnn import POOLINGS, CNNpCR, guardar, parametros
from src.utils.evaluacion import dibujar_curvas, perdida_media, umbral_youden
from src.utils.semilla import fijar_semilla

uc = config.utils()
D = config.DEFECTO


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", help="nombre del experimento (por defecto, sale de las opciones)")
    p.add_argument("--fold", type=int, default=0, help="fold de validacion, 0-4")
    p.add_argument("--ponderada", action="store_true", help="BCE con pos_weight = N0/N1")
    p.add_argument("--realce", action="store_true", help="anade EARLY-PRE y LATE-EARLY como canales")
    p.add_argument("--sin-late", action="store_true", help="descarta la fase LATE")
    p.add_argument("--aumentado", action="store_true", help="volteos y rotaciones de hasta 15 grados")
    p.add_argument("--normalizar", action="store_true",
                   help="estandariza cada corte (media 0, desviacion 1) dentro de la red")
    p.add_argument("--pooling", choices=POOLINGS, default="media",
                   help="como se resume el mapa final antes de la capa lineal")
    p.add_argument("--epocas", type=int, default=D["epocas"])
    p.add_argument("--paciencia", type=int, default=D["paciencia"],
                   help="epocas sin mejorar el AUC antes de parar")
    p.add_argument("--lote", type=int, default=D["lote"])
    p.add_argument("--lr", type=float, default=D["lr"])
    p.add_argument("--wd", type=float, default=D["wd"], help="weight decay de AdamW")
    p.add_argument("--semilla", type=int, default=D["semilla"])
    p.add_argument("--rapido", action="store_true",
                   help="prueba del circuito: 40 pacientes de train, 20 de validacion, 2 epocas")
    a = p.parse_args()

    if a.rapido:
        a.epocas = 2
    if a.nombre is None:
        partes = ["ponderada" if a.ponderada else "normal"]
        partes += [n for n, activa in (("realce", a.realce), ("sinlate", a.sin_late),
                                       ("aum", a.aumentado), ("norm", a.normalizar)) if activa]
        if a.pooling != "media":
            partes.append(a.pooling)
        partes.append(f"f{a.fold}")
        a.nombre = ("rapido_" if a.rapido else "") + "_".join(partes)
    return a


# --------------------------------------------------------------------------- #
# Entrenamiento y evaluacion
# --------------------------------------------------------------------------- #

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


def main() -> None:
    args = argumentos()
    fijar_semilla(args.semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carpeta = config.RESULTADOS / args.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pesos = config.MODELOS / f"{args.nombre}.pt"

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

    red = CNNpCR(realce=args.realce, sin_late=args.sin_late, normalizar=args.normalizar,
                 pooling=args.pooling).to(dispositivo)
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
        perdida_train = una_epoca(red, img_entreno, y_entreno, criterio, optimizador, args,
                                  dispositivo, epoca)
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
        # Cada epoca, no solo al final: si el entrenamiento se corta, queda lo hecho
        pd.DataFrame(historial).to_csv(carpeta / "historial.csv", index=False)
        dibujar_curvas(pd.DataFrame(historial), carpeta / "curvas.png", args.nombre)

        if mejora:
            mejor_auc, sin_mejorar = auc, 0
            guardar(red, ruta_pesos, epoca=epoca, auc_val=auc)
        else:
            sin_mejorar += 1
            if sin_mejorar >= args.paciencia:
                print(f"  Parada temprana: {args.paciencia} epocas sin mejorar")
                break

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
                "fold", "ponderada", "aumentado", "normalizar", "pooling", "epocas", "lote", "lr", "wd",
                "semilla", "pos_weight")})

    y = resultado["umbral_youden"]
    print(f"\nMejor epoca {resultado['mejor_epoca']}: AUC por paciente {mejor_auc:.3f}")
    print(f"Umbral Youden {umbral:.3f}: sensibilidad {y['sensibilidad']:.2f}, "
          f"especificidad {y['especificidad']:.2f}, matriz {y['matriz_confusion']}")
    print(f"AUC por cohorte: { {c: round(v, 3) for c, v in por_cohorte.items()} }")
    print(f"Guardado en {carpeta.relative_to(config.REPO)} y {ruta_pesos.relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
