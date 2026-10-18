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
    python -m src.training.entrenar --aumentado --pooling atencion --peso-caja 0.2

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
from src.data.carga import OBJETIVOS, aplicar_objetivo, aumentar, cargar_en_memoria, lotes, submuestra
from src.data.tumor import cajas_por_corte, en_rejilla, masa_en_caja, mascaras, perdida_caja
from src.models.cnn import POOLINGS, CNNpCR, guardar, parametros
from src.utils.evaluacion import auc_mismo_hospital, dibujar_curvas, perdida_media, umbral_youden
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
    p.add_argument("--reducir", type=int, default=1,
                   help="reduce la imagen por este factor dentro de la red (2: 256 -> 128 px)")
    p.add_argument("--normalizar", action="store_true",
                   help="estandariza cada corte (media 0, desviacion 1) dentro de la red")
    p.add_argument("--pooling", choices=POOLINGS, default="media",
                   help="como se resume el mapa final antes de la capa lineal")
    p.add_argument("--objetivo", choices=OBJETIVOS, default="pcr",
                   help="que se aprende: pcr, una cohorte (comprueba el codigo: se ve en la "
                        "imagen) o un subtipo tumoral (comprueba si la red ve el tumor)")
    p.add_argument("--filtros", default="16,32,64,128",
                   help="filtros de cada bloque conv, separados por comas (n bloques = n valores)")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epocas", type=int, default=D["epocas"])
    p.add_argument("--paciencia", type=int, default=D["paciencia"],
                   help="epocas sin mejorar el AUC suavizado antes de parar")
    p.add_argument("--min-epocas", type=int, default=D["min_epocas"],
                   help="nunca se para antes de esta epoca")
    p.add_argument("--suavizado", type=int, default=D["suavizado"],
                   help="la epoca se elige por la media del AUC de validacion de las ultimas N")
    p.add_argument("--lote", type=int, default=D["lote"])
    p.add_argument("--lr", type=float, default=D["lr"])
    p.add_argument("--wd", type=float, default=D["wd"], help="weight decay de AdamW")
    p.add_argument("--semilla", type=int, default=D["semilla"])
    p.add_argument("--peso-caja", type=float, default=0.0,
                   help="lambda de la atencion supervisada: perdida = BCE + lambda * L_caja, con "
                        "L_caja = -log(atencion dentro de la caja del tumor). Solo al entrenar (la "
                        "app no la necesita); exige --pooling atencion. 0 = sin supervisar")
    p.add_argument("--rapido", action="store_true",
                   help="prueba del circuito: 40 pacientes de train, 20 de validacion, 2 epocas")
    a = p.parse_args()

    a.filtros = tuple(int(f) for f in a.filtros.split(","))
    if a.peso_caja < 0:
        p.error("--peso-caja no puede ser negativo")
    if a.peso_caja > 0 and a.pooling != "atencion":
        p.error("--peso-caja > 0 exige --pooling atencion (supervisa sus pesos)")
    if a.rapido:
        a.epocas, a.min_epocas = 2, 1
    if a.nombre is None:
        partes = [] if a.objetivo == "pcr" else [f"obj{a.objetivo}"]
        partes.append("ponderada" if a.ponderada else "normal")
        partes += [n for n, activa in (("realce", a.realce), ("sinlate", a.sin_late),
                                       ("aum", a.aumentado), ("norm", a.normalizar)) if activa]
        if a.pooling != "media":
            partes.append(a.pooling)
        if a.filtros != (16, 32, 64, 128):
            partes.append("f" + "-".join(map(str, a.filtros)))
        if a.reducir != 1:
            partes.append(f"red{a.reducir}")
        if a.dropout != 0.5:
            partes.append(f"do{a.dropout:g}")
        if a.peso_caja > 0:
            partes.append(f"caja{a.peso_caja:g}")
        partes.append(f"f{a.fold}")
        a.nombre = ("rapido_" if a.rapido else "") + "_".join(partes)
    return a


# --------------------------------------------------------------------------- #
# Entrenamiento y evaluacion
# --------------------------------------------------------------------------- #

def una_epoca(red, imagenes, etiquetas, cajas, criterio, optimizador, args, dispositivo,
              epoca) -> tuple[float, float]:
    """Devuelve la BCE media (comparable con --peso-caja 0) y la L_caja media (NaN sin ella)."""
    red.train()
    total, total_caja, vistos = 0.0, 0.0, 0
    barra = tqdm(lotes(len(imagenes), args.lote, desordenar=True), total=-(-len(imagenes) // args.lote),
                 desc=f"  epoca {epoca:2d}", leave=False, ncols=90)
    for indices in barra:
        x = imagenes[indices].to(dispositivo).float() / 255
        y = etiquetas[indices].to(dispositivo)
        optimizador.zero_grad()
        if args.peso_caja > 0:
            # La mascara de la caja va como canal extra por `aumentar`: el mismo volteo
            # y giro que las fases, y se separa antes de entrar en la red.
            m = mascaras(cajas[indices].to(dispositivo))
            if args.aumentado:
                x, m = aumentar(torch.cat([x, m], 1)).split([x.shape[1], 1], dim=1)
            logits, pesos = red.con_atencion(x)
            perdida = criterio(logits, y)
            l_caja = perdida_caja(pesos, en_rejilla(m, *pesos.shape[1:]))
            (perdida + args.peso_caja * l_caja).backward()
            total_caja += l_caja.item() * len(indices)
        else:
            if args.aumentado:
                x = aumentar(x)
            perdida = criterio(red(x), y)
            perdida.backward()
        optimizador.step()
        total += perdida.item() * len(indices)
        vistos += len(indices)
        barra.set_postfix(perdida=f"{total / vistos:.4f}")
    return total / vistos, (total_caja / vistos if args.peso_caja > 0 else float("nan"))


@torch.no_grad()
def predecir(red, imagenes, dispositivo) -> np.ndarray:
    """Probabilidad por corte, en el mismo orden que las filas (sin desordenar)."""
    red.eval()
    salidas = [torch.sigmoid(red(imagenes[i].to(dispositivo).float() / 255)).cpu()
               for i in lotes(len(imagenes), 128, desordenar=False)]
    return torch.cat(salidas).numpy()


@torch.no_grad()
def atencion_en_caja(red, imagenes, cajas, dispositivo) -> tuple[float, float]:
    """Donde mira la atencion, en los cortes con caja (sin aumentado): masa media de
    atencion dentro de la caja y fraccion media del mapa final que ocupa la caja.
    Si la masa no supera a la fraccion, la atencion no prefiere el tumor."""
    red.eval()
    masas, fracciones = [], []
    for i in lotes(len(imagenes), 128, desordenar=False):
        _, pesos = red.con_atencion(imagenes[i].to(dispositivo).float() / 255)
        rejilla = en_rejilla(mascaras(cajas[i].to(dispositivo)), *pesos.shape[1:])
        con_caja = rejilla.flatten(1).sum(1) > 0
        masas.append(masa_en_caja(pesos, rejilla)[con_caja].cpu())
        fracciones.append(rejilla.flatten(1).mean(1)[con_caja].cpu())
    masas, fracciones = torch.cat(masas), torch.cat(fracciones)
    if len(masas) == 0:
        return float("nan"), float("nan")
    return masas.mean().item(), fracciones.mean().item()


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
    if args.objetivo != "pcr":
        # A partir de aqui la columna `pCR` es la etiqueta que se aprende (ver aplicar_objetivo)
        pacientes = uc.cargar_patients().set_index("pid")
        antes = entreno.patient_id.nunique() + valida.patient_id.nunique()
        entreno = aplicar_objetivo(entreno, args.objetivo, pacientes)
        valida = aplicar_objetivo(valida, args.objetivo, pacientes)
        sin_dato = antes - entreno.patient_id.nunique() - valida.patient_id.nunique()
        print(f"Objetivo: '{args.objetivo}' en lugar de pCR (diagnostico; la red solo ve la imagen)"
              + (f", {sin_dato} pacientes sin el dato quitadas" if sin_dato else ""))
    if args.rapido:
        entreno = submuestra(entreno, 40, args.semilla)
        valida = submuestra(valida, 20, args.semilla)
    entreno, valida = entreno.reset_index(drop=True), valida.reset_index(drop=True)

    print("Cargando imagenes en memoria:")
    img_entreno = cargar_en_memoria(entreno, "train")
    img_valida = cargar_en_memoria(valida, "validacion")
    y_entreno = torch.tensor(entreno.pCR.values, dtype=torch.float32)
    y_valida = torch.tensor(valida.pCR.values, dtype=torch.float32)
    pacientes = uc.cargar_patients().set_index("pid")
    cohorte_paciente = pacientes.dataset
    cajas_entreno = cajas_por_corte(entreno, pacientes)
    cajas_valida = cajas_por_corte(valida, pacientes)
    if args.pooling == "atencion":
        print(f"  cortes con caja del tumor: train {(cajas_entreno[:, 0] >= 0).float().mean():.0%}, "
              f"validacion {(cajas_valida[:, 0] >= 0).float().mean():.0%}")

    red = CNNpCR(realce=args.realce, sin_late=args.sin_late, normalizar=args.normalizar, reducir=args.reducir,
                 pooling=args.pooling, filtros=args.filtros, dropout=args.dropout).to(dispositivo)
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

    # La epoca elegida es la de mejor AUC de validacion SUAVIZADO (media de las ultimas
    # `suavizado` epocas): un pico suelto de una epoca es sobre todo ruido (±0,04 con
    # ~220 pacientes). Y no se para antes de `min_epocas`, porque el AUC sube despacio.
    historial, mejor_auc, sin_mejorar, epoca_elegida = [], -1.0, 0, 1
    for epoca in range(1, args.epocas + 1):
        inicio = time.perf_counter()
        perdida_train, perdida_caja_train = una_epoca(red, img_entreno, y_entreno, cajas_entreno, criterio,
                                                      optimizador, args, dispositivo, epoca)
        probs = predecir(red, img_valida, dispositivo)
        metricas = uc.evaluar_por_paciente(probs, valida, umbral=0.5, metodo="mean")
        auc = metricas["auc"]
        # AUC de entrenamiento con la red en modo evaluacion y sin aumentado: comparable con
        # el de validacion. Si sube a 1 mientras validacion no sube, memoriza.
        probs_train = predecir(red, img_entreno, dispositivo)
        auc_train = uc.evaluar_por_paciente(probs_train, entreno, umbral=0.5, metodo="mean")["auc"]
        recientes = [h["auc_val"] for h in historial[-(args.suavizado - 1):]] if args.suavizado > 1 else []
        auc_suave = float(np.mean(recientes + [auc]))
        planificador.step(auc_suave)
        # Sin la ventaja de reconocer el hospital: solo pares de pacientes de la misma cohorte
        por_paciente = uc.agregar_por_paciente(valida.patient_id.values, probs).join(
            valida.groupby("patient_id").pCR.first())
        auc_hospital = auc_mismo_hospital(por_paciente.prob, por_paciente.pCR,
                                          por_paciente.index.map(cohorte_paciente))
        extra = {"auc_val_mismo_hospital": auc_hospital}
        if args.peso_caja > 0:
            extra["perdida_caja_train"] = perdida_caja_train
        if args.pooling == "atencion":
            extra["masa_caja_val"], extra["fraccion_caja_val"] = atencion_en_caja(
                red, img_valida, cajas_valida, dispositivo)

        historial.append({"epoca": epoca, "perdida_train": perdida_train,
                          "perdida_val": perdida_media(probs, y_valida, criterio_cpu),
                          "auc_train": auc_train, "auc_val": auc, "auc_val_suave": auc_suave,
                          "sensibilidad_05": metricas["sensibilidad"],
                          "especificidad_05": metricas["especificidad"],
                          **extra,
                          "lr": optimizador.param_groups[0]["lr"],
                          "segundos": time.perf_counter() - inicio})
        mejora = auc_suave > mejor_auc
        print(f"  epoca {epoca:2d}  perdida {perdida_train:.4f}  AUC train {auc_train:.3f}  "
              f"val {auc:.3f} (suave {auc_suave:.3f}, mismo hospital {auc_hospital:.3f})"
              + (f"  atencion en la caja {extra['masa_caja_val']:.2f} (area {extra['fraccion_caja_val']:.2f})"
                 if "masa_caja_val" in extra else "")
              + f"  ({historial[-1]['segundos']:.0f} s){'  *' if mejora else ''}")
        # Cada epoca, no solo al final: si el entrenamiento se corta, queda lo hecho
        pd.DataFrame(historial).to_csv(carpeta / "historial.csv", index=False)
        if mejora:
            mejor_auc, sin_mejorar, epoca_elegida = auc_suave, 0, epoca
            guardar(red, ruta_pesos, epoca=epoca, auc_val=auc)
        else:
            sin_mejorar += 1
        dibujar_curvas(pd.DataFrame(historial), carpeta / "curvas.png", args.nombre, epoca_elegida)
        if not mejora and epoca >= args.min_epocas and sin_mejorar >= args.paciencia:
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
    cohorte = valida.patient_id.map(cohorte_paciente)
    por_cohorte = {}
    for c in sorted(cohorte.unique()):
        mascara = (cohorte == c).values
        if valida[mascara].pCR.nunique() == 2:
            por_cohorte[c] = uc.evaluar_por_paciente(probs[mascara], valida[mascara], metodo="mean")["auc"]

    resultado = {
        "mejor_epoca": int(paquete["info"]["epoca"]),
        "auc_val_media": float(paquete["info"]["auc_val"]),
        "auc_val_suave": mejor_auc,
        "auc_train": historial[epoca_elegida - 1]["auc_train"],
        "diferencia_train_val": historial[epoca_elegida - 1]["auc_train"] - paquete["info"]["auc_val"],
        "epocas_entrenadas": len(historial),
        "auc_por_agregacion": agregaciones,
        "auc_por_cohorte": por_cohorte,
        "umbral_05": uc.evaluar_por_paciente(probs, valida, umbral=0.5, metodo="mean"),
        "umbral_youden": uc.evaluar_por_paciente(probs, valida, umbral=umbral, metodo="mean"),
    }
    (carpeta / "metricas.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")

    # Los pesos llevan lo que la app tiene que mostrar
    guardar(red, ruta_pesos, epoca=resultado["mejor_epoca"], auc_val=resultado["auc_val_media"],
            umbral=umbral,
            metodo_agregacion="mean", entrenamiento={k: configuracion[k] for k in (
                "objetivo", "fold", "ponderada", "aumentado", "normalizar", "reducir", "pooling", "filtros",
                "dropout", "epocas", "lote",
                "lr", "wd", "semilla", "pos_weight", "peso_caja")})

    y = resultado["umbral_youden"]
    print(f"\nEpoca elegida {resultado['mejor_epoca']} de {len(historial)}: "
          f"AUC val {resultado['auc_val_media']:.3f} (suave {mejor_auc:.3f}), "
          f"train {resultado['auc_train']:.3f}, diferencia {resultado['diferencia_train_val']:.3f}")
    print(f"Umbral Youden {umbral:.3f}: sensibilidad {y['sensibilidad']:.2f}, "
          f"especificidad {y['especificidad']:.2f}, matriz {y['matriz_confusion']}")
    print(f"AUC por cohorte: { {c: round(v, 3) for c, v in por_cohorte.items()} }")
    print(f"Guardado en {carpeta.relative_to(config.REPO)} y {ruta_pesos.relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
