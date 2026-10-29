#!/usr/bin/env python3
"""Calibra la validacion de la entrada de la web: que las tres fases sean el mismo corte de una RM.

    python -m src.utils.calibrar_entrada

Escribe web/validacion_entrada.json, que leen src/inferencia.py (validar_corte) y la web
(web/js/png.js, validarCorte). Solo usa TRAIN, nunca test.

Los rasgos (src.inferencia.rasgos_corte) se miden en todos los cortes reales de train y en
falsos construidos a partir de ellos: fases de pacientes distintas (de la misma cohorte y de
otra), LATE de otra paciente, cortes distintos de la misma paciente, la misma imagen tres
veces, PRE y EARLY intercambiadas e imagenes que no son mama (ruido, degradados, formas).

Dos niveles por rasgo, solo con los reales (los falsos no fijan nada, solo miden):
  - error: claramente fuera de todo lo visto en reales, con margen: en los rasgos acotados
    (correlaciones, realce), por debajo de min - MARGEN · (mediana - min); en los positivos
    (contraste, textura, tejido), por debajo de min / FACTOR o por encima de max · FACTOR;
  - aviso (zona gris, no bloquea): fuera de los cuantiles CUANTIL_AVISO y 1 - CUANTIL_AVISO.
    El Dice de las mascaras solo avisa (ver RASGOS).
Prioridad: ningun triplete real rechazado. Se calibra con los folds 0-3, se comprueba que el
fold 4 acepta el 100 % por cohorte y que tasa de falsos se detecta por tipo; ademas, cada fold con
los umbrales de los otros cuatro (pacientes nuevas, como en la defensa). Despues se fijan los
umbrales con todo train.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config, inferencia
from src.data.carga import cargar_en_memoria

MARGEN = 1.0  # rasgos acotados (correlaciones, realce): fraccion de (mediana - min)
FACTOR = 2.0  # rasgos positivos (contraste, textura, tejido): min / FACTOR, max * FACTOR
CUANTIL_AVISO = 0.001
SEMILLA = config.DEFECTO["semilla"]
N_FALSOS = 300  # por tipo construido con cortes reales
N_SINTETICOS = 40  # por tipo de imagen que no es mama
#: Por rasgo: lado sospechoso, tipo de margen y si puede dar error o solo aviso. El Dice de las
#: mascaras de tejido solo avisa: una paciente real de spy1 (ISPY1_1193) tiene un LATE cuyo tejido
#: apenas coincide con el de EARLY (Dice 0,25) sin dejar de ser un corte valido.
RASGOS = {
    "tejido": ("bajo", "multiplicativo", True),
    "contraste": ("bajo", "multiplicativo", True),
    "textura": ("ambos", "multiplicativo", True),
    "realce": ("bajo", "aditivo", True),
    "anatomia_pe": ("bajo", "aditivo", True),
    "anatomia_el": ("bajo", "aditivo", True),
    "dice_pe": ("bajo", "aditivo", False),
    "dice_el": ("bajo", "aditivo", False),
}


def tabla_rasgos(imagenes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([inferencia.rasgos_corte(x) for x in imagenes])


def umbrales(reales: pd.DataFrame) -> dict:
    """Limites de error y de aviso de cada rasgo, solo a partir de los tripletes reales."""
    salida = {}
    for nombre, (lado, margen, da_error) in RASGOS.items():
        v = reales[nombre]
        mediana, minimo, maximo = float(v.median()), float(v.min()), float(v.max())
        u = {}
        if lado in ("bajo", "ambos"):
            if da_error:
                u["min_error"] = (
                    minimo / FACTOR if margen == "multiplicativo" else minimo - MARGEN * (mediana - minimo)
                )
            u["min_aviso"] = float(v.quantile(CUANTIL_AVISO))
        if lado in ("alto", "ambos"):
            if da_error:
                u["max_error"] = (
                    maximo * FACTOR if margen == "multiplicativo" else maximo + MARGEN * (maximo - mediana)
                )
            u["max_aviso"] = float(v.quantile(1 - CUANTIL_AVISO))
        salida[nombre] = u
    return {"rasgos": salida}


def veredicto(rasgos: pd.DataFrame, u: dict) -> pd.Series:
    """'error', 'aviso' o 'pasa' por fila, con la misma regla que inferencia.juzgar_rasgos."""

    def una(fila):
        errores, avisos = inferencia.juzgar_rasgos(fila.to_dict(), u)
        return "error" if errores else "aviso" if avisos else "pasa"

    return rasgos.apply(una, axis=1)


def falsos(filas: pd.DataFrame, img: np.ndarray, rng) -> list[tuple[str, np.ndarray]]:
    """Tripletes falsos construidos con cortes reales de `filas` y sinteticos que no son mama."""
    salida = []
    pacientes = filas.patient_id.values
    cohortes = filas.cohorte.values
    indices = np.arange(len(filas))

    def pareja(misma_cohorte: bool):
        while True:
            i, j = rng.choice(indices, 2, replace=False)
            if pacientes[i] != pacientes[j] and (cohortes[i] == cohortes[j]) == misma_cohorte:
                return i, j

    for _ in range(N_FALSOS):
        i, j = pareja(True)
        salida.append(("pacientes distintas (misma cohorte)", np.stack([img[i, 0], img[j, 1], img[j, 2]])))
        i, j = pareja(False)
        salida.append(("pacientes distintas (otra cohorte)", np.stack([img[i, 0], img[j, 1], img[j, 2]])))
        i, j = pareja(True)
        salida.append(("LATE de otra paciente", np.stack([img[i, 0], img[i, 1], img[j, 2]])))
        i = rng.choice(indices)
        salida.append(("la misma imagen tres veces", np.stack([img[i, 1]] * 3)))
        salida.append(("PRE y EARLY intercambiadas", np.stack([img[i, 1], img[i, 0], img[i, 2]])))
    por_paciente = filas.groupby("patient_id").indices
    con_varios = [v for v in por_paciente.values() if len(v) >= 2]
    for k in rng.choice(len(con_varios), min(N_FALSOS, len(con_varios)), replace=False):
        i, j = rng.choice(con_varios[k], 2, replace=False)
        salida.append(("otro corte de la misma paciente", np.stack([img[i, 0], img[j, 1], img[j, 2]])))

    yy, xx = np.mgrid[0 : inferencia.LADO, 0 : inferencia.LADO]
    for _ in range(N_SINTETICOS):
        base = rng.uniform(40, 160)
        r = rng.uniform(30, 100)
        disco = (xx - rng.uniform(60, 196)) ** 2 + (yy - rng.uniform(60, 196)) ** 2 < r**2
        sinteticos = {
            "ruido uniforme": rng.integers(0, 256, (3, 256, 256)),
            "ruido gaussiano": rng.normal(base, rng.uniform(10, 60), (3, 256, 256)),
            "imagen constante": np.full((3, 256, 256), base),
            "degradado": np.stack([xx * rng.uniform(0.3, 1)] * 3),
            "degradado que realza": np.stack([xx * f for f in (0.6, 0.8, 0.85)]),
            "disco que realza": np.stack([disco * v for v in (base, base + 60, base + 70)]),
            "tablero": np.stack([((xx // 16 + yy // 16) % 2) * base] * 3),
        }
        for nombre, x in sinteticos.items():
            salida.append((f"no es mama: {nombre}", np.clip(np.asarray(x), 0, 255).astype(np.uint8)))
    return salida


def evaluar_falsos(lista, u: dict) -> dict:
    t = pd.DataFrame([{"tipo": tipo, **inferencia.rasgos_corte(x)} for tipo, x in lista])
    t["veredicto"] = veredicto(t.drop(columns="tipo"), u)
    resumen = {}
    for tipo, g in t.groupby("tipo", sort=False):
        resumen[tipo] = {
            "n": int(len(g)),
            "error": float((g.veredicto == "error").mean()),
            "aviso": float((g.veredicto == "aviso").mean()),
            "pasa": float((g.veredicto == "pasa").mean()),
        }
    return resumen


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    uc = config.utils()
    s = uc.cargar_samples()
    train = s[s.split == "train"].reset_index(drop=True)  # nunca test
    train["cohorte"] = train.patient_id.map(uc.cargar_patients().set_index("pid").dataset)
    img = cargar_en_memoria(train, "train").numpy()
    reales = tabla_rasgos(img)
    rng = np.random.default_rng(SEMILLA)

    # 1) Calibrar con los folds 0-3 y comprobar con el fold 4 (reales y falsos de sus cortes)
    cal, comp = (train.fold != 4).values, (train.fold == 4).values
    u03 = umbrales(reales[cal])
    v4 = veredicto(reales[comp].reset_index(drop=True), u03)
    fold4 = {}
    for c in ("spy1", "spy2", "duke"):
        m = (train.cohorte[comp] == c).values
        fold4[c] = {
            "cortes": int(m.sum()),
            "aceptados": float((v4[m] != "error").mean()),
            "con_aviso": float((v4[m] == "aviso").mean()),
        }
    filas4 = train[comp].reset_index(drop=True)
    falsos4 = evaluar_falsos(falsos(filas4, img[comp], rng), u03)

    # 2) Cada fold con los umbrales de los otros cuatro: pacientes nuevas, como en la defensa
    cada_fold = {}
    for k in range(5):
        cal_k, comp_k = (train.fold != k).values, (train.fold == k).values
        v = veredicto(reales[comp_k].reset_index(drop=True), umbrales(reales[cal_k]))
        cada_fold[str(k)] = {
            "cortes": int(comp_k.sum()),
            "rechazados": int((v == "error").sum()),
            "con_aviso": int((v == "aviso").sum()),
        }

    # 3) Umbrales definitivos con todo train
    final = umbrales(reales)
    v_todo = veredicto(reales, final)
    salida = {
        "version": 1,
        "aviso": "Generado por src/utils/calibrar_entrada.py con tripletes de TRAIN; no se edita a mano. "
        "Lo leen src/inferencia.py (validar_corte) y web/js/png.js (validarCorte).",
        "tejido": inferencia.TEJIDO,
        "bloque": inferencia.BLOQUE,
        "margen": MARGEN,
        "factor": FACTOR,
        "cuantil_aviso": CUANTIL_AVISO,
        **final,
        "calibracion": {
            "cortes_train": int(len(train)),
            "train_con_umbrales_finales": {
                "aceptados": float((v_todo != "error").mean()),
                "con_aviso": float((v_todo == "aviso").mean()),
            },
            "fold4_con_umbrales_de_folds_0_3": fold4,
            "cada_fold_con_umbrales_de_los_otros_cuatro": cada_fold,
            "falsos_del_fold4_con_umbrales_de_folds_0_3": falsos4,
        },
    }
    ruta = inferencia.UMBRALES_ENTRADA
    ruta.write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Fold 4 (umbrales de los folds 0-3), cortes reales aceptados por cohorte:")
    for c, f in fold4.items():
        print(
            f"  {c:<5} {f['cortes']:>5} cortes: aceptados {100 * f['aceptados']:.2f} %, "
            f"con aviso {100 * f['con_aviso']:.2f} %"
        )
    print("Falsos construidos con el fold 4: rechazados (error) / avisados / pasan")
    for tipo, f in falsos4.items():
        print(
            f"  {tipo:<42} n={f['n']:>4}  {100 * f['error']:5.1f} % / {100 * f['aviso']:5.1f} % / "
            f"{100 * f['pasa']:5.1f} %"
        )
    print(
        "Cada fold con los umbrales de los otros cuatro: "
        + ", ".join(f"fold {k} {f['rechazados']} rechazados de {f['cortes']}" for k, f in cada_fold.items())
    )
    print(
        f"Todo train con los umbrales finales: aceptados {100 * (v_todo != 'error').mean():.2f} %, "
        f"con aviso {100 * (v_todo == 'aviso').mean():.2f} %"
    )
    print(f"Escrito {ruta.relative_to(config.REPO)}")
    if any(f["aceptados"] < 1 for f in fold4.values()) or any(f["rechazados"] for f in cada_fold.values()):
        sys.exit("ATENCION: el fold 4 tiene cortes reales rechazados: hay que ampliar el margen.")


if __name__ == "__main__":
    main()
