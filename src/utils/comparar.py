#!/usr/bin/env python3
"""Comparacion en VALIDACION de los candidatos a modelo final (iteracion 9).

    python -m src.utils.comparar

Tres configuraciones de 5 folds (`figuras.CINCO_FOLDS`): el congelado con perdida
normal, el congelado con perdida ponderada y la atencion supervisada con la caja.
Solo lee results/ (y patients.csv para la cohorte). No entrena y no usa test.

Para cada una, media ± desviacion entre folds de las metricas por epoca, AUC global
fuera de fold con IC 95 % y por cohorte (`analisis_cv.analizar`), Brier por paciente,
la atencion en la caja (si la hay) y la comparacion pareada con el congelado normal:
diferencia de AUC medio fold a fold y diferencia del AUC global con IC 95 % por
bootstrap pareado de pacientes (percentiles dentro de cada fold). Al final aplica la
regla fijada antes en docs/iteraciones.md (fila 9). Escribe results/comparacion/resumen.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config
from src.utils import figuras
from src.utils.analisis_cv import analizar, auc, media_desviacion, por_paciente

REFERENCIA = "congelado"
CANDIDATOS = ("congelado", "congelado_ponderada", "atencion_supervisada")
FICHEROS = ("config.json", "historial.csv", "metricas.json", "predicciones_val.csv")
#: Regla fijada antes de ver it9 (docs/iteraciones.md, fila 9)
REGLA_CANDIDATA = "atencion_supervisada"
REGLA_UMBRAL = 0.581


def faltan(nombres) -> list[str]:
    return [f"{n}/{f}" for n in nombres for f in FICHEROS if not (config.RESULTADOS / n / f).exists()]


def brier(prob, pcr) -> float:
    """Error cuadratico medio de la probabilidad frente a la etiqueta (0 = perfecto)."""
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(pcr, dtype=float)) ** 2))


def diferencia_pareada(a: pd.DataFrame, b: pd.DataFrame, remuestreos: int, semilla: int) -> dict:
    """AUC(b) - AUC(a) sobre las MISMAS pacientes, con IC 95 % remuestreando pacientes
    (la misma muestra para los dos modelos). Entradas: columnas percentil y pcr por paciente."""
    juntas = a[["percentil", "pcr"]].join(b[["percentil"]], rsuffix="_b", how="inner")
    pa, pb, y = juntas.percentil.values, juntas.percentil_b.values, juntas.pcr.values
    valor = auc(pb, y) - auc(pa, y)
    rng = np.random.default_rng(semilla)
    n = len(juntas)
    muestras = []
    for _ in range(remuestreos):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == 2:
            muestras.append(auc(pb[i], y[i]) - auc(pa[i], y[i]))
    return {"diferencia": float(valor), "ic95": [float(np.percentile(muestras, 2.5)),
                                                  float(np.percentile(muestras, 97.5))], "pacientes": int(n)}


def por_fold(nombres: list[str]) -> pd.DataFrame:
    """Una fila por fold con las metricas de historial.csv y metricas.json."""
    filas = []
    for n in nombres:
        e = figuras.leer_ejecucion(n)
        h, m = e["historial"], e["metricas"]
        filas.append({
            "fold": e["config"]["fold"], "ejecucion": n, "ponderada": bool(e["config"]["ponderada"]),
            "auc_medio": figuras.auc_medio(h), "auc_medio_1_15": figuras.auc_medio(h, 15),
            "mismo_hospital_medio": figuras.auc_medio(h, columna="auc_val_mismo_hospital"),
            "auc_elegido": m["auc_val_media"], "epoca_elegida": m["mejor_epoca"],
            "diferencia_train_val": m["diferencia_train_val"],
            "perdida_val_min": float(h.perdida_val.min()), "perdida_constante": figuras.constante_de(e),
            "masa_caja_15": figuras.en_epoca(h, "masa_caja_val", 15),
            "fraccion_caja_15": figuras.en_epoca(h, "fraccion_caja_val", 15),
        })
    return pd.DataFrame(filas).set_index("fold").sort_index()


def oof(nombres: list[str]) -> pd.DataFrame:
    """Probabilidad por paciente fuera de fold (prob y percentil dentro del fold)."""
    uc = config.utils()
    pacientes = uc.cargar_patients().set_index("pid")
    entrenamiento = pd.concat(uc.particion(uc.cargar_samples(), fold_val=0))
    return pd.concat([por_paciente(n, pacientes, entrenamiento) for n in nombres])


def md(valores) -> dict:
    v = [x for x in valores if not np.isnan(x)]
    return media_desviacion(v) if v else {"media": float("nan"), "desviacion": float("nan")}


def comparar(remuestreos: int, semilla: int) -> dict:
    grupos = {k: figuras.CINCO_FOLDS[k] for k in CANDIDATOS}
    perdidos = faltan([n for ns in grupos.values() for n in ns])
    if perdidos:
        raise FileNotFoundError("Faltan ficheros:\n  " + "\n  ".join(perdidos))

    tablas = {k: por_fold(v) for k, v in grupos.items()}
    analisis = {k: analizar(v, remuestreos, semilla) for k, v in grupos.items()}
    pacientes = {k: oof(v) for k, v in grupos.items()}
    ref = tablas[REFERENCIA]

    resumen = {}
    for k, t in tablas.items():
        g = analisis[k]["global"]
        r = {
            "ejecuciones": grupos[k],
            "por_fold": t.reset_index().to_dict(orient="records"),
            **{c: md(t[c]) for c in ("auc_medio", "auc_medio_1_15", "mismo_hospital_medio", "auc_elegido",
                                     "epoca_elegida", "diferencia_train_val", "perdida_val_min",
                                     "perdida_constante")},
            "folds_con_mismo_hospital": [int(f) for f in t.index[t.mismo_hospital_medio.notna()]],
            "auc_global": g["auc"], "mismo_hospital_global": g["mismo_hospital"],
            "por_cohorte": g["por_cohorte"],
        }
        p = pacientes[k]
        if not t.ponderada.any():
            r["brier"] = brier(p.prob, p.pcr)
            r["brier_constante"] = brier(np.full(len(p), p.pcr.mean()), p.pcr)
        if t.masa_caja_15.notna().any():
            r["masa_caja_15"], r["fraccion_caja_15"] = md(t.masa_caja_15), md(t.fraccion_caja_15)
        if k != REFERENCIA:
            dif = (t.auc_medio - ref.auc_medio).dropna()
            comunes = t.index[t.mismo_hospital_medio.notna() & ref.mismo_hospital_medio.notna()]
            r["frente_al_congelado"] = {
                "dif_auc_medio_por_fold": dif.round(4).to_dict(),
                "dif_auc_medio": md(dif),
                "dif_mismo_hospital_medio": {
                    "folds": [int(f) for f in comunes],
                    **md(t.mismo_hospital_medio[comunes] - ref.mismo_hospital_medio[comunes]),
                },
                "dif_auc_global": diferencia_pareada(pacientes[REFERENCIA], p, remuestreos, semilla),
            }
        resumen[k] = r

    c = resumen[REGLA_CANDIDATA]
    auc_c = c["auc_medio"]["media"]
    dif_mh = c["frente_al_congelado"]["dif_mismo_hospital_medio"]["media"]
    cumple_auc, cumple_mh = auc_c >= REGLA_UMBRAL, dif_mh >= 0
    resumen["regla"] = {
        "texto": "la atencion supervisada sustituye al congelado solo si su AUC medio de validacion en 5 "
                 "folds es >= 0,581 y su AUC del mismo hospital por epoca no es menor (fila 9, fijada antes)",
        "auc_medio": auc_c, "cumple_auc": bool(cumple_auc),
        "dif_mismo_hospital": dif_mh,
        "folds_mismo_hospital": c["frente_al_congelado"]["dif_mismo_hospital_medio"]["folds"],
        "cumple_mismo_hospital": bool(cumple_mh),
        "queda": REGLA_CANDIDATA if (cumple_auc and cumple_mh) else REFERENCIA,
    }
    return resumen


def imprimir(r: dict) -> None:
    def m(x, d=3):
        return f"{x['media']:.{d}f} ± {x['desviacion']:.{d}f}" if not np.isnan(x["media"]) else "—"

    def ic(x, clave="auc"):
        return f"{x[clave]:.3f} [{x['ic95'][0]:.3f}-{x['ic95'][1]:.3f}]"

    nombres = list(CANDIDATOS)
    filas = [
        ("AUC val medio (todas las epocas)", lambda k: m(r[k]["auc_medio"])),
        ("AUC val medio (epocas 1-15)", lambda k: m(r[k]["auc_medio_1_15"])),
        ("Mismo hospital medio por epoca",
         lambda k: f"{m(r[k]['mismo_hospital_medio'])} "
                   f"(f{''.join(map(str, r[k]['folds_con_mismo_hospital']))})"),
        ("AUC elegido (inflado)", lambda k: m(r[k]["auc_elegido"])),
        ("Epoca elegida", lambda k: m(r[k]["epoca_elegida"], 1)),
        ("Diferencia train-val (elegida)", lambda k: m(r[k]["diferencia_train_val"])),
        ("Perdida val minima", lambda k: m(r[k]["perdida_val_min"])),
        ("  constante (misma perdida)", lambda k: m(r[k]["perdida_constante"])),
        ("AUC global fuera de fold, IC 95 %", lambda k: ic(r[k]["auc_global"])),
        ("Mismo hospital global, IC 95 %", lambda k: ic(r[k]["mismo_hospital_global"])),
        *[(f"  {c} ({r[nombres[0]]['por_cohorte'][c]['pacientes']} pac.)",
           (lambda c: lambda k: ic(r[k]["por_cohorte"][c]))(c)) for c in ("spy1", "spy2", "duke")],
        ("Brier por paciente (normal)",
         lambda k: f"{r[k]['brier']:.4f} (constante {r[k]['brier_constante']:.4f})"
         if "brier" in r[k] else "— (no comparable)"),
        ("Masa en la caja / area (ep. 15)",
         lambda k: f"{m(r[k]['masa_caja_15'], 2)} / {m(r[k]['fraccion_caja_15'], 2)}"
         if "masa_caja_15" in r[k] else "—"),
    ]
    ancho = 31
    print("\nCOMPARACION EN VALIDACION (5 folds; media ± desviacion entre folds)")
    print(f"{'':<34}" + "".join(f"{n:>{ancho}}" for n in nombres))
    for titulo, f in filas:
        print(f"{titulo:<34}" + "".join(f"{f(k):>{ancho}}" for k in nombres))

    print("\nFRENTE AL CONGELADO NORMAL (pareado)")
    for k in nombres[1:]:
        c = r[k]["frente_al_congelado"]
        folds = ", ".join(f"f{f} {v:+.3f}" for f, v in c["dif_auc_medio_por_fold"].items())
        d = c["dif_auc_global"]
        print(f"  {k}:")
        print(f"    dif. AUC medio por fold: {folds}  ->  {m(c['dif_auc_medio'])}")
        print(f"    dif. mismo hospital medio por epoca (folds {c['dif_mismo_hospital_medio']['folds']}): "
              f"{m(c['dif_mismo_hospital_medio'])}")
        print(f"    dif. AUC global fuera de fold: {d['diferencia']:+.3f} [{d['ic95'][0]:+.3f} a "
              f"{d['ic95'][1]:+.3f}] (bootstrap pareado, {d['pacientes']} pac.)")

    g = r["regla"]
    print("\nREGLA (docs/iteraciones.md, fila 9): " + g["texto"])
    print(f"  AUC medio de la atencion supervisada {g['auc_medio']:.3f} >= {REGLA_UMBRAL}: "
          f"{'si' if g['cumple_auc'] else 'no'}")
    print(f"  mismo hospital por epoca no menor (folds {g['folds_mismo_hospital']}): "
          f"{g['dif_mismo_hospital']:+.3f} -> {'si' if g['cumple_mismo_hospital'] else 'no'}")
    print(f"  QUEDA: {g['queda']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remuestreos", type=int, default=2000)
    p.add_argument("--semilla", type=int, default=config.DEFECTO["semilla"])
    a = p.parse_args()
    try:
        resumen = comparar(a.remuestreos, a.semilla)
    except FileNotFoundError as error:
        sys.exit(str(error))
    destino = config.RESULTADOS / "comparacion"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float), encoding="utf-8")
    imprimir(resumen)
    print(f"\nGuardado en {(destino / 'resumen.json').relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
