#!/usr/bin/env python3
"""¿Cuanto del AUC de validacion es solo «ordenar por hospital»? (analisis 5b)

Lee las predicciones fuera de fold que ya existen de una configuracion (de 2 a 5
folds) y no entrena nada:

    python -m src.utils.analisis_cv --config red128 it5_red128_f0 it5_red128_f1 ...

Para la configuracion calcula, por paciente (media de sus cortes):

  1. AUC de cada fold y global fuera de fold. Para el global las probabilidades se
     pasan antes a percentiles dentro de cada fold: son modelos distintos, cada uno
     con su escala.
  2. El mismo AUC dentro de cada cohorte (spy1, spy2, duke).
  3. AUC entre pares del mismo hospital (`auc_mismo_hospital`): un modelo que solo
     conozca la cohorte da exactamente 0,5.
  4. Referencia «solo cohorte»: cada paciente de validacion recibe la tasa de pCR de
     su cohorte en el train de ese fold.
  5. AUC medio de validacion por epoca en la ventana comun 1-15 y en todas las epocas.

Intervalos al 95 % por bootstrap de pacientes. Escribe
results/analisis_<config>/resumen.json y una tabla en consola. Test NO se usa.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.utils.evaluacion import auc_mismo_hospital

COHORTES = ("spy1", "spy2", "duke")
VENTANA = 15


def auc(prob, pcr) -> float:
    pcr = np.asarray(pcr)
    return float(roc_auc_score(pcr, prob)) if len(np.unique(pcr)) == 2 else float("nan")


def con_ic(datos: pd.DataFrame, metrica, remuestreos: int, semilla: int) -> dict:
    """Valor de `metrica(datos)` e IC 95 % remuestreando pacientes (filas) con reemplazo.
    Los remuestreos en que la metrica no existe (falta una clase) se descartan."""
    valor = metrica(datos)
    rng = np.random.default_rng(semilla)
    n = len(datos)
    muestras = np.array([metrica(datos.iloc[rng.integers(0, n, n)]) for _ in range(remuestreos)])
    muestras = muestras[~np.isnan(muestras)]
    if np.isnan(valor) or len(muestras) == 0:
        return {"auc": valor, "ic95": [float("nan"), float("nan")], "pacientes": n}
    return {"auc": valor, "ic95": [float(np.percentile(muestras, 2.5)), float(np.percentile(muestras, 97.5))],
            "pacientes": n}


def por_paciente(nombre: str, pacientes: pd.DataFrame, entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paciente de validacion: prob (media de cortes), percentil en el fold,
    pCR, cohorte y la prediccion «solo cohorte» de ese fold."""
    carpeta = config.RESULTADOS / nombre
    fold = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))["fold"]
    cortes = pd.read_csv(carpeta / "predicciones_val.csv")
    tabla = cortes.groupby("patient_id").agg(prob=("prob", "mean"), pcr=("pCR", "first"))
    tabla["cohorte"] = tabla.index.map(pacientes.dataset)
    tabla["percentil"] = tabla.prob.rank(pct=True)

    # Tasa de pCR de cada cohorte en las pacientes de train de ESTE fold (los otros 4)
    train = entrenamiento[entrenamiento.fold != fold].groupby("patient_id").pCR.first().to_frame()
    train["cohorte"] = train.index.map(pacientes.dataset)
    tabla["solo_cohorte"] = tabla.cohorte.map(train.groupby("cohorte").pCR.mean())
    tabla["solo_cohorte_percentil"] = tabla.solo_cohorte.rank(pct=True)
    return tabla.assign(fold=fold, ejecucion=nombre)


def auc_por_epoca(nombre: str) -> dict:
    h = pd.read_csv(config.RESULTADOS / nombre / "historial.csv")
    return {"ventana_1_15": float(h.auc_val[h.epoca <= VENTANA].mean()), "todas": float(h.auc_val.mean()),
            "epocas": int(len(h))}


def media_desviacion(valores) -> dict:
    v = np.asarray(valores, dtype=float)
    return {"media": float(v.mean()), "desviacion": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}


def analizar(nombres: list[str], remuestreos: int, semilla: int) -> dict:
    uc = config.utils()
    pacientes = uc.cargar_patients().set_index("pid")
    # train + validacion del fold 0 = todo el train con su columna fold; test queda fuera
    entrenamiento = pd.concat(uc.particion(uc.cargar_samples(), fold_val=0))

    todos = pd.concat([por_paciente(n, pacientes, entrenamiento) for n in nombres])
    if todos.groupby("fold").ejecucion.nunique().gt(1).any():
        raise ValueError("dos ejecuciones con el mismo fold")

    modelo = lambda d: auc(d.percentil, d.pcr)                                          # noqa: E731
    modelo_crudo = lambda d: auc(d.prob, d.pcr)                                         # noqa: E731
    mismo_hosp = lambda d: auc_mismo_hospital(d.percentil, d.pcr, d.cohorte)           # noqa: E731
    cohorte = lambda d: auc(d.solo_cohorte_percentil, d.pcr)                            # noqa: E731
    cohorte_cruda = lambda d: auc(d.solo_cohorte, d.pcr)                                # noqa: E731
    ic = dict(remuestreos=remuestreos, semilla=semilla)

    folds = {}
    for fold, d in todos.groupby("fold"):
        folds[int(fold)] = {
            "ejecucion": d.ejecucion.iloc[0],
            "auc": con_ic(d, modelo_crudo, **ic),
            "mismo_hospital": con_ic(d, mismo_hosp, **ic),
            "solo_cohorte": cohorte_cruda(d),
            "por_epoca": auc_por_epoca(d.ejecucion.iloc[0]),
        }
    return {
        "ejecuciones": list(nombres),
        "pacientes": int(len(todos)),
        "global": {
            "auc": con_ic(todos, modelo, **ic),
            "mismo_hospital": con_ic(todos, mismo_hosp, **ic),
            "solo_cohorte": con_ic(todos, cohorte, **ic),
            "por_cohorte": {c: con_ic(todos[todos.cohorte == c], modelo, **ic) for c in COHORTES},
        },
        "folds": folds,
        "entre_folds": {
            "auc": media_desviacion([f["auc"]["auc"] for f in folds.values()]),
            "mismo_hospital": media_desviacion([f["mismo_hospital"]["auc"] for f in folds.values()]),
            "solo_cohorte": media_desviacion([f["solo_cohorte"] for f in folds.values()]),
            "auc_medio_epocas_1_15": media_desviacion(
                [f["por_epoca"]["ventana_1_15"] for f in folds.values()]),
            "auc_medio_todas_epocas": media_desviacion([f["por_epoca"]["todas"] for f in folds.values()]),
        },
    }


def imprimir(nombre: str, r: dict) -> None:
    def con(m):
        return f"{m['auc']:.3f} [{m['ic95'][0]:.3f}-{m['ic95'][1]:.3f}]"

    def md(m):
        return f"{m['media']:.3f} ± {m['desviacion']:.3f}"

    g, e = r["global"], r["entre_folds"]
    print(f"\n=== {nombre}  ({len(r['folds'])} folds, {r['pacientes']} pacientes de validacion)")
    print(f"{'fold':>6} {'AUC':>24} {'mismo hospital':>24} {'solo cohorte':>13} "
          f"{'medio 1-15':>11} {'medio todas':>12}")
    for fold, f in r["folds"].items():
        print(f"{fold:>6} {con(f['auc']):>24} {con(f['mismo_hospital']):>24} {f['solo_cohorte']:>13.3f} "
              f"{f['por_epoca']['ventana_1_15']:>11.3f} {f['por_epoca']['todas']:>12.3f}")
    print(f"{'global':>6} {con(g['auc']):>24} {con(g['mismo_hospital']):>24} {con(g['solo_cohorte']):>24}")
    print(f"{'media':>6} {md(e['auc']):>24} {md(e['mismo_hospital']):>24} {md(e['solo_cohorte']):>13}"
          f"  medio 1-15 {md(e['auc_medio_epocas_1_15'])}, todas {md(e['auc_medio_todas_epocas'])}")
    print("  por cohorte (global): " + ", ".join(
        f"{c} {con(m)} ({m['pacientes']} pac.)" for c, m in g["por_cohorte"].items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="nombre de la configuracion (results/analisis_<config>/)")
    p.add_argument("ejecuciones", nargs="+", help="carpetas de results/, una por fold (de 2 a 5)")
    p.add_argument("--remuestreos", type=int, default=2000)
    p.add_argument("--semilla", type=int, default=config.DEFECTO["semilla"])
    a = p.parse_args()
    if not 2 <= len(a.ejecuciones) <= 5:
        p.error("hacen falta de 2 a 5 ejecuciones")

    resumen = analizar(a.ejecuciones, a.remuestreos, a.semilla)
    carpeta = config.RESULTADOS / f"analisis_{a.config}"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    imprimir(a.config, resumen)
    print(f"Guardado en {(carpeta / 'resumen.json').relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
