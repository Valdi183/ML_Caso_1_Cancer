#!/usr/bin/env python3
"""Metricas descriptivas para el informe. No cambia nada del modelo ni del umbral.

    python -m src.utils.metricas_extra

El modelo esta cerrado y test ya se uso una vez (results/test/). Este script NO pasa
ninguna imagen por ninguna red: lo de test sale de las predicciones ya guardadas
(results/test/predicciones_test.csv, por corte y de cada uno de los 5 modelos), y lo de
validacion de las predicciones fuera de fold (results/<ejecucion>/predicciones_val.csv).

  a. Curva precision-recall y AP (validacion fuera de fold y test), con la linea de un
     modelo sin señal (AP = prevalencia) y el punto del umbral.
  b. Calibracion: Brier, Brier de la constante (tasa de pCR de train), Brier skill score,
     intercepto y pendiente (logistica de y sobre el logit de la probabilidad).
  c. p-valor del AUC de test frente a 0,5 por permutacion (por paciente y por corte;
     en el de corte se permutan las etiquetas ENTRE PACIENTES) y potencia del test.
  d. Matriz de confusion fuera de fold con el umbral, como la de test.
  e. Consistencia entre cortes de una paciente (fuera de fold): desviaciones, ICC y
     pacientes con cortes a los dos lados del umbral.
  f. Los 5 modelos en test: correlaciones, dispersion y cuantas pacientes supera cada uno.
  g. Modelo clinico (subtipo + edad + volumen), solo validacion, con los mismos folds.

Todo por paciente (probabilidad = media de sus cortes) salvo donde se dice; IC 95 % por
bootstrap de pacientes con semilla fija. Escribe results/metricas_extra/resumen.json y
docs/figuras/metricas_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from src import config
from src.utils import figuras
from src.utils.analisis_cv import COHORTES, auc
from src.utils.comparar import brier
from src.utils.evaluacion import auc_mismo_hospital
from src.utils.umbral import CONFIGURACIONES, metricas, umbral_especificidad
from src.utils.umbral import cargar as cargar_fuera_de_fold

PLAN = config.RESULTADOS / "test" / "plan.json"
TEST_CORTES = config.RESULTADOS / "test" / "predicciones_test.csv"
TEST_PACIENTES = config.RESULTADOS / "test" / "predicciones_test_pacientes.csv"
REMUESTREOS, PERMUTACIONES, SEMILLA = 2000, 10_000, config.DEFECTO["semilla"]
CLINICAS = {"subtipo + edad + volumen": (["age", "tum_vol"], ["HR_HER2_STATUS"]),
            "subtipo + edad (sin volumen)": (["age"], ["HR_HER2_STATUS"]),
            "solo volumen": (["tum_vol"], [])}
PRINCIPAL = "subtipo + edad + volumen"
ESPECIFICIDADES = (0.95, 0.90, 0.80, 0.70, 0.50)


# --------------------------------------------------------------------------- #
# Metricas (probadas en tests/test_metricas_extra.py)
# --------------------------------------------------------------------------- #

def brier_skill(prob, pcr, constante: float) -> float:
    """1 - Brier / Brier de predecir siempre `constante`: 0 = no mejora a la constante."""
    return 1 - brier(prob, pcr) / brier(np.full(len(pcr), constante), pcr)


def precision_media(prob, pcr) -> float:
    """AP (area bajo la curva precision-recall). Sin señal, su esperanza es la prevalencia."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(pcr), np.asarray(prob, dtype=float)))


def _logistica(y, x=None, desplazamiento=None, iteraciones: int = 100) -> np.ndarray:
    """Regresion logistica SIN penalizar por Newton-Raphson: [intercepto] o
    [intercepto, pendiente]; `desplazamiento` entra con coeficiente fijo 1."""
    y = np.asarray(y, dtype=float)
    X = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    fijo = np.zeros(len(y)) if desplazamiento is None else np.asarray(desplazamiento, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(iteraciones):
        p = 1 / (1 + np.exp(-(X @ beta + fijo)))
        paso = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        beta += paso
        if np.abs(paso).max() < 1e-10:
            break
    return beta


def calibracion(prob, pcr) -> dict:
    """Pendiente e intercepto de la logistica y ~ a + b * logit(p) (perfecta: a=0, b=1;
    b < 1 = probabilidades demasiado extremas, b > 1 = demasiado comprimidas). Ademas el
    intercepto con la pendiente fijada en 1 («calibracion global»: < 0 = sobreestima)."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    a, b = _logistica(pcr, z)
    return {"intercepto": float(a), "pendiente": float(b),
            "intercepto_pendiente_1": float(_logistica(pcr, desplazamiento=z)[0])}


def _auc_rangos(suma_rangos_pos, n_pos, n_total):
    """AUC de Mann-Whitney a partir de la suma de rangos de los positivos (vectorizado)."""
    n_neg = n_total - n_pos
    return (suma_rangos_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def permutacion_auc(prob, pcr, grupos=None, permutaciones: int = PERMUTACIONES,
                    semilla: int = SEMILLA) -> dict:
    """p-valor del AUC frente a 0,5 permutando las etiquetas ENTRE PACIENTES (`grupos`: la
    paciente de cada fila; None = una fila por paciente). Con cortes, todos los de una
    paciente se llevan su etiqueta juntos: permutar cortes sueltos rompe la correlacion
    entre cortes de la misma paciente y da p-valores demasiado pequeños."""
    prob, pcr = np.asarray(prob, dtype=float), np.asarray(pcr).astype(int)
    grupos = np.arange(len(prob)) if grupos is None else np.asarray(grupos)
    rangos = rankdata(prob)
    ids, posicion = np.unique(grupos, return_inverse=True)
    suma_rangos = np.bincount(posicion, weights=rangos)
    cortes = np.bincount(posicion)
    etiqueta = np.zeros(len(ids), dtype=int)
    etiqueta[posicion] = pcr
    observado = float(_auc_rangos(suma_rangos @ etiqueta, cortes @ etiqueta, len(prob)))

    rng = np.random.default_rng(semilla)
    nulos = []
    for inicio in range(0, permutaciones, 1000):
        bloque = rng.permuted(np.tile(etiqueta, (min(1000, permutaciones - inicio), 1)), axis=1)
        nulos.append(_auc_rangos(bloque @ suma_rangos, bloque @ cortes, len(prob)))
    nulos = np.concatenate(nulos)
    extremos = (np.abs(nulos - 0.5) >= abs(observado - 0.5)).sum()
    return {"auc": observado, "permutaciones": permutaciones,
            "p_una_cola": float((1 + (nulos >= observado).sum()) / (permutaciones + 1)),
            "p_dos_colas": float((1 + extremos) / (permutaciones + 1)),
            "nulo_percentiles_2.5_97.5": [float(np.percentile(nulos, q)) for q in (2.5, 97.5)]}


def potencia_formula(auc_real: float, n_pos: int, n_neg: int) -> float:
    """P(IC 95 % excluye 0,5) con el error tipico de Hanley y McNeil (1982)."""
    a = auc_real
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    se = np.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a ** 2) + (n_neg - 1) * (q2 - a ** 2)) / (n_pos * n_neg))
    return float(norm.cdf((a - 0.5) / se - 1.96) + norm.cdf((0.5 - a) / se - 1.96))


def potencia_simulacion(auc_real: float, n_pos: int, n_neg: int, simulaciones: int = 1000,
                        remuestreos: int = 1000, semilla: int = SEMILLA) -> float:
    """P(IC 95 % excluye 0,5) simulando: puntuaciones binormales con ese AUC (positivos
    N(d, 1), negativos N(0, 1), d = raiz(2) * Phi^-1(AUC)) y, en cada simulacion, el
    mismo IC que en test (bootstrap percentil de pacientes, sin estratificar)."""
    rng = np.random.default_rng(semilla)
    d = np.sqrt(2) * norm.ppf(auc_real)
    n = n_pos + n_neg
    excluye = 0
    for _ in range(simulaciones):
        pos, neg = rng.normal(d, 1, n_pos), rng.normal(0, 1, n_neg)
        gana = (pos[:, None] > neg[None, :]).astype(float)      # pares (positivo, negativo)
        # Bootstrap de las n pacientes: cuantas veces sale cada una
        veces = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(remuestreos)])
        vp, vn = veces[:, :n_pos].astype(float), veces[:, n_pos:].astype(float)
        aucs = ((vp @ gana) * vn).sum(axis=1) / (vp.sum(axis=1) * vn.sum(axis=1))
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        excluye += lo > 0.5 or hi < 0.5
    return excluye / simulaciones


def umbral_youden_tabla(prob, pcr) -> float:
    """Umbral que maximiza sensibilidad + especificidad - 1 con la regla pCR si prob > umbral,
    probando como umbral cada probabilidad observada (y uno por debajo de todas)."""
    prob, y = np.asarray(prob, dtype=float), np.asarray(pcr).astype(bool)
    candidatos = np.r_[prob.min() - 1e-9, np.unique(prob)]
    pred = prob[None, :] > candidatos[:, None]
    sens = (pred & y).sum(axis=1) / y.sum()
    esp = (~pred & ~y).sum(axis=1) / (~y).sum()
    return float(candidatos[np.argmax(sens + esp - 1)])


def referencia_azar(pcr, sensibilidad: float = 0.10, especificidad: float = 0.90) -> dict:
    """Recuentos ESPERADOS de un modelo sin señal que marca pCR al azar con esa tasa: su
    VPP es la prevalencia, sea cual sea la tasa."""
    y = np.asarray(pcr).astype(int)
    pos, neg = y.sum(), len(y) - y.sum()
    vp, fn = sensibilidad * pos, (1 - sensibilidad) * pos
    vn, fp = especificidad * neg, (1 - especificidad) * neg
    vpp, vpn = vp / (vp + fp), vn / (vn + fn)
    f1_1 = 2 * vpp * sensibilidad / (vpp + sensibilidad)
    f1_0 = 2 * vpn * especificidad / (vpn + especificidad)
    return {"VP": float(vp), "FP": float(fp), "FN": float(fn), "VN": float(vn), "sensibilidad": sensibilidad,
            "especificidad": especificidad, "VPP": float(vpp), "f1_macro": float((f1_1 + f1_0) / 2)}


def tabla_umbrales(prob, pcr, umbral_oficial: float) -> list[dict]:
    """Filas descriptivas: umbrales por especificidad objetivo (misma regla que el oficial),
    Youden, el oficial y las dos referencias sin señal."""
    prob, pcr = np.asarray(prob, dtype=float), np.asarray(pcr).astype(int)
    claves = ("VP", "FP", "FN", "VN", "sensibilidad", "especificidad", "VPP", "f1_macro")

    def fila(nombre, umbral):
        m = metricas(prob, pcr, umbral)
        return {"criterio": nombre, "umbral": umbral, **{k: m[k] for k in claves}}

    filas = [fila(f"especificidad >= {e:.2f}", umbral_especificidad(prob[pcr == 0], e))
             for e in ESPECIFICIDADES]
    filas.append(fila("Youden", umbral_youden_tabla(prob, pcr)))
    filas.append(fila("oficial (0,3635)", umbral_oficial))
    filas.append({"criterio": "siempre no pCR", "umbral": None,
                  **{k: metricas(prob, pcr, None)[k] for k in claves}})
    filas.append({"criterio": "azar con especificidad 0,90 (esperado)", "umbral": None,
                  **referencia_azar(pcr)})
    return filas


def icc(valores, grupos) -> float:
    """ICC(1) de un ANOVA de un factor con grupos de tamaño distinto: fraccion de la
    varianza que es entre pacientes (1 = todos los cortes de una paciente iguales)."""
    d = pd.DataFrame({"v": np.asarray(valores, dtype=float), "g": np.asarray(grupos)})
    g = d.groupby("g").v
    k, medias = g.size(), g.mean()
    n, a = len(d), len(k)
    msb = (k * (medias - d.v.mean()) ** 2).sum() / (a - 1)
    msw = ((d.v - d.g.map(medias)) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def bootstrap(funcion, n: int, remuestreos: int = REMUESTREOS, semilla: int = SEMILLA) -> dict:
    """{clave: [IC 2,5 %, IC 97,5 %]} de `funcion(indices) -> dict` remuestreando n filas."""
    rng = np.random.default_rng(semilla)
    muestras = pd.DataFrame([funcion(rng.integers(0, n, n)) for _ in range(remuestreos)])
    return {k: [float(np.nanpercentile(muestras[k], q)) for q in (2.5, 97.5)] for k in muestras}


# --------------------------------------------------------------------------- #
# Bloques del informe
# --------------------------------------------------------------------------- #

def bloque_pr_calibracion(prob, pcr, umbral: float, tasa_train: float) -> dict:
    """a y b para una tabla por paciente."""
    prob, pcr = np.asarray(prob, dtype=float), np.asarray(pcr).astype(int)

    def todo(i):
        return {"ap": precision_media(prob[i], pcr[i]), "brier": brier(prob[i], pcr[i]),
                "brier_skill": brier_skill(prob[i], pcr[i], tasa_train), **calibracion(prob[i], pcr[i])}

    valores = todo(np.arange(len(prob)))
    m = metricas(prob, pcr, umbral)
    return {**valores, "ic95": bootstrap(todo, len(prob)),
            "ap_sin_senal": float(pcr.mean()), "pacientes": int(len(prob)), "pcr": int(pcr.sum()),
            "brier_constante": brier(np.full(len(pcr), tasa_train), pcr), "tasa_pcr_train": tasa_train,
            "punto_umbral": {"umbral": umbral, "precision": m["VPP"], "recall": m["sensibilidad"]}}


def bloque_consistencia(cortes: pd.DataFrame, umbral: float) -> dict:
    """e: cuanto cambia la probabilidad entre los cortes de una misma paciente."""
    g = cortes.groupby("patient_id").prob
    dentro = g.std(ddof=1).dropna()
    medias = g.mean()
    lados = g.agg(lambda s: bool((s > umbral).any() and (s <= umbral).any()))
    por_fold = {int(f): icc(c.prob, c.patient_id) for f, c in cortes.groupby("fold")}
    return {"pacientes": int(len(medias)), "cortes": int(len(cortes)),
            "cortes_por_paciente_mediana": float(g.size().median()),
            "desviacion_dentro_media": float(dentro.mean()),
            "desviacion_dentro_mediana": float(dentro.median()),
            "desviacion_entre_pacientes": float(medias.std(ddof=1)),
            "icc": icc(cortes.prob, cortes.patient_id),
            "icc_por_fold": por_fold, "icc_media_folds": float(np.mean(list(por_fold.values()))),
            "fraccion_cortes_a_los_dos_lados_del_umbral": float(lados.mean()),
            "pacientes_cortes_a_los_dos_lados": int(lados.sum())}


def bloque_modelos_test(cortes: pd.DataFrame, umbral: float) -> dict:
    """f: los 5 modelos por separado frente al conjunto, por paciente (solo descriptivo)."""
    columnas = [c for c in cortes.columns if c.startswith("prob_")]
    tabla = cortes.groupby("patient_id")[columnas + ["prob"]].mean()
    nombres = {c: c.removeprefix("prob_") for c in columnas}
    modelos = tabla[columnas].rename(columns=nombres)
    return {"pearson": modelos.corr("pearson").round(4).to_dict(),
            "spearman": modelos.corr("spearman").round(4).to_dict(),
            "desviacion": {**{nombres[c]: float(tabla[c].std(ddof=1)) for c in columnas},
                           "conjunto": float(tabla.prob.std(ddof=1))},
            "media": {**{nombres[c]: float(tabla[c].mean()) for c in columnas},
                      "conjunto": float(tabla.prob.mean())},
            "pacientes_sobre_umbral": {**{nombres[c]: int((tabla[c] > umbral).sum()) for c in columnas},
                                       "conjunto": int((tabla.prob > umbral).sum())},
            "pacientes": int(len(tabla)), "umbral": umbral}


def oof_clinico(pac: pd.DataFrame, numericas: list[str], categoricas: list[str]) -> tuple[pd.Series, dict]:
    """Logistica con validacion cruzada por la columna fold. La imputacion, el escalado y
    el one-hot van DENTRO del pipeline, que se ajusta solo con los 4 folds de train."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    columnas = numericas + categoricas
    prob, por_fold = pd.Series(np.nan, index=pac.index), {}
    for k in range(5):
        entreno, valida = pac[pac.fold != k], pac[pac.fold == k]
        partes = []
        if numericas:
            partes.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                           numericas))
        if categoricas:
            partes.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                OneHotEncoder(handle_unknown="ignore")), categoricas))
        modelo = make_pipeline(ColumnTransformer(partes), LogisticRegression(max_iter=1000))
        modelo.fit(entreno[columnas], entreno.pCR)
        prob.loc[valida.index] = modelo.predict_proba(valida[columnas])[:, 1]
        por_fold[k] = auc(prob.loc[valida.index], valida.pCR)
    return prob, por_fold


def bloque_clinico(pac: pd.DataFrame, auc_medio_cnn: dict[int, float]) -> dict:
    """g: modelo clinico solo en validacion, con los mismos folds que la CNN."""
    resultado = {}
    for nombre, (numericas, categoricas) in CLINICAS.items():
        prob, por_fold = oof_clinico(pac, numericas, categoricas)
        d = pac.assign(prob=prob)
        d["percentil"] = d.groupby("fold").prob.rank(pct=True)
        aucs = np.array([por_fold[k] for k in range(5)])
        r = {"auc_por_fold": por_fold,
             "auc_folds_media": float(aucs.mean()), "auc_folds_desviacion_ddof0": float(aucs.std()),
             "auc_folds_desviacion_ddof1": float(aucs.std(ddof=1))}
        for clave, columna in (("auc_global", "prob"), ("auc_global_percentiles_por_fold", "percentil")):
            x, y = d[columna].to_numpy(), d.pCR.to_numpy()
            r[clave] = {"auc": auc(x, y), **bootstrap(lambda i, x=x, y=y: {"ic95": auc(x[i], y[i])}, len(d))}
        x, y, c = d.prob.to_numpy(), d.pCR.to_numpy(), d.dataset.to_numpy()
        r["mismo_hospital"] = {"auc": auc_mismo_hospital(x, y, c),
                               **bootstrap(lambda i: {"ic95": auc_mismo_hospital(x[i], y[i], c[i])}, len(d))}
        if nombre == PRINCIPAL:
            dif = np.array([por_fold[k] - auc_medio_cnn[k] for k in range(5)])
            r["frente_a_cnn"] = {"auc_medio_epocas_cnn_por_fold": auc_medio_cnn,
                                 "diferencia_por_fold": {k: float(v) for k, v in enumerate(dif)},
                                 "diferencia_media": float(dif.mean()),
                                 "diferencia_desviacion": float(dif.std(ddof=1))}
        resultado[nombre] = r
    return resultado


def bloque_volumen(pac: pd.DataFrame) -> dict:
    """tum_vol por cohorte, y cuanto delata la cohorte por si solo."""
    por_cohorte = {}
    for c in COHORTES:
        v = pac.loc[pac.dataset == c, "tum_vol"]
        descripcion = v.describe(percentiles=[0.25, 0.5, 0.75]).drop("count")
        por_cohorte[c] = {"pacientes": int(len(v)), "faltan": int(v.isna().sum()),
                          "valores_distintos": int(v.nunique()),
                          **{k: float(x) for k, x in descripcion.items()},
                          "auc_pcr_dentro_de_la_cohorte": auc(v.fillna(v.median()), pac.loc[v.index, "pCR"])}
    v = pac.tum_vol.fillna(pac.tum_vol.median())
    return {"por_cohorte": por_cohorte,
            "auc_volumen_para_reconocer_duke": 1 - auc(v, (pac.dataset == "duke").astype(int)),
            "nota": "auc_volumen_para_reconocer_duke: AUC de 'volumen pequeño' para separar Duke del resto"}


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def dibujar(val: pd.DataFrame, test: pd.DataFrame, r: dict, pac: pd.DataFrame, umbral: float) -> list:
    from sklearn.metrics import precision_recall_curve
    plt = figuras.plt          # figuras ya importa torch antes que matplotlib
    rutas = []

    figura, ejes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (titulo, tabla, b) in zip(ejes, (("Validacion fuera de fold", val, r["validacion"]),
                                             ("Test (conjunto de 5)", test, r["test"]))):
        precision, recall, _ = precision_recall_curve(tabla.pcr, tabla.prob)
        ic = b["ic95"]["ap"]
        ax.step(recall, precision, where="post", color=figuras.AZUL, lw=2,
                label=f"AP {b['ap']:.3f} [{ic[0]:.3f}-{ic[1]:.3f}]")
        ax.axhline(b["ap_sin_senal"], color=figuras.GRIS, ls=":", lw=1,
                   label=f"sin señal (prevalencia {b['ap_sin_senal']:.3f})")
        u = b["punto_umbral"]
        ax.plot(u["recall"], u["precision"], "o", color=figuras.ROJO, ms=9,
                label=f"umbral {umbral:.4f}: precision {u['precision']:.2f}, recall {u['recall']:.2f}")
        ax.set(title=f"{titulo}, por paciente ({b['pacientes']} pac., {b['pcr']} pCR)",
               xlabel="recall (sensibilidad)", ylabel="precision (VPP)", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="upper right")
    figura.tight_layout()
    rutas.append(figuras.guardar(figura, "metricas_precision_recall"))

    # Matriz fuera de fold con el mismo formato que docs/figuras/test_matriz_confusion.png
    m = metricas(val.prob, val.pcr, umbral)
    figura, ax = plt.subplots(figsize=(4.8, 4.2))
    matriz = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
    ax.imshow(matriz, cmap="Blues")
    for (i, j), v in np.ndenumerate(matriz):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > matriz.max() / 2 else "black")
    ax.set_xticks([0, 1], ["predice no pCR", "predice pCR"])
    ax.set_yticks([0, 1], ["no pCR (real)", "pCR (real)"])
    ax.set_title(f"Validacion fuera de fold por paciente, umbral {umbral:.4f}")
    rutas.append(figuras.guardar(figura, "metricas_val_matriz_confusion"))

    # Sensibilidad y especificidad frente al umbral (validacion fuera de fold)
    malla = np.linspace(val.prob.min(), val.prob.max(), 400)
    pred = val.prob.to_numpy()[None, :] > malla[:, None]
    y = val.pcr.to_numpy().astype(bool)
    figura, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(malla, (pred & y).sum(axis=1) / y.sum(), color=figuras.AZUL, lw=2, label="sensibilidad")
    ax.plot(malla, (~pred & ~y).sum(axis=1) / (~y).sum(), color=figuras.NARANJA, lw=2, label="especificidad")
    otros = [f for f in r["umbrales_validacion"]["filas"]
             if f["umbral"] is not None and not np.isclose(f["umbral"], umbral)]
    for i, f in enumerate(sorted(otros, key=lambda f: f["umbral"])):
        ax.axvline(f["umbral"], color=figuras.GRIS, ls=":", lw=0.8)
        ax.text(f["umbral"], 0.62 + 0.07 * i, " " + f["criterio"].replace("especificidad >= ", "esp "),
                fontsize=7, ha="left", va="center", color=figuras.GRIS,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.axvline(umbral, color=figuras.ROJO, lw=1.5, label=f"umbral oficial {umbral:.4f} (esp 0.90)")
    ax.set(title="Validacion fuera de fold, por paciente (punteadas: otros umbrales de la tabla)",
           xlabel="umbral (pCR si probabilidad > umbral)", ylabel="tasa", ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")
    rutas.append(figuras.guardar(figura, "metricas_val_sens_esp_umbral"))

    figura, ax = plt.subplots(figsize=(6.5, 4.2))
    colores = dict(zip(COHORTES, (figuras.AZUL, figuras.NARANJA, figuras.ROJO)))
    bordes = np.logspace(-2, 3, 41)
    for c in COHORTES:
        v = pac.loc[pac.dataset == c, "tum_vol"].dropna()
        # Fraccion de pacientes por intervalo (con density, un intervalo log estrecho con
        # una sola paciente aparece como un pico enorme)
        ax.hist(v.clip(lower=bordes[0]), bins=bordes, weights=np.full(len(v), 1 / len(v)), histtype="step",
                lw=2, color=colores[c], label=f"{c} ({len(v)} pac., mediana {v.median():.1f})")
    ax.set_xscale("log")
    ax.set(title="tum_vol por cohorte (pacientes de train)", xlabel="tum_vol (escala log)",
           ylabel="fraccion de pacientes de la cohorte")
    ax.legend(fontsize=8)
    rutas.append(figuras.guardar(figura, "metricas_tum_vol_cohorte"))
    plt.close("all")
    return rutas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulaciones", type=int, default=1000, help="simulaciones para la potencia")
    a = p.parse_args()

    faltan = [str(f.relative_to(config.REPO)) for f in (PLAN, TEST_CORTES, TEST_PACIENTES) if not f.exists()]
    faltan += [f"results/{n}/predicciones_val.csv" for n in CONFIGURACIONES["normal"]
               if not (config.RESULTADOS / n / "predicciones_val.csv").exists()]
    if faltan:
        sys.exit("Faltan ficheros:\n  " + "\n  ".join(faltan))

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    umbral = float(plan["umbral"])
    uc = config.utils()
    samples = uc.cargar_samples()
    pacientes = uc.cargar_patients().set_index("pid")
    entrenamiento = samples[samples.split == "train"]                    # sin test
    tasa_train = float(entrenamiento.groupby("patient_id").pCR.first().mean())

    # Validacion fuera de fold (cada paciente, con el modelo que no la vio)
    val, val_cortes = cargar_fuera_de_fold(CONFIGURACIONES["normal"], uc, pacientes, entrenamiento)
    # Test: SOLO las predicciones ya guardadas por evaluar_test (no se carga ninguna imagen)
    test = pd.read_csv(TEST_PACIENTES).set_index("patient_id")
    test_cortes = pd.read_csv(TEST_CORTES)
    if not ((test.prob > umbral).astype(int) == test.prediccion).all():
        sys.exit("Las predicciones guardadas de test no cuadran con el umbral de plan.json")

    pac = pacientes[pacientes.split == "train"].copy()
    pac["fold"] = entrenamiento.groupby("patient_id").fold.first()
    auc_medio_cnn = {}
    for n in CONFIGURACIONES["normal"]:
        fold = json.loads((config.RESULTADOS / n / "config.json").read_text(encoding="utf-8"))["fold"]
        auc_medio_cnn[int(fold)] = float(pd.read_csv(config.RESULTADOS / n / "historial.csv").auc_val.mean())

    n_pos, n_neg = int(test.pcr.sum()), int((test.pcr == 0).sum())
    m_val = metricas(val.prob, val.pcr, umbral)
    resumen = {
        "aviso": "Solo descriptivo: no cambia ni el modelo, ni el umbral, ni la agregacion. Test sale de las "
                 "predicciones guardadas en results/test/ (no se paso ninguna imagen por ninguna red).",
        "umbral": umbral, "remuestreos": REMUESTREOS, "semilla": SEMILLA,
        "validacion": bloque_pr_calibracion(val.prob, val.pcr, umbral, tasa_train),
        "test": bloque_pr_calibracion(test.prob, test.pcr, umbral, tasa_train),
        "permutacion_test": {
            "por_paciente": permutacion_auc(test.prob, test.pcr),
            "por_corte": {**permutacion_auc(test_cortes.prob, test_cortes.pCR, test_cortes.patient_id),
                          "nota": "etiquetas permutadas entre pacientes (cada paciente arrastra todos "
                                  "sus cortes)"}},
        "potencia": {"auc_real": 0.55, "pcr": n_pos, "no_pcr": n_neg,
                     "formula_hanley_mcneil": potencia_formula(0.55, n_pos, n_neg),
                     "simulacion": potencia_simulacion(0.55, n_pos, n_neg, a.simulaciones),
                     "simulaciones": a.simulaciones,
                     "metodo_simulacion": "puntuaciones binormales con AUC 0,55; IC por bootstrap "
                                          "percentil de pacientes (1000 remuestreos), como en test"},
        "matriz_validacion": {k: m_val[k] for k in ("VP", "FP", "FN", "VN", "sensibilidad", "especificidad",
                                                    "VPP", "VPN")},
        "umbrales_validacion": {
            "nota": "descriptivo, validacion fuera de fold por paciente; el umbral oficial no cambia",
            "filas": tabla_umbrales(val.prob, val.pcr, umbral)},
        "consistencia_cortes_validacion": bloque_consistencia(val_cortes, umbral),
        "modelos_en_test": bloque_modelos_test(test_cortes, umbral),
        "clinico_validacion": bloque_clinico(pac, auc_medio_cnn),
        "tum_vol": bloque_volumen(pac),
        "cnn_validacion_referencia": {"auc_medio_epocas_por_fold": auc_medio_cnn,
                                      "auc_global_fuera_de_fold": json.loads(
                                          (config.RESULTADOS / "umbral" / "resumen.json").read_text(
                                              encoding="utf-8"))["normal"]["auc_global"]},
    }
    rutas = dibujar(val, test, resumen, pac, umbral)
    resumen["figuras"] = [str(r.relative_to(config.REPO)) for r in rutas]

    destino = config.RESULTADOS / "metricas_extra"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float, ensure_ascii=False),
                                          encoding="utf-8")
    imprimir(resumen)
    print(f"\nGuardado en {(destino / 'resumen.json').relative_to(config.REPO)}; figuras: "
          + ", ".join(resumen["figuras"]))


def imprimir(r: dict) -> None:
    def ic(b, k):
        return f"{b[k]:.3f} [{b['ic95'][k][0]:.3f}-{b['ic95'][k][1]:.3f}]"

    for nombre in ("validacion", "test"):
        b = r[nombre]
        print(f"\n=== {nombre} ({b['pacientes']} pac., {b['pcr']} pCR)")
        print(f"AP {ic(b, 'ap')} (sin señal {b['ap_sin_senal']:.3f});  Brier {ic(b, 'brier')} "
              f"(constante {b['brier_constante']:.4f});  skill {ic(b, 'brier_skill')}")
        print(f"calibracion: intercepto {ic(b, 'intercepto')}, pendiente {ic(b, 'pendiente')}, "
              f"intercepto con pendiente 1 {ic(b, 'intercepto_pendiente_1')}")
    pp, pc = r["permutacion_test"]["por_paciente"], r["permutacion_test"]["por_corte"]
    print(f"\nPermutacion test: paciente AUC {pp['auc']:.3f} p={pp['p_dos_colas']:.4f};  "
          f"corte AUC {pc['auc']:.3f} p={pc['p_dos_colas']:.4f} (dos colas)")
    po = r["potencia"]
    print(f"Potencia (AUC real 0,55, {po['pcr']}/{po['no_pcr']}): formula {po['formula_hanley_mcneil']:.3f}, "
          f"simulacion {po['simulacion']:.3f}")
    c = r["consistencia_cortes_validacion"]
    print(f"Consistencia: desv. dentro {c['desviacion_dentro_media']:.4f}, "
          f"entre {c['desviacion_entre_pacientes']:.4f}, ICC {c['icc']:.3f} "
          f"(media folds {c['icc_media_folds']:.3f}); a los dos lados del umbral "
          f"{c['fraccion_cortes_a_los_dos_lados_del_umbral']:.3f}")
    print("Umbrales (validacion fuera de fold, descriptivo):")
    for f in r["umbrales_validacion"]["filas"]:
        u = "—" if f["umbral"] is None else f"{f['umbral']:.4f}"
        print(f"  {f['criterio']:<40} {u:>7}  VP {f['VP']:>6.1f} FP {f['FP']:>6.1f} FN {f['FN']:>6.1f} "
              f"VN {f['VN']:>6.1f}  sens {f['sensibilidad']:.3f} esp {f['especificidad']:.3f} "
              f"VPP {f['VPP']:.3f} F1m {f['f1_macro']:.3f}")
    m = r["modelos_en_test"]
    print("Modelos en test, sobre el umbral: "
          + ", ".join(f"{k} {v}" for k, v in m["pacientes_sobre_umbral"].items()))
    for nombre, x in r["clinico_validacion"].items():
        g, mh = x["auc_global"], x["mismo_hospital"]
        print(f"Clinico {nombre}: folds {x['auc_folds_media']:.3f} ± {x['auc_folds_desviacion_ddof0']:.3f}; "
              f"global {g['auc']:.3f} [{g['ic95'][0]:.3f}-{g['ic95'][1]:.3f}]; "
              f"mismo hospital {mh['auc']:.3f} [{mh['ic95'][0]:.3f}-{mh['ic95'][1]:.3f}]")


if __name__ == "__main__":
    main()
