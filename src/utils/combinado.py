#!/usr/bin/env python3
"""¿Añadir la imagen a las variables clinicas aporta algo? Solo validacion, nada de test.

    python -m src.utils.combinado

Ni el clinico ni el combinado se evaluan en test. Este script NO pasa ninguna imagen por
ninguna red: la CNN entra por sus predicciones fuera de fold ya guardadas
(results/<ejecucion>/predicciones_val.csv del congelado), por paciente (media de cortes).

  1. Clinicas: las del punto g de metricas_extra (subtipo + edad + volumen), con tum_vol
     tal cual (imputado con la mediana y escalado), como alli.
  2. Combinacion anidada por fold: para cada fold k, la logistica con [clinicas + logit de
     la probabilidad fuera de fold de la CNN] se ajusta en los otros 4 folds y predice k.
     Imputacion, escalado y one-hot van dentro del pipeline (oof_clinico de metricas_extra).
  3. Frente al clinico solo y la CNN sola: AUC por fold y diferencia pareada, AUC global
     fuera de fold y entre pares del mismo hospital con IC (bootstrap de pacientes), y una
     prueba de razon de verosimilitudes de añadir el logit de la CNN al clinico.

Prediccion fijada antes: aunque fueran independientes, la combinacion optima daria ~0,70
(+0,005 a +0,015 sobre 0,692), indistinguible del clinico solo.
Escribe results/combinado/resumen.json y docs/figuras/combinado_validacion.png.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src import config
from src.utils import figuras
from src.utils.analisis_cv import auc
from src.utils.evaluacion import auc_mismo_hospital
from src.utils.metricas_extra import CLINICAS, PRINCIPAL, REMUESTREOS, SEMILLA, bootstrap, oof_clinico
from src.utils.umbral import CONFIGURACIONES
from src.utils.umbral import cargar as cargar_fuera_de_fold

NUMERICAS, CATEGORICAS = CLINICAS[PRINCIPAL]
CLINICO, CNN, COMBINADO = "clinico", "cnn", "clinico + cnn"
COMBINADO_FOLD = "clinico + cnn (logit estandarizado por fold)"
AVISO = ("Solo validacion: ni el clinico ni el combinado se evaluan en test. La epoca de cada CNN se "
         "eligio con estos mismos folds de validacion, asi que su probabilidad fuera de fold (y con ella "
         "la aportacion de la CNN al combinado) sale algo inflada: la CNN sola da 0,589 de AUC por fold en "
         "la epoca elegida frente a 0,551 de media de todas las epocas. Ademas, en el combinado el "
         "coeficiente del fold k se ajusta con predicciones de los modelos de los otros folds, que se "
         "entrenaron con las pacientes del fold k (fuga pequeña, en el mismo sentido).")


# --------------------------------------------------------------------------- #
# Metricas (probadas en tests/test_combinado.py)
# --------------------------------------------------------------------------- #

def logit(prob) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def estandarizar_por_fold(x, fold) -> np.ndarray:
    """(x - media) / desviacion dentro de cada fold. Sin etiquetas: quita la diferencia de
    escala entre los 5 modelos (cada fold lo predice un modelo distinto)."""
    s = pd.Series(np.asarray(x, dtype=float))
    g = s.groupby(np.asarray(fold))
    return ((s - g.transform("mean")) / g.transform("std")).to_numpy()


def logistica(y, X, iteraciones: int = 100) -> dict:
    """Logistica SIN penalizar por Newton-Raphson. X sin la columna de unos (se añade).
    Devuelve coeficientes [intercepto, ...], errores tipicos (inversa de la informacion de
    Fisher) y log-verosimilitud."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    beta = np.zeros(X.shape[1])
    for _ in range(iteraciones):
        p = 1 / (1 + np.exp(-(X @ beta)))
        informacion = X.T @ (X * (p * (1 - p))[:, None])
        paso = np.linalg.solve(informacion, X.T @ (y - p))
        beta += paso
        if np.abs(paso).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    informacion = X.T @ (X * (p * (1 - p))[:, None])
    return {"beta": beta, "ee": np.sqrt(np.diag(np.linalg.inv(informacion))),
            "logverosimilitud": float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())}


def matriz_clinica(pac: pd.DataFrame, cohorte: bool = False) -> pd.DataFrame:
    """Diseño de la logistica de la prueba de razon de verosimilitudes, con TODAS las pacientes
    (es un ajuste en la muestra, no una prediccion): numericas imputadas con la mediana y
    estandarizadas; categoricas con la moda y indicadoras sin la categoria mas frecuente."""
    partes = []
    for c in NUMERICAS:
        v = pac[c].fillna(pac[c].median())
        partes.append(((v - v.mean()) / v.std()).rename(c))
    for c in CATEGORICAS + (["dataset"] if cohorte else []):
        v = pac[c].fillna(pac[c].mode()[0])
        partes.append(pd.get_dummies(v, prefix=c, dtype=float).drop(columns=f"{c}_{v.mode()[0]}"))
    return pd.concat(partes, axis=1)


def razon_verosimilitudes(y, X_base: pd.DataFrame, extra, nombre: str = "logit_cnn") -> dict:
    """Prueba de razon de verosimilitudes (1 grado de libertad) de añadir `extra` a X_base,
    con el coeficiente de `extra`, su IC de Wald y la razon de odds por unidad y por desviacion."""
    extra = np.asarray(extra, dtype=float)
    base = logistica(y, X_base)
    completo = logistica(y, np.column_stack([X_base.to_numpy(dtype=float), extra]))
    estadistico = 2 * (completo["logverosimilitud"] - base["logverosimilitud"])
    b, ee = float(completo["beta"][-1]), float(completo["ee"][-1])
    ic = [b - norm.ppf(0.975) * ee, b + norm.ppf(0.975) * ee]
    sd = float(extra.std(ddof=1))
    return {"variable": nombre, "covariables_base": list(X_base.columns), "pacientes": int(len(extra)),
            "logverosimilitud_base": base["logverosimilitud"],
            "logverosimilitud_con_cnn": completo["logverosimilitud"],
            "estadistico_lr": float(estadistico), "grados_libertad": 1,
            "p_valor": float(chi2.sf(estadistico, 1)),
            "coeficiente": b, "error_tipico": ee, "ic95_wald": ic,
            "razon_odds_por_unidad_de_logit": float(np.exp(b)),
            "razon_odds_por_unidad_ic95": [float(np.exp(x)) for x in ic],
            "desviacion_del_logit": sd,
            "razon_odds_por_desviacion": float(np.exp(b * sd)),
            "razon_odds_por_desviacion_ic95": [float(np.exp(x * sd)) for x in ic]}


def diferencia_pareada(a: dict[int, float], b: dict[int, float]) -> dict:
    """a - b fold a fold: media, desviacion (ddof=1, como en metricas_extra) y folds en que gana a."""
    d = np.array([a[k] - b[k] for k in sorted(a)])
    return {"por_fold": {k: float(v) for k, v in zip(sorted(a), d)}, "media": float(d.mean()),
            "desviacion_ddof1": float(d.std(ddof=1)), "folds_a_favor": int((d > 0).sum())}


def bloque_global(probs: dict[str, np.ndarray], pcr, cohorte, referencia: str = CLINICO,
                  remuestreos: int = REMUESTREOS) -> dict:
    """AUC global fuera de fold y entre pares del mismo hospital de cada modelo, y su
    diferencia con `referencia`, con IC por bootstrap de pacientes (los mismos remuestreos
    para todos: las diferencias son pareadas)."""
    y, c = np.asarray(pcr).astype(int), np.asarray(cohorte)
    probs = {k: np.asarray(v, dtype=float) for k, v in probs.items()}

    def todo(i):
        r = {}
        for k, p in probs.items():
            r[f"{k}|global"] = auc(p[i], y[i])
            r[f"{k}|mismo_hospital"] = auc_mismo_hospital(p[i], y[i], c[i])
        for k in probs:
            for m in ("global", "mismo_hospital"):
                r[f"{k}|{m}_menos_{referencia}"] = r[f"{k}|{m}"] - r[f"{referencia}|{m}"]
        return r

    valores, ic = todo(np.arange(len(y))), bootstrap(todo, len(y), remuestreos)
    resultado = {}
    for k in probs:
        resultado[k] = {}
        for m in ("global", "mismo_hospital"):
            resultado[k][m] = {"auc": valores[f"{k}|{m}"], "ic95": ic[f"{k}|{m}"]}
            if k != referencia:
                clave = f"{k}|{m}_menos_{referencia}"
                resultado[k][m][f"diferencia_con_{referencia}"] = {"valor": valores[clave], "ic95": ic[clave]}
    return resultado


# --------------------------------------------------------------------------- #
# Figura
# --------------------------------------------------------------------------- #

def dibujar(pac: pd.DataFrame, probs: dict, por_fold: dict, r: dict):
    from sklearn.metrics import roc_curve
    plt = figuras.plt          # figuras ya importa torch antes que matplotlib
    colores = {CLINICO: figuras.AZUL, CNN: figuras.GRIS, COMBINADO: figuras.ROJO}
    figura, (izq, der) = plt.subplots(1, 2, figsize=(12, 5))

    folds = sorted(por_fold[CLINICO])
    for k, color in colores.items():
        aucs = [por_fold[k][f] for f in folds]
        izq.plot(folds, aucs, "o-", color=color, lw=1.5,
                 label=f"{k}: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    izq.plot(folds, [r["cnn_auc_medio_epocas_por_fold"][f] for f in folds], "x:", color=figuras.GRIS,
             lw=1, label="cnn, AUC medio de todas las epocas (referencia)")
    izq.axhline(0.5, color="black", lw=0.6, ls=":")
    izq.set(title="AUC por fold (cada fold, con un modelo que no lo vio)", xlabel="fold",
            ylabel="AUC por paciente", xticks=folds, ylim=(0.45, 0.8))
    izq.legend(fontsize=8, loc="lower left")

    y = pac.pCR.to_numpy()
    for k, color in colores.items():
        fpr, tpr, _ = roc_curve(y, probs[k])
        g = r["global"][k]["global"]
        der.plot(fpr, tpr, color=color, lw=2,
                 label=f"{k}: {g['auc']:.3f} [{g['ic95'][0]:.3f}-{g['ic95'][1]:.3f}]")
    der.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":")
    d = r["global"][COMBINADO]["global"][f"diferencia_con_{CLINICO}"]
    der.set(title=f"ROC global fuera de fold ({len(y)} pac.); combinado - clinico "
                  f"{d['valor']:+.3f} [{d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f}]",
            xlabel="1 - especificidad", ylabel="sensibilidad")
    der.title.set_fontsize(9)
    der.legend(fontsize=8, loc="lower right")
    figura.suptitle("Validacion fuera de fold: clinico (subtipo + edad + volumen) frente a clinico + CNN "
                    "(sin test)", fontsize=11)
    figura.tight_layout()
    ruta = figuras.guardar(figura, "combinado_validacion")
    plt.close("all")
    return ruta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    faltan = [f"results/{n}/predicciones_val.csv" for n in CONFIGURACIONES["normal"]
              if not (config.RESULTADOS / n / "predicciones_val.csv").exists()]
    if faltan:
        sys.exit("Faltan ficheros:\n  " + "\n  ".join(faltan))

    uc = config.utils()
    samples = uc.cargar_samples()
    pacientes = uc.cargar_patients().set_index("pid")
    entrenamiento = samples[samples.split == "train"]                    # sin test
    val, _ = cargar_fuera_de_fold(CONFIGURACIONES["normal"], uc, pacientes, entrenamiento)

    pac = pacientes[pacientes.split == "train"].copy()
    pac["fold"] = entrenamiento.groupby("patient_id").fold.first()
    pac["prob_cnn"] = val.prob.reindex(pac.index)
    if pac.prob_cnn.isna().any() or len(val) != len(pac):
        sys.exit("No todas las pacientes de train tienen probabilidad fuera de fold de la CNN")
    pac["logit_cnn"] = logit(pac.prob_cnn)
    pac["logit_cnn_fold"] = estandarizar_por_fold(pac.logit_cnn, pac.fold)

    probs, por_fold = {}, {}
    probs[CLINICO], por_fold[CLINICO] = oof_clinico(pac, NUMERICAS, CATEGORICAS)
    probs[COMBINADO], por_fold[COMBINADO] = oof_clinico(pac, NUMERICAS + ["logit_cnn"], CATEGORICAS)
    probs[COMBINADO_FOLD], por_fold[COMBINADO_FOLD] = oof_clinico(pac, NUMERICAS + ["logit_cnn_fold"],
                                                                  CATEGORICAS)
    probs[CNN] = pac.prob_cnn
    por_fold[CNN] = {k: auc(d.prob_cnn, d.pCR) for k, d in pac.groupby("fold")}
    medio_epocas = {}
    for n in CONFIGURACIONES["normal"]:
        fold = json.loads((config.RESULTADOS / n / "config.json").read_text(encoding="utf-8"))["fold"]
        medio_epocas[int(fold)] = float(pd.read_csv(config.RESULTADOS / n / "historial.csv").auc_val.mean())

    def resumen_folds(d):
        a = np.array([d[k] for k in sorted(d)])
        return {"auc_por_fold": {int(k): float(v) for k, v in d.items()}, "media": float(a.mean()),
                "desviacion_ddof0": float(a.std()), "desviacion_ddof1": float(a.std(ddof=1))}

    y = pac.pCR.astype(int).to_numpy()
    base, base_cohorte = matriz_clinica(pac), matriz_clinica(pac, cohorte=True)
    resumen = {
        "aviso": AVISO,
        "prediccion_previa": "Aunque fueran independientes, la combinacion optima daria ~0,70 (+0,005 a "
                             "+0,015 sobre 0,692 por fold), indistinguible del clinico solo.",
        "clinicas": {"numericas": NUMERICAS, "categoricas": CATEGORICAS,
                     "tum_vol": "tal cual, como en el punto g de metricas_extra: imputado con la mediana y "
                                "estandarizado, ajustados en los 4 folds de entrenamiento de cada fold"},
        "cnn": {"ejecuciones": CONFIGURACIONES["normal"],
                "entrada": "logit de la probabilidad fuera de fold por paciente (media de cortes), del "
                           "modelo del congelado que no vio ese fold, en su epoca elegida"},
        "pacientes": int(len(pac)), "pcr": int(y.sum()), "remuestreos": REMUESTREOS, "semilla": SEMILLA,
        "por_fold": {k: resumen_folds(v) for k, v in por_fold.items()},
        "cnn_auc_medio_epocas_por_fold": medio_epocas,
        "diferencias_por_fold": {
            f"{COMBINADO} - {CLINICO}": diferencia_pareada(por_fold[COMBINADO], por_fold[CLINICO]),
            f"{COMBINADO} - {CNN}": diferencia_pareada(por_fold[COMBINADO], por_fold[CNN]),
            f"{COMBINADO_FOLD} - {CLINICO}": diferencia_pareada(por_fold[COMBINADO_FOLD], por_fold[CLINICO])},
        "global": bloque_global(probs, y, pac.dataset),
        "razon_verosimilitudes": {
            "nota": "logisticas sin penalizar ajustadas con las 1.097 pacientes (en la muestra, no fuera de "
                    "fold); IC de Wald",
            "clinico": razon_verosimilitudes(y, base, pac.logit_cnn),
            "clinico_y_cohorte": razon_verosimilitudes(y, base_cohorte, pac.logit_cnn),
            "clinico_logit_estandarizado_por_fold": razon_verosimilitudes(y, base, pac.logit_cnn_fold,
                                                                          "logit_cnn_fold")},
        "correlacion_logit_cnn_con_clinico": {
            "spearman_con_prob_clinica_fuera_de_fold": float(pd.Series(probs[CLINICO]).corr(
                pac.prob_cnn, method="spearman"))},
        "nota_sensibilidad": f"'{COMBINADO_FOLD}': el logit de la CNN centrado y escalado dentro de cada "
                             "fold (sin etiquetas), porque cada fold lo predice un modelo con otra escala; "
                             "no sustituye al analisis principal",
    }
    resumen["figura"] = str(dibujar(pac, probs, por_fold, resumen).relative_to(config.REPO))

    destino = config.RESULTADOS / "combinado"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float, ensure_ascii=False),
                                          encoding="utf-8")
    imprimir(resumen)
    print(f"\nGuardado en {(destino / 'resumen.json').relative_to(config.REPO)}; figura: {resumen['figura']}")


def imprimir(r: dict) -> None:
    print(f"=== Validacion fuera de fold ({r['pacientes']} pac., {r['pcr']} pCR); sin test")
    for k, v in r["por_fold"].items():
        g, h = r["global"][k]["global"], r["global"][k]["mismo_hospital"]
        print(f"{k:<46} folds {v['media']:.3f} ± {v['desviacion_ddof0']:.3f}   "
              f"global {g['auc']:.3f} [{g['ic95'][0]:.3f}-{g['ic95'][1]:.3f}]   "
              f"mismo hospital {h['auc']:.3f} [{h['ic95'][0]:.3f}-{h['ic95'][1]:.3f}]")
    for k, d in r["diferencias_por_fold"].items():
        print(f"{k:<58} {d['media']:+.4f} ± {d['desviacion_ddof1']:.4f} ({d['folds_a_favor']}/5 a favor)")
    for k, t in r["razon_verosimilitudes"].items():
        if k != "nota":
            print(f"LR {k:<40} chi2 {t['estadistico_lr']:.2f}, p {t['p_valor']:.4f}; "
                  f"coef {t['coeficiente']:.3f} [{t['ic95_wald'][0]:.3f}-{t['ic95_wald'][1]:.3f}]; "
                  f"OR por desviacion {t['razon_odds_por_desviacion']:.3f}")


if __name__ == "__main__":
    main()
