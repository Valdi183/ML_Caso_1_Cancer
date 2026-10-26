#!/usr/bin/env python3
"""Umbral del modelo final con un criterio clinico fijado antes de ver los numeros.

    python -m src.utils.umbral

Criterio: un «pCR = 1» con la RM previa podria llevar a desescalar el tratamiento, asi
que el error grave es el falso positivo: **especificidad >= 0,90 por paciente**,
agregando por la media de sus cortes. Umbral oficial (`umbral_final`): el percentil 90
de la probabilidad por paciente de TODAS las pacientes pCR=0 fuera de fold juntas (se
predice pCR si prob > umbral); es el que cumple el criterio sobre el conjunto y no se
elige mirando la sensibilidad. Como informacion de la incertidumbre del umbral se
guardan los t_k de cada fold (percentil 90 de sus pCR=0), su dispersion y la
especificidad de cada fold con el umbral oficial.

La regla inicial (umbral = mediana de los t_k, 0,3393) daba especificidad 0,839 en el
conjunto porque la escala de probabilidad cambia entre folds: se corrigio el
procedimiento para cumplir el criterio fijado, sin cambiar el criterio.

Usa solo las predicciones fuera de fold de validacion (results/<ejecucion>/
predicciones_val.csv) del modelo congelado con perdida normal y, como comparacion,
con perdida ponderada. Test NO se usa. Escribe results/umbral/resumen.json, la
figura de calibracion docs/figuras/umbral_calibracion.png y una tabla en consola.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config
from src.utils import figuras
from src.utils.analisis_cv import COHORTES, analizar, auc, por_paciente
from src.utils.comparar import brier

CONFIGURACIONES = {"normal": figuras.CINCO_FOLDS["congelado"],
                   "ponderada": figuras.CINCO_FOLDS["congelado_ponderada"]}
CLAVES_COMPARACION = {"normal": "congelado", "ponderada": "congelado_ponderada"}
ESPECIFICIDAD = 0.90
METODOS = ("mean", "median", "max", "voto")


# --------------------------------------------------------------------------- #
# Regla del umbral y metricas (probadas en tests/test_umbral.py)
# --------------------------------------------------------------------------- #

def umbral_especificidad(prob_negativos, especificidad: float = ESPECIFICIDAD) -> float:
    """Percentil `especificidad` de las probabilidades de los negativos: prediciendo
    positivo solo si prob > umbral, al menos esa fraccion de negativos queda por debajo.
    Sin interpolar ("higher": un valor observado): interpolando, con n negativos podria
    quedar por debajo algo menos del 90 % (p. ej. 137 de 153)."""
    return float(np.quantile(np.asarray(prob_negativos, dtype=float), especificidad, method="higher"))


def umbrales_por_fold(tabla: pd.DataFrame, especificidad: float = ESPECIFICIDAD) -> dict[int, float]:
    """t_k de cada fold, con la probabilidad por paciente (columnas fold, prob, pcr)."""
    return {int(f): umbral_especificidad(d.prob[d.pcr == 0], especificidad) for f, d in tabla.groupby("fold")}


def umbral_conjunto(tabla: pd.DataFrame, especificidad: float = ESPECIFICIDAD) -> float:
    """Umbral oficial: el de `umbral_especificidad` sobre las pCR=0 de todos los folds juntos."""
    return umbral_especificidad(tabla.prob[tabla.pcr == 0], especificidad)


def metricas(prob, pcr, umbral: float | None) -> dict:
    """Matriz y metricas prediciendo pCR si prob > umbral (umbral None = siempre no pCR)."""
    y = np.asarray(pcr).astype(bool)
    pred = np.zeros_like(y) if umbral is None else np.asarray(prob, dtype=float) > umbral
    vp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, vn = int((~pred & y).sum()), int((~pred & ~y).sum())

    def div(a, b):
        return a / b if b else float("nan")

    def f1(p, r):
        return 0.0 if (np.isnan(p) or p + r == 0) else 2 * p * r / (p + r)

    sens, esp, vpp, vpn = div(vp, vp + fn), div(vn, vn + fp), div(vp, vp + fp), div(vn, vn + fn)
    f1_1, f1_0 = f1(vpp, sens), f1(vpn, esp)
    return {"VP": vp, "FP": fp, "FN": fn, "VN": vn, "sensibilidad": sens, "especificidad": esp,
            "VPP": vpp, "VPN": vpn,
            "pcr1": {"precision": vpp, "recall": sens, "f1": f1_1},
            "pcr0": {"precision": vpn, "recall": esp, "f1": f1_0},
            "f1_macro": (f1_1 + f1_0) / 2, "exactitud": div(vp + vn, len(y))}


ESCALARES = ("sensibilidad", "especificidad", "VPP", "VPN", "f1_macro", "exactitud")


def plano(m: dict) -> dict:
    """Metricas escalares, incluidas las de cada clase (para el bootstrap)."""
    r = {k: m[k] for k in ESCALARES}
    for clase in ("pcr1", "pcr0"):
        r.update({f"{clase}_{k}": v for k, v in m[clase].items()})
    return r


def con_ic(prob, pcr, grupos, umbral, remuestreos: int, semilla: int) -> dict:
    """metricas() y su IC 95 % remuestreando PACIENTES (`grupos`: la paciente de cada
    fila; con una fila por paciente es un bootstrap normal, con cortes es por paciente)."""
    prob, pcr, grupos = np.asarray(prob, dtype=float), np.asarray(pcr), np.asarray(grupos)
    resultado = metricas(prob, pcr, umbral)
    ids, posicion = np.unique(grupos, return_inverse=True)
    filas_de = [np.flatnonzero(posicion == i) for i in range(len(ids))]
    rng = np.random.default_rng(semilla)
    muestras = []
    for _ in range(remuestreos):
        i = np.concatenate([filas_de[j] for j in rng.integers(0, len(ids), len(ids))])
        muestras.append(plano(metricas(prob[i], pcr[i], umbral)))
    muestras = pd.DataFrame(muestras)
    resultado["ic95"] = {
        k: [float(np.nanpercentile(muestras[k], q)) for q in (2.5, 97.5)] if muestras[k].notna().any()
        else [float("nan")] * 2
        for k in muestras}
    return resultado


def fiabilidad(prob, pcr, grupos: int = 10) -> pd.DataFrame:
    """Diagrama de fiabilidad: grupos por cuantiles de la probabilidad, con su
    probabilidad media y la fraccion observada de pCR."""
    d = pd.DataFrame({"prob": np.asarray(prob, dtype=float), "pcr": np.asarray(pcr, dtype=float)})
    d["grupo"] = pd.qcut(d.prob.rank(method="first"), grupos, labels=False)
    return d.groupby("grupo").agg(prob_media=("prob", "mean"), observada=("pcr", "mean"),
                                  pacientes=("pcr", "size")).reset_index()


# --------------------------------------------------------------------------- #
# Datos fuera de fold
# --------------------------------------------------------------------------- #

def cargar(nombres: list[str], uc, pacientes: pd.DataFrame, entrenamiento: pd.DataFrame):
    """(por paciente, por corte) fuera de fold, con fold y cohorte."""
    tabla = pd.concat([por_paciente(n, pacientes, entrenamiento) for n in nombres])
    cortes = []
    for n in nombres:
        c = pd.read_csv(config.RESULTADOS / n / "predicciones_val.csv")
        c["fold"] = json.loads((config.RESULTADOS / n / "config.json").read_text(encoding="utf-8"))["fold"]
        cortes.append(c)
    cortes = pd.concat(cortes)
    cortes["cohorte"] = cortes.patient_id.map(pacientes.dataset)
    return tabla, cortes


def auc_por_agregacion(cortes: pd.DataFrame, uc, umbral_voto: float) -> dict:
    """AUC global fuera de fold con cada agregacion, pasando cada fold a percentiles.
    El voto cuenta los cortes por encima del umbral final (a 0,5 casi ningun corte lo
    supera con la perdida normal)."""
    resultado = {}
    for metodo in METODOS:
        partes = []
        for _, c in cortes.groupby("fold"):
            agregado = uc.agregar_por_paciente(c.patient_id.values, c.prob.values, metodo=metodo,
                                               umbral=umbral_voto)
            agregado = agregado.join(c.groupby("patient_id").pCR.first())
            agregado["percentil"] = agregado.prob.rank(pct=True)
            partes.append(agregado)
        juntas = pd.concat(partes)
        resultado[metodo] = auc(juntas.percentil, juntas.pCR)
    return resultado


def analisis_auc(nombre: str, remuestreos: int, semilla: int) -> dict:
    """AUC global con IC y por cohorte: de results/comparacion/resumen.json si ya esta."""
    ruta = config.RESULTADOS / "comparacion" / "resumen.json"
    if ruta.exists():
        r = json.loads(ruta.read_text(encoding="utf-8")).get(CLAVES_COMPARACION[nombre])
        if r and r["ejecuciones"] == CONFIGURACIONES[nombre]:
            return {"auc_global": r["auc_global"], "por_cohorte": r["por_cohorte"],
                    "fuente": "results/comparacion/resumen.json"}
    g = analizar(CONFIGURACIONES[nombre], remuestreos, semilla)["global"]
    return {"auc_global": g["auc"], "por_cohorte": g["por_cohorte"], "fuente": "analisis_cv.analizar"}


def evaluar(nombre: str, uc, pacientes, entrenamiento, remuestreos: int, semilla: int) -> dict:
    tabla, cortes = cargar(CONFIGURACIONES[nombre], uc, pacientes, entrenamiento)
    t = umbrales_por_fold(tabla)
    umbral = umbral_conjunto(tabla)
    ic = dict(remuestreos=remuestreos, semilla=semilla)
    umbral_por_cohorte = {}
    for c in COHORTES:
        d = tabla[tabla.cohorte == c]
        m = metricas(d.prob, d.pcr, umbral)
        umbral_por_cohorte[c] = {"pacientes": int(len(d)), "pcr": int(d.pcr.sum()),
                          "sensibilidad": m["sensibilidad"], "especificidad": m["especificidad"],
                          "VP": m["VP"], "FP": m["FP"]}
    especificidad_por_fold = {int(f): metricas(d.prob, d.pcr, umbral)["especificidad"]
                              for f, d in tabla.groupby("fold")}
    return {
        "ejecuciones": CONFIGURACIONES[nombre],
        **analisis_auc(nombre, remuestreos, semilla),
        "umbral_final": umbral,
        "regla": "percentil 90 (method='higher') de la probabilidad por paciente de todas las pCR=0 "
                 "fuera de fold; pCR si prob > umbral_final",
        "informativo": {
            "t_k": t,
            "t_k_dispersion": {"min": min(t.values()), "max": max(t.values()),
                               "mediana": float(np.median(list(t.values()))),
                               "desviacion": float(np.std(list(t.values()), ddof=1))},
            "especificidad_por_fold_con_umbral_final": especificidad_por_fold,
        },
        "pacientes": {"n": int(len(tabla)), "pcr": int(tabla.pcr.sum())},
        "por_paciente": con_ic(tabla.prob, tabla.pcr, tabla.index, umbral, **ic),
        "siempre_no_pcr": con_ic(tabla.prob, tabla.pcr, tabla.index, None, **ic),
        "umbral_por_cohorte": umbral_por_cohorte,
        "cortes_sueltos": {"n": int(len(cortes)),
                           **con_ic(cortes.prob, cortes.pCR, cortes.patient_id, umbral, **ic)},
        "brier": brier(tabla.prob, tabla.pcr),
        "brier_constante": brier(np.full(len(tabla), tabla.pcr.mean()), tabla.pcr),
        "fiabilidad": fiabilidad(tabla.prob, tabla.pcr).to_dict(orient="records"),
        "auc_por_agregacion": auc_por_agregacion(cortes, uc, umbral),
    }


def dibujar_calibracion(resultados: dict):
    plt = figuras.plt          # figuras ya importa torch antes que matplotlib

    figura, ejes = plt.subplots(1, 2, figsize=(12, 5))
    a, b = ejes
    for (nombre, r), color in zip(resultados.items(), (figuras.AZUL, figuras.NARANJA)):
        f = pd.DataFrame(r["fiabilidad"])
        a.plot(f.prob_media, f.observada, "o-", color=color,
               label=f"{nombre} (Brier {r['brier']:.4f}; constante {r['brier_constante']:.4f})")
        a.axvline(r["umbral_final"], color=color, ls=":", lw=1)
    a.plot([0, 1], [0, 1], color=figuras.GRIS, lw=1, label="calibracion perfecta")
    a.set(title="Fiabilidad por paciente (10 grupos por cuantiles; punteada: umbral)",
          xlabel="probabilidad media predicha", ylabel="fraccion observada de pCR",
          xlim=(0, 0.8), ylim=(0, 0.8))
    a.legend(fontsize=8)
    for (nombre, r), color in zip(resultados.items(), (figuras.AZUL, figuras.NARANJA)):
        f = pd.DataFrame(r["fiabilidad"])
        desplazamiento = -0.2 if nombre == "normal" else 0.2
        b.bar(np.arange(len(f)) + desplazamiento, f.observada - f.prob_media, width=0.4,
              color=color, label=nombre)
    b.axhline(0, color=figuras.GRIS, lw=1)
    b.set(title="Observada - predicha por grupo", xlabel="grupo (de menor a mayor probabilidad)",
          ylabel="diferencia", xticks=range(10))
    b.legend(fontsize=8)
    figura.tight_layout()
    return figura


# --------------------------------------------------------------------------- #
# Consola
# --------------------------------------------------------------------------- #

def imprimir(r: dict) -> None:
    def ic(m, k):
        v, (lo, hi) = m[k], m["ic95"][k]
        return f"{v:.3f} [{lo:.3f}-{hi:.3f}]" if not np.isnan(v) else "—"

    for nombre in ("normal", "ponderada"):
        x = r[nombre]
        g = x["auc_global"]
        print(f"\n=== PERDIDA {nombre.upper()}  ({x['pacientes']['n']} pacientes fuera de fold, "
              f"{x['pacientes']['pcr']} pCR)")
        print(f"AUC global fuera de fold: {g['auc']:.3f} [{g['ic95'][0]:.3f}-{g['ic95'][1]:.3f}]  "
              f"(de {x['fuente']})")
        print("  por cohorte: " + ", ".join(
            f"{c} {m['auc']:.3f} [{m['ic95'][0]:.3f}-{m['ic95'][1]:.3f}]"
            for c, m in x["por_cohorte"].items()))
        info = x["informativo"]
        negativos = x["pacientes"]["n"] - x["pacientes"]["pcr"]
        print(f"UMBRAL FINAL (percentil 90 de las {negativos} pCR=0 juntas): {x['umbral_final']:.4f}")
        print("  informativo, t_k por fold: " + ", ".join(f"f{f} {v:.4f}" for f, v in info["t_k"].items())
              + f"  (min {info['t_k_dispersion']['min']:.4f}, max {info['t_k_dispersion']['max']:.4f}, "
                f"desv. {info['t_k_dispersion']['desviacion']:.4f})")
        print("  informativo, especificidad por fold con el umbral final: "
              + ", ".join(f"f{f} {v:.3f}"
                          for f, v in info["especificidad_por_fold_con_umbral_final"].items()))
        print(f"{'':<22}{'por paciente (media)':>24}{'siempre no pCR':>24}{'corte suelto':>24}")
        p, s, c = x["por_paciente"], x["siempre_no_pcr"], x["cortes_sueltos"]
        matrices = ["/".join(str(m[k]) for k in ("VP", "FP", "FN", "VN")) for m in (p, s, c)]
        print(f"{'matriz VP/FP/FN/VN':<22}" + "".join(f"{v:>24}" for v in matrices))
        for k, titulo in (("sensibilidad", "sensibilidad"), ("especificidad", "especificidad"),
                          ("VPP", "VPP"), ("VPN", "VPN"), ("pcr1_f1", "F1 pCR=1"), ("pcr0_f1", "F1 pCR=0"),
                          ("f1_macro", "F1 macro"), ("exactitud", "exactitud")):
            fila = []
            for m in (p, s, c):
                plana = {**plano(m), "ic95": m["ic95"]}
                fila.append(ic(plana, k))
            print(f"{titulo:<22}" + "".join(f"{v:>24}" for v in fila))
        print(f"{'':<22}precision/recall pCR=1: {p['pcr1']['precision']:.3f}/{p['pcr1']['recall']:.3f};  "
              f"pCR=0: {p['pcr0']['precision']:.3f}/{p['pcr0']['recall']:.3f}")
        print("por cohorte (paciente):  " + ";  ".join(
            f"{k} sens {v['sensibilidad']:.3f} esp {v['especificidad']:.3f} ({v['pacientes']} pac., "
            f"{v['pcr']} pCR)" for k, v in x["umbral_por_cohorte"].items()))
        print(f"Brier por paciente: {x['brier']:.4f} (constante {x['brier_constante']:.4f})")
        print("AUC por agregacion (percentiles por fold): " + ", ".join(
            f"{k} {v:.3f}" for k, v in x["auc_por_agregacion"].items()) + "  (voto con el umbral final)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remuestreos", type=int, default=2000)
    p.add_argument("--semilla", type=int, default=config.DEFECTO["semilla"])
    a = p.parse_args()

    perdidos = [f"{n}/predicciones_val.csv" for ns in CONFIGURACIONES.values() for n in ns
                if not (config.RESULTADOS / n / "predicciones_val.csv").exists()]
    if perdidos:
        sys.exit("Faltan ficheros:\n  " + "\n  ".join(perdidos))

    uc = config.utils()
    pacientes = uc.cargar_patients().set_index("pid")
    entrenamiento = pd.concat(uc.particion(uc.cargar_samples(), fold_val=0))   # sin test
    resumen = {"criterio": f"especificidad >= {ESPECIFICIDAD} por paciente (media de sus cortes)",
               "nota": "la regla inicial (mediana de los t_k por fold, 0,3393) daba especificidad 0,839 por "
                       "la diferencia de escala entre folds; se corrigio el procedimiento, no el criterio",
               **{n: evaluar(n, uc, pacientes, entrenamiento, a.remuestreos, a.semilla)
                  for n in CONFIGURACIONES}}

    destino = config.RESULTADOS / "umbral"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float), encoding="utf-8")
    figura = dibujar_calibracion({n: resumen[n] for n in CONFIGURACIONES})
    ruta = figuras.guardar(figura, "umbral_calibracion")
    imprimir(resumen)
    print(f"\nGuardado en {(destino / 'resumen.json').relative_to(config.REPO)} y "
          f"{ruta.relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
