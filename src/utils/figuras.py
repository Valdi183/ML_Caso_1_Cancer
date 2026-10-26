"""Lectura de resultados y figuras de la comparativa de modelos.

Lo usa `notebooks/comparativa_modelos.ipynb`: aqui estan los calculos (probados en
tests/test_figuras.py) y cada figura como una funcion que devuelve la `Figure`; el
cuaderno solo las llama, las muestra y guarda las del informe con `guardar`.

Solo lee lo que ya hay en results/ y models/ y las imagenes de validacion. Test NO.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# isort: off
import torch  # antes que matplotlib: al reves, en Windows chocan sus DLL
import matplotlib.pyplot as plt
# isort: on

from src import config

# --------------------------------------------------------------------------- #
# Que ejecuciones entran (docs/iteraciones.md) y como se agrupan
# --------------------------------------------------------------------------- #

def _folds(base: str, folds=range(5)) -> list[str]:
    return [f"{base}_f{k}" for k in folds]


EJECUCIONES: dict[str, list[str]] = {
    "1": ["it1_normal_f0", "it1_ponderada_f0"],
    "2": ["it2_pequena_reg_f0", "it2_lr1e-4_f0"],
    "3": ["it3_pequena_realce_f0", "it3_pequena_norm_f0"],
    "4": _folds("it4_pequena_reg", range(1, 5)),
    "5": _folds("it5_red128") + _folds("it5_atencion"),
    "6": [n for v in ("3bloques", "5bloques", "ancha", "mini", "mediamax", "64px")
          for n in _folds(f"it6_{v}", (0, 1))],
    "7": _folds("it7_3b_atencion", (0, 1)) + _folds("it7_3b_atsup", (0, 1)),
    "8": (_folds("it8_3bloques", (2, 3, 4)) + _folds("it8_3bloques_pond")
          + _folds("it8_triple_atencion", (0, 1)) + _folds("it8_triple_atsup", (0, 1))),
    "9": _folds("it9_3b_atsup", (2, 3, 4)),
    "control": ["objduke_normal_f0", "objtriple_normal_f0"],
}

#: Configuraciones con los 5 folds (una ejecucion por fold, en orden)
CINCO_FOLDS: dict[str, list[str]] = {
    "2A": ["it2_pequena_reg_f0", *_folds("it4_pequena_reg", range(1, 5))],
    "it5_red128": _folds("it5_red128"),
    "it5_atencion": _folds("it5_atencion"),
    "congelado": ["it6_3bloques_f0", "it6_3bloques_f1", *_folds("it8_3bloques", (2, 3, 4))],
    "congelado_ponderada": _folds("it8_3bloques_pond"),
    "atencion_supervisada": ["it7_3b_atsup_f0", "it7_3b_atsup_f1", *_folds("it9_3b_atsup", (2, 3, 4))],
}
CONGELADO = "congelado"

#: Todas las configuraciones de pCR que tienen los folds 0 y 1
FOLDS_0_1: dict[str, list[str]] = {
    **{k: v[:2] for k, v in CINCO_FOLDS.items()},
    **{f"it6_{v}": _folds(f"it6_{v}", (0, 1)) for v in ("5bloques", "ancha", "mini", "mediamax", "64px")},
    "it7_atencion_libre": _folds("it7_3b_atencion", (0, 1)),
}

#: Valores por defecto de entrenar.py para las ejecuciones antiguas que no los guardaban
DEFECTOS = {"filtros": [16, 32, 64, 128], "reducir": 1, "pooling": "media", "ponderada": False,
            "peso_caja": 0.0, "objetivo": "pcr", "realce": False, "normalizar": False,
            "aumentado": False, "lr": 1e-3, "wd": 1e-4, "dropout": 0.5}

COLORES = ["#2B6CB0", "#DD6B20", "#2F855A", "#C53030", "#6B46C1", "#B7791F", "#319795", "#718096"]
AZUL, NARANJA, GRIS, ROJO = "#2B6CB0", "#DD6B20", "#718096", "#C53030"


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #

def faltan(resultados: Path = config.RESULTADOS) -> list[str]:
    """Ejecuciones de EJECUCIONES sin carpeta en results/."""
    return [n for ns in EJECUCIONES.values() for n in ns if not (resultados / n / "historial.csv").exists()]


def leer_ejecucion(nombre: str, resultados: Path = config.RESULTADOS) -> dict:
    """historial (DataFrame), config (con los valores por defecto que faltaran),
    metricas y predicciones de validacion por corte de la epoca elegida."""
    carpeta = resultados / nombre
    cfg = {**DEFECTOS, **json.loads((carpeta / "config.json").read_text(encoding="utf-8"))}
    return {
        "nombre": nombre,
        "historial": pd.read_csv(carpeta / "historial.csv"),
        "config": cfg,
        "metricas": json.loads((carpeta / "metricas.json").read_text(encoding="utf-8")),
        "predicciones": pd.read_csv(carpeta / "predicciones_val.csv"),
    }


def auc_medio(historial: pd.DataFrame, hasta: int | None = None, columna: str = "auc_val") -> float:
    """Media por epoca de `columna` (todas las epocas, o las 1..hasta). NaN si no existe."""
    if columna not in historial:
        return float("nan")
    filas = historial if hasta is None else historial[historial.epoca <= hasta]
    return float(filas[columna].mean()) if len(filas) else float("nan")


def en_epoca(historial: pd.DataFrame, columna: str, epoca: int) -> float:
    if columna not in historial or epoca not in set(historial.epoca):
        return float("nan")
    return float(historial.loc[historial.epoca == epoca, columna].iloc[0])


def perdida_constante(prevalencia: float, pos_weight: float | None = None) -> float:
    """Perdida de validacion del mejor modelo constante, en la misma escala que perdida_val.

    Normal: la entropia de la prevalencia q (predecir siempre q). Ponderada (BCE con
    pos_weight w, media por corte): el optimo es p = wq / (wq + 1 - q); con w = N0/N1
    da p = 0,5 y la perdida ln 2 · (wq + 1 - q), el ~0,979 de pCR.
    """
    q, w = float(prevalencia), 1.0 if pos_weight is None else float(pos_weight)
    p = w * q / (w * q + 1 - q)
    return float(-(w * q * np.log(p) + (1 - q) * np.log(1 - p)))


def constante_de(ejecucion: dict) -> float:
    """Perdida constante de una ejecucion: prevalencia de sus cortes de validacion y,
    si es ponderada, su pos_weight."""
    cfg = ejecucion["config"]
    return perdida_constante(ejecucion["predicciones"].pCR.mean(),
                             cfg.get("pos_weight") if cfg["ponderada"] else None)


def tipo_perdida(cfg: dict) -> str:
    """Ejecuciones con la misma escala de perdida (comparten ejes): normal, ponderada
    o el objetivo de diagnostico."""
    if cfg["objetivo"] != "pcr":
        return cfg["objetivo"]
    return "ponderada" if cfg["ponderada"] else "normal"


def etiqueta(cfg: dict) -> str:
    """Configuracion en pocas palabras (lo que cambia respecto a los valores por defecto)."""
    partes = ["-".join(map(str, cfg["filtros"])), f"{256 // cfg['reducir']} px", cfg["pooling"],
              "ponderada" if cfg["ponderada"] else "normal"]
    if cfg["peso_caja"] > 0:
        partes.append(f"caja {cfg['peso_caja']:g}")
    if cfg["objetivo"] != "pcr":
        partes.append(f"obj {cfg['objetivo']}")
    for clave, texto in (("aumentado", "aum"), ("realce", "realce"), ("normalizar", "normalizar")):
        if cfg[clave]:
            partes.append(texto)
    for clave in ("lr", "wd", "dropout"):
        if cfg[clave] != DEFECTOS[clave]:
            partes.append(f"{clave} {cfg[clave]:g}")
    return " · ".join(partes)


def fila_inventario(ejecucion: dict, iteracion: str) -> dict:
    h, cfg, m = ejecucion["historial"], ejecucion["config"], ejecucion["metricas"]
    return {
        "iteracion": iteracion, "nombre": ejecucion["nombre"], "configuracion": etiqueta(cfg),
        "filtros": "-".join(map(str, cfg["filtros"])), "reducir": cfg["reducir"],
        "pooling": cfg["pooling"], "ponderada": bool(cfg["ponderada"]),
        "peso_caja": float(cfg["peso_caja"]), "objetivo": cfg["objetivo"],
        "parametros": cfg.get("parametros"), "fold": cfg["fold"], "epocas": int(h.epoca.max()),
        "epoca_elegida": m["mejor_epoca"],
        "auc_val_medio": auc_medio(h), "auc_val_medio_1_15": auc_medio(h, 15),
        "auc_elegido": m["auc_val_media"], "auc_train_15": en_epoca(h, "auc_train", 15),
        "perdida_val_min": float(h.perdida_val.min()),
        "perdida_constante": constante_de(ejecucion),
    }


def inventario(resultados: Path = config.RESULTADOS) -> pd.DataFrame:
    """Una fila por ejecucion existente de EJECUCIONES (las que faltan: `faltan`)."""
    filas = [fila_inventario(leer_ejecucion(n, resultados), it)
             for it, nombres in EJECUCIONES.items() for n in nombres
             if (resultados / n / "historial.csv").exists()]
    return pd.DataFrame(filas)


def por_paciente(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad por paciente (media de sus cortes), su desviacion y numero de cortes."""
    return predicciones.groupby("patient_id").agg(
        prob=("prob", "mean"), desviacion=("prob", "std"), cortes=("prob", "size"), pcr=("pCR", "first"))


def sens_esp(prob, pcr, umbral: float = 0.5) -> tuple[float, float]:
    """Sensibilidad y especificidad prediciendo pCR cuando prob >= umbral."""
    prob, pcr = np.asarray(prob, dtype=float), np.asarray(pcr).astype(bool)
    positivo = prob >= umbral
    sens = positivo[pcr].mean() if pcr.any() else float("nan")
    esp = (~positivo[~pcr]).mean() if (~pcr).any() else float("nan")
    return float(sens), float(esp)


def resumen_configuracion(inv: pd.DataFrame, nombres: list[str], columna: str = "auc_val_medio") -> dict:
    """Media, desviacion y valores por fold de `columna` para un grupo de ejecuciones."""
    v = inv.set_index("nombre").loc[nombres, columna].astype(float).values
    return {"media": float(v.mean()), "desviacion": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "valores": v.tolist()}


def solo_cohorte(resultados: Path = config.RESULTADOS) -> dict | None:
    """Referencia «solo cohorte» de results/analisis_* (no depende del modelo, solo de
    los folds): media de los 5 folds y de los folds 0 y 1. None si no hay analisis."""
    for nombre in ("analisis_it5_red128", "analisis_2A", "analisis_it5_atencion"):
        ruta = resultados / nombre / "resumen.json"
        if ruta.exists():
            r = json.loads(ruta.read_text(encoding="utf-8"))
            folds = {int(k): v["solo_cohorte"] for k, v in r["folds"].items()}
            if len(folds) == 5:
                return {"cinco": float(np.mean(list(folds.values()))),
                        "f01": float(np.mean([folds[0], folds[1]]))}
    return None


def guardar(figura, nombre: str, carpeta: Path = config.FIGURAS) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.png"
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def limites_perdida(ejecuciones: list[dict]) -> dict[str, tuple[float, float]]:
    """Mismos ejes de perdida para las ejecuciones con la misma escala (tipo_perdida)."""
    rangos: dict[str, list[float]] = {}
    for e in ejecuciones:
        h = e["historial"]
        valores = pd.concat([h.perdida_train, h.perdida_val])
        r = rangos.setdefault(tipo_perdida(e["config"]), [np.inf, -np.inf])
        r[0], r[1] = min(r[0], valores.min()), max(r[1], valores.max())
    return {k: (a - 0.03 * (b - a), b + 0.03 * (b - a)) for k, (a, b) in rangos.items()}


def curvas_ejecucion(ejes, e: dict, limites: dict) -> None:
    """Los 3 paneles de una ejecucion: perdida, AUC y lr."""
    h, cfg, m = e["historial"], e["config"], e["metricas"]
    elegida = m["mejor_epoca"]
    constante = constante_de(e)
    ax = ejes[0]
    ax.plot(h.epoca, h.perdida_train, color=AZUL, label="train")
    ax.plot(h.epoca, h.perdida_val, color=NARANJA, label="val")
    ax.axhline(constante, color=GRIS, ls=":", lw=1, label=f"constante {constante:.3f}")
    ax.set(ylim=limites[tipo_perdida(cfg)], ylabel=e["nombre"])
    ax = ejes[1]
    if "auc_train" in h:
        ax.plot(h.epoca, h.auc_train, color=AZUL, label="train")
    ax.plot(h.epoca, h.auc_val, color=NARANJA, alpha=0.45, lw=1, label="val")
    if "auc_val_suave" in h:
        ax.plot(h.epoca, h.auc_val_suave, color=NARANJA, lw=2, label="val suavizado")
    ax.axhline(0.5, color=ROJO, ls=":", lw=1)
    ax.axvline(elegida, color=GRIS, ls="--", lw=1)
    ax.set(ylim=(0.3, 1.0))
    ax.text(0.98, 0.04, f"elegida {elegida}: val {m['auc_val_media']:.3f}", transform=ax.transAxes,
            ha="right", fontsize=8, color=GRIS)
    ax = ejes[2]
    ax.plot(h.epoca, h.lr, color=GRIS)
    ax.set(yscale="log")
    ejes[0].set_title(etiqueta(cfg), fontsize=8, loc="left")


def curvas_iteracion(ejecuciones: list[dict], limites: dict, titulo: str):
    """Una fila de 3 paneles (perdida, AUC, lr) por ejecucion."""
    figura, ejes = plt.subplots(len(ejecuciones), 3, figsize=(14, 2.6 * len(ejecuciones)),
                                squeeze=False, sharex="row")
    for fila, e in zip(ejes, ejecuciones):
        curvas_ejecucion(fila, e, limites)
    ejes[0, 0].legend(fontsize=7, loc="upper right")
    ejes[0, 1].legend(fontsize=7, loc="upper left")
    for ax, t in zip(ejes[0], ("Perdida", "AUC por paciente", "lr")):
        ax.text(0.5, 1.12, t, transform=ax.transAxes, ha="center", fontsize=10, weight="bold")
    for ax in ejes[-1]:
        ax.set_xlabel("epoca")
    figura.suptitle(titulo, y=1.0, fontsize=12)
    figura.tight_layout()
    return figura


def curvas_superpuestas(ejecuciones: list[dict], titulo: str):
    """Los folds de una configuracion juntos: AUC de train (discontinua) y val, y perdida.
    El punto marca la epoca elegida de cada fold."""
    figura, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2))
    for color, e in zip(COLORES, ejecuciones):
        h, el, fold = e["historial"], e["metricas"]["mejor_epoca"], e["config"]["fold"]
        if "auc_train" in h:
            a.plot(h.epoca, h.auc_train, color=color, ls="--", lw=1)
        a.plot(h.epoca, h.auc_val, color=color, lw=1.6, label=f"fold {fold}")
        a.plot(el, en_epoca(h, "auc_val", el), "o", color=color, ms=7, mec="black")
        b.plot(h.epoca, h.perdida_train, color=color, ls="--", lw=1)
        b.plot(h.epoca, h.perdida_val, color=color, lw=1.6)
        b.plot(el, en_epoca(h, "perdida_val", el), "o", color=color, ms=7, mec="black")
    e0 = ejecuciones[0]
    constante = float(np.mean([constante_de(e) for e in ejecuciones]))
    a.axhline(0.5, color=ROJO, ls=":", lw=1)
    a.set(title="AUC por paciente (discontinua: train; continua: val; punto: elegida)",
          xlabel="epoca", ylim=(0.3, 1.0))
    a.legend(fontsize=8)
    b.axhline(constante, color=GRIS, ls=":", lw=1, label=f"constante {constante:.3f}")
    b.set(title="Perdida (discontinua: train; continua: val)", xlabel="epoca")
    b.legend(fontsize=8)
    figura.suptitle(f"{titulo}  ·  {etiqueta(e0['config'])}", fontsize=11)
    figura.tight_layout()
    return figura


def resumen_configuraciones(inv: pd.DataFrame, referencia: dict | None):
    """AUC de val medio (todas las epocas) por configuracion: (1) las de 5 folds;
    (2) todas en los folds 0 y 1. Cada punto pequeño es un fold."""
    figura, ejes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.7]})
    for ax, grupos, clave, titulo in ((ejes[0], CINCO_FOLDS, "cinco", "5 folds"),
                                      (ejes[1], FOLDS_0_1, "f01", "mismos folds 0 y 1")):
        orden = sorted(grupos, key=lambda k: resumen_configuracion(inv, grupos[k])["media"])
        for y, nombre in enumerate(orden):
            r = resumen_configuracion(inv, grupos[nombre])
            color = ROJO if nombre == CONGELADO else AZUL
            ax.errorbar(r["media"], y, xerr=r["desviacion"], fmt="o", color=color, capsize=3, ms=7)
            ax.plot(r["valores"], [y] * len(r["valores"]), "|", color=color, ms=10, alpha=0.6)
            derecha = max(r["media"] + r["desviacion"], *r["valores"])
            ax.text(derecha + 0.002, y, f"{r['media']:.3f}", va="center", fontsize=8, color=color)
        ax.set_yticks(range(len(orden)), orden)
        for marca, nombre in zip(ax.get_yticklabels(), orden):
            if nombre == CONGELADO:
                marca.set_color(ROJO)
                marca.set_fontweight("bold")
        ax.axvline(0.5, color=GRIS, ls=":", lw=1, label="azar")
        if referencia:
            ax.axvline(referencia[clave], color=NARANJA, ls="--", lw=1,
                       label=f"solo cohorte {referencia[clave]:.3f}")
        ax.set(title=f"AUC de val medio de todas las epocas ({titulo})", xlabel="AUC", xlim=(0.47, 0.62))
        ax.legend(fontsize=8, loc="upper left")
    figura.tight_layout()
    return figura


def capacidad_memorizacion(inv: pd.DataFrame):
    """Parametros frente a AUC de train en la epoca 15 y AUC de val medio (media de f0 y f1;
    el congelado tambien con sus 5 folds). Cuadrados: el mapa final ve la imagen entera."""
    grupos = {"base 128 px": FOLDS_0_1["it5_red128"],
              **{k.replace("it6_", ""): v for k, v in FOLDS_0_1.items() if k.startswith("it6_")},
              "3bloques (congelado)": FOLDS_0_1[CONGELADO]}
    entera = {"5bloques", "64px"}
    figura, ejes = plt.subplots(1, 2, figsize=(13, 4.6))
    t = inv.set_index("nombre")
    for nombre, nombres in grupos.items():
        x = float(t.loc[nombres[0], "parametros"])
        color = ROJO if "congelado" in nombre else (NARANJA if nombre in entera else AZUL)
        marca = "s" if nombre in entera else "o"
        for ax, col in zip(ejes, ("auc_train_15", "auc_val_medio")):
            y = float(t.loc[nombres, col].mean())
            ax.plot(x, y, marca, color=color, ms=9)
            ax.annotate(nombre, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    r = resumen_configuracion(inv, CINCO_FOLDS[CONGELADO])
    x = float(t.loc[CINCO_FOLDS[CONGELADO][0], "parametros"])
    ejes[1].errorbar(x, r["media"], yerr=r["desviacion"], fmt="*", color=ROJO, ms=12, capsize=3,
                     label="congelado, 5 folds")
    ejes[0].set(title="Memorizacion: AUC de train en la epoca 15", xscale="log", xlabel="parametros (log)",
                ylabel="AUC train")
    ejes[1].set(title="AUC de val medio de todas las epocas", xscale="log", xlabel="parametros (log)",
                ylabel="AUC val")
    ejes[1].axhline(0.5, color=GRIS, ls=":", lw=1)
    ejes[1].legend(fontsize=8)
    figura.text(0.5, -0.02, "circulo: mapa final de 16x16 o 8x8   ·   cuadrado naranja: el mapa final ve la "
                "imagen entera (5 bloques, 64 px)   ·   media de los folds 0 y 1", ha="center", fontsize=8)
    figura.tight_layout()
    return figura


def normal_vs_ponderada(normal: list[dict], ponderada: list[dict]):
    """Probabilidad por paciente fuera de fold por clase con cada perdida, y sensibilidad
    y especificidad a 0,5 por fold."""
    figura, ejes = plt.subplots(1, 3, figsize=(16, 4.4))
    tasa = float(pd.concat([por_paciente(e["predicciones"]) for e in normal]).pcr.mean())
    for ax, grupo, titulo in ((ejes[0], normal, "Perdida normal"), (ejes[1], ponderada, "Perdida ponderada")):
        p = pd.concat([por_paciente(e["predicciones"]) for e in grupo])
        bordes = np.linspace(0, 1, 41)
        ax.hist(p.prob[p.pcr == 0], bins=bordes, alpha=0.6, color=AZUL, density=True, label="pCR = 0")
        ax.hist(p.prob[p.pcr == 1], bins=bordes, alpha=0.6, color=NARANJA, density=True, label="pCR = 1")
        ax.axvline(0.5, color="black", ls="--", lw=1, label="0,5")
        ax.axvline(tasa, color=ROJO, ls=":", lw=1.5, label=f"tasa de pCR {tasa:.3f}")
        ax.set(title=f"{titulo}\nprobabilidad por paciente ({len(p)} pac., fuera de fold)",
               xlabel="probabilidad de pCR", ylabel="densidad")
        ax.legend(fontsize=8)
    ax = ejes[2]
    for desplaz, grupo, color, nombre in ((-0.15, normal, AZUL, "normal"),
                                          (0.15, ponderada, NARANJA, "ponderada")):
        for e in grupo:
            p = por_paciente(e["predicciones"])
            s, es = sens_esp(p.prob, p.pcr)
            f = e["config"]["fold"]
            ax.plot(f + desplaz, s, "o", color=color, ms=8,
                    label=f"sensibilidad ({nombre})" if f == 0 else None)
            ax.plot(f + desplaz, es, "s", color=color, ms=8, mfc="white",
                    label=f"especificidad ({nombre})" if f == 0 else None)
    ax.set(title="Umbral 0,5\nsensibilidad y especificidad por paciente y fold", xlabel="fold",
           ylim=(-0.05, 1.05), xticks=range(5))
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    figura.tight_layout()
    return figura


def cohortes_y_hospital(resumenes: dict[str, dict], ejecuciones: dict[str, list[dict]]):
    """(1) AUC fuera de fold por cohorte con IC 95 % (analisis_cv.analizar);
    (2) por epoca, AUC del mismo hospital frente al AUC global."""
    figura, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8))
    cohortes = ("spy1", "spy2", "duke")
    for i, (nombre, r) in enumerate(resumenes.items()):
        for j, c in enumerate(cohortes):
            m = r["global"]["por_cohorte"][c]
            x = j + (i - (len(resumenes) - 1) / 2) * 0.22
            a.errorbar(x, m["auc"], yerr=[[m["auc"] - m["ic95"][0]], [m["ic95"][1] - m["auc"]]], fmt="o",
                       color=COLORES[i], capsize=3, label=nombre if j == 0 else None)
    primero = next(iter(resumenes.values()))["global"]["por_cohorte"]
    a.set_xticks(range(3), [f"{c}\n({primero[c]['pacientes']} pac.)" for c in cohortes])
    a.axhline(0.5, color=GRIS, ls=":", lw=1)
    a.set(title="AUC fuera de fold por cohorte, IC 95 % (epoca elegida)", ylabel="AUC")
    a.legend(fontsize=8)
    for i, (nombre, grupo) in enumerate(ejecuciones.items()):
        h = pd.concat([e["historial"] for e in grupo if "auc_val_mismo_hospital" in e["historial"]])
        if len(h):
            b.plot(h.auc_val, h.auc_val_mismo_hospital, "o", ms=4, alpha=0.6, color=COLORES[i], label=nombre)
    lim = (0.35, 0.72)
    b.plot(lim, lim, color=GRIS, lw=1, label="mismo hospital = global")
    b.axhline(0.5, color=ROJO, ls=":", lw=1)
    b.set(title="Por epoca: AUC del mismo hospital frente al global (val)", xlabel="AUC val global",
          ylabel="AUC val mismo hospital", xlim=lim, ylim=lim)
    b.legend(fontsize=8)
    figura.tight_layout()
    return figura


def atencion_por_epoca(grupos: dict[str, list[dict]]):
    """Masa de atencion dentro de la caja (continua) frente a la fraccion del mapa que
    ocupa la caja (discontinua), en validacion, por epoca."""
    figura, ejes = plt.subplots(1, len(grupos), figsize=(4.6 * len(grupos), 4), sharey=True, squeeze=False)
    for ax, (nombre, grupo) in zip(ejes[0], grupos.items()):
        for color, e in zip(COLORES, grupo):
            h = e["historial"]
            ax.plot(h.epoca, h.masa_caja_val, color=color, lw=1.6, label=f"fold {e['config']['fold']}")
            ax.plot(h.epoca, h.fraccion_caja_val, color=color, ls="--", lw=1)
        ax.set(title=nombre, xlabel="epoca", ylim=(0, 1))
        ax.legend(fontsize=8)
    ejes[0, 0].set_ylabel("masa en la caja (continua) / area de la caja (discontinua)")
    figura.tight_layout()
    return figura


def elegir_pacientes(filas: pd.DataFrame, cajas: dict, n_por_clase: int = 3,
                     semilla: int = 42) -> pd.DataFrame:
    """n pacientes con caja de cada clase y, de cada una, su corte central."""
    con_caja = filas[filas.patient_id.map(lambda p: cajas.get(p) is not None)]
    pcr = con_caja.groupby("patient_id").pCR.first()
    rng = np.random.default_rng(semilla)
    elegidas = [p for clase in (1, 0)
                for p in rng.choice(pcr.index[pcr == clase], n_por_clase, replace=False)]
    cortes = con_caja[con_caja.patient_id.isin(elegidas)].sort_values(["patient_id", "sample_id"])
    posicion = cortes.groupby("patient_id").cumcount()
    total = cortes.groupby("patient_id").patient_id.transform("size")
    centrales = cortes[posicion.values == (total // 2).values].set_index("patient_id")
    return centrales.loc[elegidas].reset_index()


def mapas_atencion(redes: dict[str, torch.nn.Module], filas: pd.DataFrame, imagenes: np.ndarray, cajas: list):
    """Por paciente: EARLY-PRE con la caja, y el mapa de atencion de cada red encima."""
    import matplotlib.patches as mpatches
    import torch.nn.functional as F

    x = torch.from_numpy(imagenes).float()
    salidas = {}
    with torch.no_grad():
        for nombre, red in redes.items():
            logits, pesos = red.con_atencion(x)
            mapa = F.interpolate(pesos[:, None], size=imagenes.shape[-2:], mode="nearest")[:, 0]
            salidas[nombre] = (torch.sigmoid(logits).numpy(), mapa.numpy())
    columnas = 1 + len(redes)
    figura, ejes = plt.subplots(len(filas), columnas, figsize=(3.3 * columnas, 3.3 * len(filas)))
    for i, fila in filas.iterrows():
        realce = imagenes[i, 1] - imagenes[i, 0]
        f0, f1, c0, c1 = cajas[i]
        for j, ax in enumerate(ejes[i]):
            ax.imshow(realce, cmap="gray", vmin=-0.1, vmax=max(0.3, float(realce.max())))
            ax.add_patch(mpatches.Rectangle((c0, f0), c1 - c0, f1 - f0, fill=False, ec="#38A169", lw=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(f"{fila.patient_id} · pCR={int(fila.pCR)}\nEARLY-PRE y caja", fontsize=9)
            else:
                nombre = list(redes)[j - 1]
                prob, mapa = salidas[nombre]
                ax.imshow(mapa[i], cmap="magma", alpha=0.55)
                ax.set_title(f"{nombre}\nP(pCR) del corte {prob[i]:.2f}", fontsize=9)
    figura.tight_layout()
    return figura


def controles(positivo: dict, negativo: dict):
    """Control positivo (cohorte Duke) y negativo (subtipo triple) lado a lado."""
    figura, ejes = plt.subplots(2, 2, figsize=(13, 7.5))
    for col, (e, titulo) in enumerate(((positivo, "Control positivo: cohorte Duke"),
                                       (negativo, "Control negativo: subtipo triple"))):
        h, m = e["historial"], e["metricas"]
        constante = perdida_constante(e["predicciones"].pCR.mean())
        ax = ejes[0, col]
        ax.plot(h.epoca, h.perdida_train, color=AZUL, label="train")
        ax.plot(h.epoca, h.perdida_val, color=NARANJA, label="val")
        ax.axhline(constante, color=GRIS, ls=":", lw=1, label=f"constante {constante:.3f}")
        ax.set(title=f"{titulo}\nperdida", xlabel="epoca")
        ax.legend(fontsize=8)
        ax = ejes[1, col]
        if "auc_train" in h:
            ax.plot(h.epoca, h.auc_train, color=AZUL, label="train")
        ax.plot(h.epoca, h.auc_val, color=NARANJA, label="val")
        ax.axhline(0.5, color=ROJO, ls=":", lw=1)
        ax.axvline(m["mejor_epoca"], color=GRIS, ls="--", lw=1)
        ax.set(title=f"AUC por paciente (elegida {m['mejor_epoca']}: {m['auc_val_media']:.3f})",
               xlabel="epoca", ylim=(0.3, 1.02))
        ax.legend(fontsize=8)
    red = etiqueta({**positivo["config"], "objetivo": "pcr"}).replace(" · normal", "")
    figura.text(0.5, -0.01, f"Los dos: {red}, perdida normal  ·  sin AUC de train registrado (anteriores a "
                "la iteracion 1)", ha="center", fontsize=8)
    figura.tight_layout()
    return figura


def corte_vs_paciente(ejecuciones: list[dict]):
    """Probabilidad de cada corte frente a la media de su paciente, y cuanto varian las
    probabilidades de los cortes de una misma paciente."""
    cortes = pd.concat([e["predicciones"] for e in ejecuciones])
    pacientes = pd.concat([por_paciente(e["predicciones"]) for e in ejecuciones])
    figura, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2))
    bordes = np.linspace(0, max(0.8, float(cortes.prob.max())), 50)
    a.hist(cortes.prob, bins=bordes, density=True, alpha=0.55, color=AZUL, label=f"corte ({len(cortes)})")
    a.hist(pacientes.prob, bins=bordes, density=True, alpha=0.55, color=NARANJA,
           label=f"media por paciente ({len(pacientes)})")
    a.set(title="Probabilidad de pCR: corte suelto frente a paciente", xlabel="probabilidad",
          ylabel="densidad")
    a.legend(fontsize=8)
    b.hist(pacientes.desviacion.dropna(), bins=40, color=GRIS)
    b.axvline(pacientes.desviacion.median(), color=ROJO, ls="--", lw=1,
              label=f"mediana {pacientes.desviacion.median():.3f}")
    b.set(title="Desviacion de la probabilidad entre los cortes de cada paciente", xlabel="desviacion",
          ylabel="pacientes")
    b.legend(fontsize=8)
    figura.tight_layout()
    return figura
