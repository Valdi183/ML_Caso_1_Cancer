#!/usr/bin/env python3
"""Evaluacion final en TEST, una sola vez (guia del caso, C5).

    python -m src.utils.evaluar_test

Es el UNICO sitio del repositorio donde se carga test (`uc.conjunto_test`). Todo esta
decidido antes en validacion y fijado en results/test/plan.json (pesos con su SHA-256,
umbral y agregacion); el script comprueba que los pesos no han cambiado y **se niega a
ejecutarse si ya existe results/test/resumen.json**. Despues no se cambia nada.

  - Modelo: conjunto de los 5 modelos del congelado; probabilidad de un corte = media
    de las 5; probabilidad de la paciente = media de sus cortes.
  - Umbral: el de plan.json (0,3635; pCR si probabilidad > umbral).

Escribe results/test/resumen.json, results/test/predicciones_test.csv (por corte) y
predicciones_test_pacientes.csv, y las figuras docs/figuras/test_*.png.

    python -m src.utils.evaluar_test --ensayo --salida <carpeta temporal>

pasa el mismo codigo sobre la validacion del fold 0 (no carga test) y lo escribe todo
en <carpeta temporal>: sirve para que un error de codigo aparezca antes de la unica
ejecucion sobre test. Sus numeros no significan nada (4 de los 5 modelos vieron esas
pacientes al entrenar).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.data.carga import cargar_en_memoria, lotes
from src.models.cnn import cargar
from src.utils import figuras
from src.utils.analisis_cv import COHORTES, auc
from src.utils.analisis_cv import con_ic as auc_con_ic
from src.utils.comparar import brier
from src.utils.evaluacion import auc_mismo_hospital
from src.utils.umbral import con_ic, fiabilidad, metricas, plano

PLAN = config.RESULTADOS / "test" / "plan.json"
FRIDMAN = {"auc": 0.72, "sensibilidad": 0.27, "especificidad": 0.95}
REMUESTREOS, SEMILLA = 2000, config.DEFECTO["semilla"]


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def comprobar_plan(plan: dict) -> list[Path]:
    """Los pesos del plan, comprobando que son exactamente los que se fijaron."""
    rutas = []
    for p in plan["pesos"]:
        ruta = config.REPO / p["fichero"]
        if not ruta.exists():
            sys.exit(f"Falta {p['fichero']}")
        if sha256(ruta) != p["sha256"]:
            sys.exit(f"{p['fichero']} no coincide con el SHA-256 de plan.json: no se evalua")
        rutas.append(ruta)
    return rutas


@torch.no_grad()
def predecir(redes: list, imagenes: torch.Tensor, dispositivo) -> np.ndarray:
    """(N cortes, 5 modelos) con la probabilidad de cada modelo, en el orden de las filas
    (lotes sin desordenar = shuffle=False). Misma entrada que al entrenar: uint8 / 255."""
    columnas = []
    for red in redes:
        red.eval()
        columnas.append(torch.cat([torch.sigmoid(red(imagenes[i].to(dispositivo).float() / 255)).cpu()
                                   for i in lotes(len(imagenes), 128, desordenar=False)]).numpy())
    return np.stack(columnas, axis=1)


def auc_ic_por_paciente(prob, pcr, grupos) -> dict:
    """AUC con IC 95 % remuestreando pacientes (para cortes: todos los de cada paciente)."""
    prob, pcr, grupos = np.asarray(prob, dtype=float), np.asarray(pcr), np.asarray(grupos)
    ids, posicion = np.unique(grupos, return_inverse=True)
    filas_de = [np.flatnonzero(posicion == i) for i in range(len(ids))]
    rng = np.random.default_rng(SEMILLA)
    muestras = []
    for _ in range(REMUESTREOS):
        i = np.concatenate([filas_de[j] for j in rng.integers(0, len(ids), len(ids))])
        muestras.append(auc(prob[i], pcr[i]))
    muestras = np.array(muestras)
    muestras = muestras[~np.isnan(muestras)]
    return {"auc": auc(prob, pcr), "ic95": [float(np.percentile(muestras, q)) for q in (2.5, 97.5)]}


def evaluar(filas: pd.DataFrame, plan: dict, rutas: list[Path], tasa_train: float, pacientes: pd.DataFrame,
            destino: Path, carpeta_figuras: Path, uc) -> dict:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    redes = [cargar(r, dispositivo)[0] for r in rutas]
    umbral = float(plan["umbral"])

    imagenes = cargar_en_memoria(filas, "evaluacion")
    por_modelo = predecir(redes, imagenes, dispositivo)
    cortes = filas[["sample_id", "patient_id", "pCR"]].reset_index(drop=True).copy()
    for j, r in enumerate(rutas):
        cortes[f"prob_{r.stem}"] = por_modelo[:, j]
    cortes["prob"] = por_modelo.mean(axis=1)
    cortes["cohorte"] = cortes.patient_id.map(pacientes.dataset)

    tabla = cortes.groupby("patient_id").agg(prob=("prob", "mean"), pcr=("pCR", "first"),
                                             cohorte=("cohorte", "first"), cortes=("prob", "size"))
    tabla["prediccion"] = (tabla.prob > umbral).astype(int)
    cortes["prediccion"] = (cortes.prob > umbral).astype(int)
    ic = dict(remuestreos=REMUESTREOS, semilla=SEMILLA)

    def auc_ic(d, columna="prob"):
        return auc_con_ic(d, lambda x: auc(x[columna], x.pcr), **ic)

    por_cohorte = {}
    for c in COHORTES:
        d = tabla[tabla.cohorte == c]
        m = metricas(d.prob, d.pcr, umbral)
        por_cohorte[c] = {"pacientes": int(len(d)), "pcr": int(d.pcr.sum()), **auc_ic(d),
                          "sensibilidad": m["sensibilidad"], "especificidad": m["especificidad"],
                          "VP": m["VP"], "FP": m["FP"], "FN": m["FN"], "VN": m["VN"]}

    modelos = {}
    for r in rutas:
        pm = cortes.groupby("patient_id")[f"prob_{r.stem}"].mean()
        modelos[r.stem] = auc(pm.loc[tabla.index], tabla.pcr)
    aucs = np.array(list(modelos.values()))

    resumen = {
        "plan": plan,
        "dispositivo": str(dispositivo),
        "pacientes": {"n": int(len(tabla)), "pcr": int(tabla.pcr.sum()), "cortes": int(len(cortes))},
        "umbral": umbral,
        "por_paciente": {"auc": auc_ic(tabla), **con_ic(tabla.prob, tabla.pcr, tabla.index, umbral, **ic)},
        "siempre_no_pcr": con_ic(tabla.prob, tabla.pcr, tabla.index, None, **ic),
        "por_cohorte": por_cohorte,
        "mismo_hospital": auc_con_ic(tabla, lambda x: auc_mismo_hospital(x.prob, x.pcr, x.cohorte), **ic),
        "brier": brier(tabla.prob, tabla.pcr),
        "brier_constante": brier(np.full(len(tabla), tasa_train), tabla.pcr),
        "tasa_pcr_train": tasa_train,
        "fiabilidad": fiabilidad(tabla.prob, tabla.pcr).to_dict(orient="records"),
        "corte_suelto": {"auc": auc_ic_por_paciente(cortes.prob, cortes.pCR, cortes.patient_id),
                         **con_ic(cortes.prob, cortes.pCR, cortes.patient_id, umbral, **ic)},
        "modelos_por_separado": {"auc_por_paciente": modelos, "media": float(aucs.mean()),
                                 "desviacion": float(aucs.std(ddof=1)), "min": float(aucs.min()),
                                 "max": float(aucs.max())},
        "referencia_fridman_2025": FRIDMAN,
    }

    destino.mkdir(parents=True, exist_ok=True)
    cortes.to_csv(destino / "predicciones_test.csv", index=False)
    tabla.reset_index().to_csv(destino / "predicciones_test_pacientes.csv", index=False)
    rutas_figuras = dibujar(tabla, cortes, resumen, umbral, carpeta_figuras, uc, filas)
    resumen["figuras"] = [str(p.relative_to(config.REPO)) if p.is_relative_to(config.REPO) else str(p)
                          for p in rutas_figuras]
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float, ensure_ascii=False),
                                          encoding="utf-8")
    return resumen


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def dibujar(tabla, cortes, r, umbral, carpeta: Path, uc, filas) -> list[Path]:
    from sklearn.metrics import roc_curve
    plt = figuras.plt
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []

    def guardar(figura, nombre):
        ruta = carpeta / f"{nombre}.png"
        figura.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(figura)
        rutas.append(ruta)

    p = r["por_paciente"]
    # ROC con el punto del umbral
    fpr, tpr, _ = roc_curve(tabla.pcr, tabla.prob)
    figura, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color=figuras.AZUL, lw=2, label=f"conjunto, AUC {p['auc']['auc']:.3f} "
                                                      f"[{p['auc']['ic95'][0]:.3f}-{p['auc']['ic95'][1]:.3f}]")
    ax.plot([0, 1], [0, 1], color=figuras.GRIS, ls=":", lw=1, label="azar")
    ax.plot(1 - p["especificidad"], p["sensibilidad"], "o", color=figuras.ROJO, ms=9,
            label=f"umbral {umbral:.4f}: sens. {p['sensibilidad']:.2f}, esp. {p['especificidad']:.2f}")
    ax.plot(1 - FRIDMAN["especificidad"], FRIDMAN["sensibilidad"], "s", color=figuras.NARANJA, ms=8,
            label="Fridman et al. 2025 (AUC 0,72)")
    ax.set(title=f"ROC en test, por paciente ({r['pacientes']['n']} pac.)", xlabel="1 - especificidad",
           ylabel="sensibilidad", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")
    guardar(figura, "test_roc")

    # Calibracion
    f = pd.DataFrame(r["fiabilidad"])
    figura, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(f.prob_media, f.observada, "o-", color=figuras.AZUL,
            label=f"conjunto (Brier {r['brier']:.4f}; constante {r['brier_constante']:.4f})")
    ax.plot([0, 1], [0, 1], color=figuras.GRIS, lw=1, label="calibracion perfecta")
    ax.axvline(umbral, color=figuras.ROJO, ls=":", lw=1, label="umbral")
    ax.set(title="Calibracion en test (10 grupos por cuantiles)", xlabel="probabilidad media predicha",
           ylabel="fraccion observada de pCR", xlim=(0, 0.8), ylim=(0, 0.8))
    ax.legend(fontsize=8, loc="upper left")
    guardar(figura, "test_calibracion")

    # Matriz de confusion
    figura, ax = plt.subplots(figsize=(4.8, 4.2))
    matriz = np.array([[p["VN"], p["FP"]], [p["FN"], p["VP"]]])
    ax.imshow(matriz, cmap="Blues")
    for (i, j), v in np.ndenumerate(matriz):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > matriz.max() / 2 else "black")
    ax.set_xticks([0, 1], ["predice no pCR", "predice pCR"])
    ax.set_yticks([0, 1], ["no pCR (real)", "pCR (real)"])
    ax.set_title(f"Test por paciente, umbral {umbral:.4f}")
    guardar(figura, "test_matriz_confusion")

    # Casos fallados: 3 FP con mayor probabilidad y 3 FN con menor probabilidad
    fp = tabla[(tabla.prediccion == 1) & (tabla.pcr == 0)].sort_values("prob", ascending=False).head(3)
    fn = tabla[(tabla.prediccion == 0) & (tabla.pcr == 1)].sort_values("prob").head(3)
    casos = [("falso positivo", pid, fila) for pid, fila in fp.iterrows()] + \
            [("falso negativo", pid, fila) for pid, fila in fn.iterrows()]
    if casos:
        figura, ejes = plt.subplots(len(casos), 4, figsize=(12, 3.1 * len(casos)), squeeze=False)
        for fila_ejes, (tipo, pid, pac) in zip(ejes, casos):
            # El corte de la paciente con la probabilidad mas extrema en la direccion del error
            suyos = cortes[cortes.patient_id == pid]
            corte = suyos.loc[suyos.prob.idxmax() if tipo == "falso positivo" else suyos.prob.idxmin()]
            x = uc.cargar_imagen(filas.set_index("sample_id").loc[corte.sample_id])
            paneles = (("PRE", x[0], "gray", (0, 1)), ("EARLY", x[1], "gray", (0, 1)),
                       ("LATE", x[2], "gray", (0, 1)), ("EARLY-PRE", x[1] - x[0], "gray", (-0.1, 0.4)))
            for ax, (nombre, img, cmap, (lo, hi)) in zip(fila_ejes, paneles):
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(nombre, fontsize=9)
            fila_ejes[0].set_ylabel(f"{tipo}\n{pid} ({pac.cohorte})\npaciente {pac.prob:.3f}, "
                                    f"corte {corte.prob:.3f}", fontsize=9)
        figura.suptitle(f"Casos fallados en test (umbral {umbral:.4f}): 3 FP de mayor y 3 FN de menor "
                        "probabilidad", fontsize=11)
        figura.tight_layout()
        guardar(figura, "test_casos_fallados")
    return rutas


# --------------------------------------------------------------------------- #
# Consola
# --------------------------------------------------------------------------- #

def imprimir(r: dict) -> None:
    def con(m, k):
        v, (lo, hi) = m[k], m["ic95"][k]
        return f"{v:.3f} [{lo:.3f}-{hi:.3f}]" if not np.isnan(v) else "—"

    p, s, c = r["por_paciente"], r["siempre_no_pcr"], r["corte_suelto"]
    n = r["pacientes"]
    print(f"\n=== TEST (una vez): {n['n']} pacientes ({n['pcr']} pCR), {n['cortes']} cortes; "
          f"conjunto de 5 modelos; umbral {r['umbral']:.4f}")
    print(f"{'':<20}{'por paciente':>24}{'siempre no pCR':>24}{'corte suelto':>24}{'Fridman 2025':>14}")
    auc_p, auc_c = p["auc"], c["auc"]

    def texto_auc(m):
        return f"{m['auc']:.3f} [{m['ic95'][0]:.3f}-{m['ic95'][1]:.3f}]"

    print(f"{'AUC':<20}{texto_auc(auc_p):>24}{'0.500':>24}{texto_auc(auc_c):>24}{FRIDMAN['auc']:>14.2f}")
    matrices = ["/".join(str(m[k]) for k in ("VP", "FP", "FN", "VN")) for m in (p, s, c)]
    print(f"{'VP/FP/FN/VN':<20}" + "".join(f"{v:>24}" for v in matrices))
    for k, titulo in (("sensibilidad", "sensibilidad"), ("especificidad", "especificidad"), ("VPP", "VPP"),
                      ("VPN", "VPN"), ("pcr1_precision", "precision pCR=1"), ("pcr1_recall", "recall pCR=1"),
                      ("pcr1_f1", "F1 pCR=1"), ("pcr0_precision", "precision pCR=0"),
                      ("pcr0_recall", "recall pCR=0"), ("pcr0_f1", "F1 pCR=0"), ("f1_macro", "F1 macro"),
                      ("exactitud", "exactitud")):
        valores = [con({**plano(m), "ic95": m["ic95"]}, k) for m in (p, s, c)]
        ref = f"{FRIDMAN[k]:.2f}" if k in FRIDMAN else ""
        print(f"{titulo:<20}" + "".join(f"{v:>24}" for v in valores) + f"{ref:>14}")
    print("por cohorte (paciente):")
    for nombre, m in r["por_cohorte"].items():
        print(f"  {nombre:<5} {m['pacientes']:>4} pac. ({m['pcr']:>3} pCR)  AUC {m['auc']:.3f} "
              f"[{m['ic95'][0]:.3f}-{m['ic95'][1]:.3f}]  sens. {m['sensibilidad']:.3f}  "
              f"esp. {m['especificidad']:.3f}")
    mh = r["mismo_hospital"]
    print(f"AUC entre pares del mismo hospital: {mh['auc']:.3f} [{mh['ic95'][0]:.3f}-{mh['ic95'][1]:.3f}]")
    print(f"Brier por paciente: {r['brier']:.4f} (constante con la tasa de train {r['tasa_pcr_train']:.3f}: "
          f"{r['brier_constante']:.4f})")
    mm = r["modelos_por_separado"]
    print("AUC de cada modelo por separado (descriptivo): " + ", ".join(
        f"{k} {v:.3f}" for k, v in mm["auc_por_paciente"].items())
        + f"  -> {mm['media']:.3f} ± {mm['desviacion']:.3f} (conjunto {auc_p['auc']:.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ensayo", action="store_true",
                   help="probar el codigo sobre la validacion del fold 0, sin cargar test")
    p.add_argument("--salida", type=Path,
                   help="con --ensayo: carpeta donde escribirlo todo (fuera de results/)")
    a = p.parse_args()

    destino = config.RESULTADOS / "test"
    if not a.ensayo and (destino / "resumen.json").exists():
        sys.exit("results/test/resumen.json ya existe: test ya se evaluo y no se vuelve a evaluar.")
    if a.ensayo and (a.salida is None or a.salida.resolve().is_relative_to(config.RESULTADOS.resolve())
                     or a.salida.resolve().is_relative_to(config.FIGURAS.resolve())):
        sys.exit("--ensayo necesita --salida fuera de results/ y de docs/figuras/")
    if not PLAN.exists():
        sys.exit("Falta results/test/plan.json: el plan se fija antes de evaluar.")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rutas = comprobar_plan(plan)

    uc = config.utils()
    samples = uc.cargar_samples()
    pacientes = uc.cargar_patients().set_index("pid")
    entrenamiento = samples[samples.split == "train"]
    tasa_train = float(entrenamiento.groupby("patient_id").pCR.first().mean())

    if a.ensayo:
        _, filas = uc.particion(samples, fold_val=0)
        destino, carpeta_figuras = a.salida, a.salida
        print("ENSAYO sobre la validacion del fold 0 (no es test; los numeros no significan nada)")
    else:
        filas = uc.conjunto_test(samples)
        carpeta_figuras = config.FIGURAS
    resumen = evaluar(filas, plan, rutas, tasa_train, pacientes, destino, carpeta_figuras, uc)
    imprimir(resumen)
    print(f"\nGuardado en {destino}: resumen.json, predicciones_test.csv, predicciones_test_pacientes.csv")
    print("Figuras: " + ", ".join(resumen["figuras"]))


if __name__ == "__main__":
    main()
