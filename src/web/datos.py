"""Todas las cifras de la web (index.html y predecir.html) en un unico JSON: web/datos.json.

    python -m src.web.exportar --solo-datos      # o `python -m src.web.datos`

Cada cifra sale de results/ (los JSON y CSV que generan los scripts del proyecto), de la red
(arquitectura, campo receptivo) o de docs/data/referencias.json (lo que no sale de results/:
la referencia del profesor, Fridman et al., la licencia). La web no escribe ninguna cifra a mano.
No se pasa ninguna imagen de test por ninguna red: de test solo se leen las predicciones ya
guardadas en results/test/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import inferencia
from src import visualizacion as vis
from src.data.imagen import FASES, leer_corte

REPO = Path(__file__).resolve().parents[2]
RESULTADOS = REPO / "results"
WEB = REPO / "web"
REFERENCIAS = REPO / "docs" / "data" / "referencias.json"
EJEMPLOS = REPO / "app" / "ejemplos"
NOMBRES_COHORTE = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}
EJECUCIONES = ["it6_3bloques_f0", "it6_3bloques_f1", "it8_3bloques_f2", "it8_3bloques_f3", "it8_3bloques_f4"]
PONDERADAS = [f"it8_3bloques_pond_f{i}" for i in range(5)]


def leer(*partes) -> dict:
    return json.loads(RESULTADOS.joinpath(*partes).read_text(encoding="utf-8"))


def limpio(x):
    """JSON valido: NaN e infinitos pasan a null y numpy a tipos de Python."""
    if isinstance(x, dict):
        return {str(k): limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpio(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --------------------------------------------------------------------------- #
# Datos del caso
# --------------------------------------------------------------------------- #

def datos_del_caso(umbral: dict, test: dict) -> dict:
    """Tamaños del dataset, de results/ (sin abrir el dataset): pacientes y cortes de train salen
    del umbral fuera de fold y los de test de la evaluacion de test."""
    u = umbral["normal"]
    train = {"pacientes": u["pacientes"]["n"], "pcr": u["pacientes"]["pcr"],
             "cortes": u["cortes_sueltos"]["n"]}
    prueba = {"pacientes": test["pacientes"]["n"], "pcr": test["pacientes"]["pcr"],
              "cortes": test["pacientes"]["cortes"]}
    cohortes = []
    for c, nombre in NOMBRES_COHORTE.items():
        t, p = u["por_cohorte"][c]["pacientes"], test["por_cohorte"][c]["pacientes"]
        cohortes.append({"id": c, "nombre": nombre, "pacientes": t + p, "train": t, "test": p,
                         "test_fraccion": p / test["pacientes"]["n"]})
    total = train["pacientes"] + prueba["pacientes"]
    pcr = train["pcr"] + prueba["pcr"]
    assert sum(c["pacientes"] for c in cohortes) == total
    return {"pacientes": total, "cortes": train["cortes"] + prueba["cortes"], "pcr": pcr,
            "tasa_pcr": pcr / total, "train": train, "test": prueba, "cohortes": cohortes,
            "fases": list(FASES), "folds": 5, "lado": 256}


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

def campo_receptivo(red) -> dict:
    """Campo receptivo en pixeles del corte original (256 px) de cada etapa, recorriendo la red:
    r += (k - 1) * salto y salto *= paso, para la media de bloques (k = paso = reducir), las
    convoluciones y los max pool. Es el de un pixel del mapa final."""
    r, salto, pasos = 1, 1, []

    def avanzar(k, paso):
        nonlocal r, salto
        r += (k - 1) * salto
        salto *= paso

    avanzar(red.config["reducir"], red.config["reducir"])
    pasos.append(("reducir", r))
    for i, bloque in enumerate(red.bloques):
        for m in bloque:
            if m.__class__.__name__ == "Conv2d":
                avanzar(m.kernel_size[0], m.stride[0])
            elif m.__class__.__name__ == "MaxPool2d":
                avanzar(m.kernel_size, m.stride)
        pasos.append((f"bloque {i + 1}", r))
    lado = 256
    return {"px": r, "fraccion_area": (r / lado) ** 2, "por_etapa": [{"etapa": n, "px": v} for n, v in pasos],
            "salto_final": salto}


def descripcion_modelo(conjunto: inferencia.Conjunto) -> dict:
    from src.utils.arquitectura import etapas
    red = conjunto.redes[0]
    entrenamiento = dict(conjunto.info[conjunto.nombres[0]]["entrenamiento"])
    entrenamiento.pop("fold")
    return {"etapas": etapas(red), "parametros": conjunto.parametros, "config": conjunto.config_red,
            "n_modelos": len(conjunto.redes), "n_bloques": len(conjunto.config_red["filtros"]),
            "resolucion_interna": 256 // conjunto.config_red["reducir"],
            "campo_receptivo": campo_receptivo(red), "entrenamiento": entrenamiento,
            "pesos": [{"nombre": n, "fold": conjunto.info[n]["entrenamiento"]["fold"],
                       "epoca": conjunto.info[n]["epoca"], "sha256": conjunto.sha256[n]}
                      for n in conjunto.nombres]}


# --------------------------------------------------------------------------- #
# Como se eligio
# --------------------------------------------------------------------------- #

def etiqueta_config(fila) -> str:
    """Una configuracion en pocas palabras (de las columnas del inventario)."""
    extra = []
    c = fila.configuracion
    if fila.pooling != "media":
        extra.append({"atencion": "atención", "mediamax": "media + máximo", "max": "máximo"}[fila.pooling])
    if fila.peso_caja:
        extra.append("atención con caja")
    if fila.ponderada:
        extra.append("ponderada")
    if "realce" in c:
        extra.append("canales de realce")
    if "normalizar" in c:
        extra.append("normalizar corte")
    if (m := re.search(r"lr ([\d.e-]+)", c)):
        extra.append(f"lr {m[1]}")
    if "wd 0.001" not in c:
        extra.append("sin regularizar")
    return f"{fila.filtros} · {256 // fila.reducir} px" + "".join(f" · {e}" for e in extra)


def configuraciones(congelado: str) -> list[dict]:
    """Las configuraciones probadas para pCR, de results/comparacion/inventario.csv. Las que se
    ejecutaron en tandas distintas (it6 + it8, it7 + it9) se juntan por su configuracion: cada
    una con los folds en los que se probo (de 1 a 5)."""
    inv = pd.read_csv(RESULTADOS / "comparacion" / "inventario.csv")
    inv = inv[inv.objetivo == "pcr"]
    filas = []
    for cfg, g in inv.groupby("configuracion", sort=False):
        f = g.iloc[0]
        g = g.sort_values("fold")
        filas.append({
            "etiqueta": etiqueta_config(f), "configuracion": cfg, "bloques": len(f.filtros.split("-")),
            "parametros": int(f.parametros), "folds": [int(v) for v in g.fold],
            "auc_por_fold": [float(v) for v in g.auc_val_medio], "auc_medio": float(g.auc_val_medio.mean()),
            "iteraciones": sorted({str(v) for v in g.iteracion}),
            "congelado": cfg == congelado})
    return sorted(filas, key=lambda d: d["auc_medio"])


def resumen_seleccion(cfgs: list[dict], comparacion: dict, curvas: list[dict]) -> dict:
    """Cifras para el texto de «Como se eligio»: el rango de las configuraciones probadas en 4 o 5
    folds (las de 1 o 2 folds se mueven por ruido y no son comparables) y lo que varia el modelo
    elegido de un fold a otro."""
    amplias = [c["auc_medio"] for c in cfgs if len(c["folds"]) >= 4]
    por_fold = [f["auc_medio"] for f in comparacion["congelado"]["por_fold"]]
    return {"n_configuraciones": len(cfgs), "n_con_4_o_5_folds": len(amplias),
            "rango_4_o_5_folds": {"min": min(amplias), "max": max(amplias)},
            "rango_todas": {"min": min(c["auc_medio"] for c in cfgs),
                            "max": max(c["auc_medio"] for c in cfgs)},
            "congelado_por_fold": {"min": min(por_fold), "max": max(por_fold)},
            "auc_val_max": max(max(c["auc_val"]) for c in curvas),
            "auc_train_max": max(max(c["auc_train"]) for c in curvas)}


def control_de_cohorte() -> dict:
    """La misma red aprendiendo a reconocer la cohorte Duke (results/objduke_normal_f0): comprueba
    que el circuito funciona y que el hospital se ve en la imagen."""
    h = pd.read_csv(RESULTADOS / "objduke_normal_f0" / "historial.csv")
    return {"auc_val_max": float(h.auc_val.max()), "epocas": int(len(h))}


def curvas_de_entrenamiento(conjunto: inferencia.Conjunto) -> list[dict]:
    salida = []
    for nombre in EJECUCIONES:
        h = pd.read_csv(RESULTADOS / nombre / "historial.csv")
        salida.append({"ejecucion": nombre, "fold": conjunto.info[nombre]["entrenamiento"]["fold"],
                       "epoca_elegida": conjunto.info[nombre]["epoca"], "epoca": [int(v) for v in h.epoca],
                       "auc_train": [float(v) for v in h.auc_train],
                       "auc_val": [float(v) for v in h.auc_val]})
    return salida


def perdida_normal_y_ponderada(comparacion: dict) -> dict:
    """Pérdida normal frente a ponderada: AUC medio de validacion y sensibilidad con corte 0,5 por
    paciente en la epoca elegida de cada fold (de historial.csv)."""
    def lado(ejecuciones, clave):
        por_fold = comparacion[clave]["por_fold"]
        sens, esp = [], []
        for r, f in zip(ejecuciones, por_fold):
            h = pd.read_csv(RESULTADOS / r / "historial.csv").set_index("epoca")
            sens.append(float(h.loc[f["epoca_elegida"], "sensibilidad_05"]))
            esp.append(float(h.loc[f["epoca_elegida"], "especificidad_05"]))
        return {"auc_medio": comparacion[clave]["auc_medio"],
                "auc_elegido": comparacion[clave]["auc_elegido"],
                "auc_por_fold": [f["auc_medio"] for f in por_fold],
                "sensibilidad_05": {"media": float(np.mean(sens)), "desviacion": float(np.std(sens, ddof=1)),
                                    "por_fold": sens},
                "especificidad_05": {"media": float(np.mean(esp)), "desviacion": float(np.std(esp, ddof=1))}}
    return {"normal": lado(EJECUCIONES, "congelado"), "ponderada": lado(PONDERADAS, "congelado_ponderada"),
            "pos_weight": 2.4}


def curva_umbral(umbral: dict) -> dict:
    """Sensibilidad y especificidad por paciente fuera de fold en funcion del umbral (media de los
    cortes de cada paciente), de las predicciones de validacion de los 5 folds."""
    d = pd.concat([pd.read_csv(RESULTADOS / r / "predicciones_val.csv") for r in EJECUCIONES])
    p = d.groupby("patient_id").agg(prob=("prob", "mean"), pcr=("pCR", "first"))
    u_final = umbral["normal"]["umbral_final"]
    pos, neg = p.prob[p.pcr == 1].to_numpy(), p.prob[p.pcr == 0].to_numpy()
    rejilla = np.round(np.linspace(0.1, 0.7, 121), 4)
    puntos = [{"umbral": float(t), "sensibilidad": float((pos > t).mean()),
               "especificidad": float((neg <= t).mean())} for t in rejilla]
    en_final = {"sensibilidad": float((pos > u_final).mean()),
                "especificidad": float((neg <= u_final).mean())}
    n = umbral["normal"]["por_paciente"]
    assert abs(en_final["especificidad"] - n["especificidad"]) < 1e-9, "la curva no reproduce results/umbral"
    minima = float(re.search(r"([\d.]+) por paciente", umbral["criterio"])[1])
    return {"valor": u_final, "criterio": umbral["criterio"], "especificidad_minima": minima,
            "regla": umbral["normal"]["regla"],
            "pacientes": umbral["normal"]["pacientes"], "curva": puntos, "en_el_umbral": en_final,
            "matriz": {k: n[k] for k in ("VP", "FP", "FN", "VN")},
            "por_paciente": {k: n[k] for k in ("sensibilidad", "especificidad", "VPP", "VPN", "f1_macro")},
            "por_corte": {k: umbral["normal"]["cortes_sueltos"][k]
                          for k in ("sensibilidad", "especificidad", "VPP")},
            "siempre_no_pcr_f1_macro": umbral["normal"]["siempre_no_pcr"]["f1_macro"],
            "alternativas": leer("metricas_extra", "resumen.json")["umbrales_validacion"]["filas"]}


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #

def resultados_test(test: dict, extra: dict) -> dict:
    from sklearn.metrics import roc_curve
    p = pd.read_csv(RESULTADOS / "test" / "predicciones_test_pacientes.csv")
    fpr, tpr, umbrales = roc_curve(p.pcr, p.prob)
    u = test["umbral"]
    pp = test["por_paciente"]
    return {
        "pacientes": test["pacientes"], "umbral": u,
        "auc": pp["auc"], "matriz": {k: pp[k] for k in ("VP", "FP", "FN", "VN")},
        "por_paciente": {k: pp[k] for k in ("sensibilidad", "especificidad", "VPP", "VPN", "f1_macro",
                                            "exactitud",
                                            "pcr1", "pcr0", "ic95")},
        "siempre_no_pcr": {k: v for k, v in test["siempre_no_pcr"].items() if k != "ic95"},
        "corte_suelto": {k: test["corte_suelto"][k] for k in ("auc", "sensibilidad", "especificidad", "VPP",
                                                              "f1_macro")},
        "por_cohorte": test["por_cohorte"], "mismo_hospital": test["mismo_hospital"],
        "roc": {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr],
                "punto": {"fpr": 1 - pp["especificidad"], "tpr": pp["sensibilidad"], "umbral": u}},
        "fiabilidad": test["fiabilidad"], "brier": test["brier"], "brier_constante": test["brier_constante"],
        "tasa_pcr_train": test["tasa_pcr_train"],
        "modelos_por_separado": test["modelos_por_separado"],
        "calibracion": extra["test"], "permutacion": extra["permutacion_test"]["por_paciente"],
        "potencia": extra["potencia"],
        "ap": {"test": extra["test"]["ap"], "sin_senal": extra["test"]["ap_sin_senal"]},
        "fridman": test["referencia_fridman_2025"]}


# --------------------------------------------------------------------------- #
# Portada: el corte mas expresivo de los ejemplos de train
# --------------------------------------------------------------------------- #

def elegir_portada(ejemplos: list[dict]) -> dict:
    """Entre los ejemplos de train, el de mayor aumento medio de EARLY respecto a PRE en la zona que
    mas realza (la de la curva de captacion). Se elige por esa cifra, no a ojo."""
    mejor = None
    for ej in ejemplos:
        x = leer_corte([EJEMPLOS / f"{ej['corte']}_{f}.png" for f in FASES])
        m = vis.curva_captacion(x)["medias"]
        subida = m["EARLY"] - m["PRE"]
        if mejor is None or subida > mejor["subida"]:
            mejor = {"corte": ej["corte"], "cohorte": ej["cohorte"], "subida": subida, "medias": m}
    return mejor


# --------------------------------------------------------------------------- #
# Todo junto
# --------------------------------------------------------------------------- #

def construir(conjunto: inferencia.Conjunto, ejemplos: list[dict]) -> dict:
    umbral, test = leer("umbral", "resumen.json"), leer("test", "resumen.json")
    comparacion, extra = leer("comparacion", "resumen.json"), leer("metricas_extra", "resumen.json")
    refs = json.loads(REFERENCIAS.read_text(encoding="utf-8"))
    cfgs = configuraciones(congelado="8-16-32 · 128 px · media · normal · aum · wd 0.001 · dropout 0.7")
    assert any(c["congelado"] and len(c["folds"]) == 5 for c in cfgs), "no encuentro el modelo congelado"
    congelado = comparacion["congelado"]
    comb = vis.imagen_frente_a_clinica()
    curvas = curvas_de_entrenamiento(conjunto)
    cons = extra["consistencia_cortes_validacion"]
    dos_lados = cons["pacientes_cortes_a_los_dos_lados"] / cons["pacientes"]
    por_fold_combinado = leer("combinado", "resumen.json")["por_fold"]["clinico + cnn"]["auc_por_fold"]
    sat = leer("saturacion", "resumen.json")
    return limpio({
        "formato": 1,
        "aviso": "Generado por src/web/datos.py; no se edita a mano.",
        "datos": datos_del_caso(umbral, test),
        "modelo": descripcion_modelo(conjunto),
        "seleccion": {
            "configuraciones": cfgs,
            "curvas": curvas,
            "validacion": {"auc_medio": congelado["auc_medio"], "auc_elegido": congelado["auc_elegido"],
                           "diferencia_train_val": congelado["diferencia_train_val"],
                           "por_fold": congelado["por_fold"], "global": congelado["auc_global"],
                           "por_cohorte": congelado["por_cohorte"],
                           "perdida_constante": congelado["perdida_constante"],
                           "perdida_val_min": congelado["perdida_val_min"]},
            "resumen": resumen_seleccion(cfgs, comparacion, curvas),
            "control_cohorte": control_de_cohorte(),
            "perdida": perdida_normal_y_ponderada(comparacion),
            "umbral": curva_umbral(umbral),
            "atencion_supervisada": comparacion["atencion_supervisada"]},
        "test": resultados_test(test, extra),
        "comparacion": {"referencias": refs, "clinica_e_imagen": comb,
                        "diferencia_profesor": refs["profesor"]["auc_test"]
                        - test["por_paciente"]["auc"]["auc"],
                        "consistencia": {"icc": cons["icc"], "fraccion_pacientes_dos_lados": dos_lados},
                        "combinado": {"auc_por_fold": por_fold_combinado}},
        "saturacion": {"criterio": sat["criterio"], "todas": sat["todas"], "por_cohorte": sat["por_cohorte"]},
        "fiarse": vis.cifras_confianza(),
        "portada": elegir_portada(ejemplos),
        "creditos": {**refs["dataset"], "repositorio": refs["repositorio"]},
        "umbral": conjunto.umbral,
    })


def escribir(conjunto: inferencia.Conjunto, ejemplos: list[dict], destino: Path = WEB / "datos.json") -> dict:
    datos = construir(conjunto, ejemplos)
    destino.write_text(json.dumps(datos, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return datos


def main() -> None:
    import csv
    conjunto = inferencia.cargar_conjunto()
    with open(EJEMPLOS / "ejemplos.csv", encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    datos = escribir(conjunto, [{"corte": r["corte"], "cohorte": r["cohorte"]} for r in filas])
    print(f"web/datos.json: {(WEB / 'datos.json').stat().st_size / 1024:.0f} KB; "
          f"{len(datos['seleccion']['configuraciones'])} configuraciones; "
          f"portada {datos['portada']['corte']}")


if __name__ == "__main__":
    main()
