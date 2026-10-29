"""Los JSON de los controles de la web (src/web/exportar.py) salen de results/ y no traen test ni
identificadores: web/historial.json (control de epoca) y web/umbral_validacion.json (explorador)."""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
EJECUCIONES = ["it6_3bloques_f0", "it6_3bloques_f1", "it8_3bloques_f2", "it8_3bloques_f3", "it8_3bloques_f4"]


def leer(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


def test_historial_es_el_de_results():
    folds = leer(WEB / "historial.json")["folds"]
    assert [f["fold"] for f in folds] == [0, 1, 2, 3, 4]
    for nombre, f in zip(EJECUCIONES, folds):
        h = pd.read_csv(REPO / "results" / nombre / "historial.csv")
        assert f["epoca"] == h.epoca.tolist()
        assert f["epoca_elegida"] == leer(REPO / "results" / nombre / "metricas.json")["mejor_epoca"]
        for k in ("auc_train", "auc_val", "perdida_train", "perdida_val"):
            assert max(abs(a - b) for a, b in zip(f[k], h[k])) <= 5e-5          # redondeado a 4 decimales


def test_explorador_reproduce_la_matriz_oficial_de_validacion():
    u = leer(WEB / "umbral_validacion.json")
    oficial = leer(REPO / "results" / "umbral" / "resumen.json")["normal"]
    assert u["umbral_oficial"] == oficial["umbral_final"]
    assert u["columnas"] == ["prob", "pcr", "cohorte"] and len(u["pacientes"]) == oficial["pacientes"]["n"]
    t = u["umbral_oficial"]
    matriz = {"VP": 0, "FP": 0, "FN": 0, "VN": 0}
    for p, y, _ in u["pacientes"]:
        matriz["VP" if p > t and y else "FP" if p > t else "FN" if y else "VN"] += 1
    assert matriz == {k: oficial["por_paciente"][k] for k in matriz}


def test_explorador_sin_identificadores_ni_test():
    texto = (WEB / "umbral_validacion.json").read_text(encoding="utf-8")
    for prefijo in ("ISPY", "ACRIN", "Breast_MRI"):
        assert prefijo not in texto
    u = leer(WEB / "umbral_validacion.json")
    assert {c for _, _, c in u["pacientes"]} == {"spy1", "spy2", "duke"}
    probs = [p for p, _, _ in u["pacientes"]]
    assert probs == sorted(probs)                                   # ordenado: sin el orden de origen
    test = pd.read_csv(REPO / "results" / "test" / "predicciones_test_pacientes.csv")
    assert not set(round(v, 12) for v in test.prob) & set(round(p, 12) for p, _, _ in u["pacientes"])
