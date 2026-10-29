"""Contenido de la web (web/index.html, web/predecir.html y web/datos.json) que no necesita ONNX:
la referencia de la asignatura no se atribuye a nadie ni lleva intervalo, y las cifras nuevas de
«¿Por qué la media?» y de las lineas base son las de results/."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PROHIBIDO = re.compile(r"profesor|comunicaci[oó]n personal|mismas restricciones|se solapan", re.IGNORECASE)


def leer(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8"))


def test_la_web_no_atribuye_la_referencia():
    ficheros = [
        WEB / "index.html",
        WEB / "predecir.html",
        WEB / "datos.json",
        REPO / "docs" / "data" / "referencias.json",
        *sorted((WEB / "js").glob("*.js")),
    ]
    for f in ficheros:
        assert not PROHIBIDO.search(f.read_text(encoding="utf-8")), f.name


def test_la_referencia_de_la_asignatura_no_tiene_intervalo():
    ref = leer(REPO / "docs" / "data" / "referencias.json")["referencia_asignatura"]
    assert ref["auc_test"] == 0.64
    assert not any("ic" in k or "intervalo" in k for k in ref)
    datos = leer(WEB / "datos.json")
    assert datos["comparacion"]["referencias"]["referencia_asignatura"] == ref
    test = leer(REPO / "results" / "test" / "resumen.json")
    assert (
        datos["comparacion"]["diferencia_referencia"] == ref["auc_test"] - test["por_paciente"]["auc"]["auc"]
    )


def test_por_que_la_media_sale_de_results():
    datos = leer(WEB / "datos.json")["agregacion"]
    umbral = leer(REPO / "results" / "umbral" / "resumen.json")["normal"]
    assert datos["cortes_auc"] == umbral["auc_por_agregacion"]
    sobre = leer(REPO / "results" / "metricas_extra" / "resumen.json")["modelos_en_test"][
        "pacientes_sobre_umbral"
    ]
    sueltos = [v for k, v in sobre.items() if k != "conjunto"]
    assert datos["modelos_sobre_umbral_test"] == {
        "min": min(sueltos),
        "max": max(sueltos),
        "conjunto": sobre["conjunto"],
    }
    assert datos["valores_voto"] == 6  # 0 a 5 votos de los 5 modelos


def test_lineas_base_salen_de_results():
    b = leer(WEB / "datos.json")["comparacion"]["baselines"]
    test = leer(REPO / "results" / "test" / "resumen.json")["siempre_no_pcr"]
    assert b["siempre_no_pcr"]["exactitud"] == test["exactitud"] and b["siempre_no_pcr"]["auc"] == 0.5
    clin = leer(REPO / "results" / "metricas_extra" / "resumen.json")["clinico_validacion"][
        "subtipo + edad + volumen"
    ]
    assert b["clinico"]["media"] == clin["auc_folds_media"]
    assert b["red_base"]["parametros"] == 294129 and b["red_base"]["folds"] == [0]


def test_los_graficos_nuevos_estan_registrados():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    index_js = (WEB / "js" / "index.js").read_text(encoding="utf-8")
    graficos = (WEB / "js" / "graficos.js").read_text(encoding="utf-8")
    assert (
        'data-grafico="folds"' in html and "folds: G.folds" in index_js and "function folds(d, W)" in graficos
    )
    assert "¿Por qué la media?" in html
