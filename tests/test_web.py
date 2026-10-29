"""La web estatica (web/): el ONNX da lo mismo que PyTorch y su JSON es coherente con results/.

onnxruntime en CPU frente a PyTorch FP32 (`src.inferencia.pipeline`): diferencia maxima < 1e-5 y la
misma clase con el umbral, en los 4 ejemplos y, con el dataset, en 51 cortes de TRAIN de las 3
cohortes (nunca de test). La lectura del PNG en el navegador se prueba con navegadores reales
(`python -m src.web.probar_navegador`), no aqui."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from src import config, inferencia
from src.data.imagen import FASES, leer_corte

ort = pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ONNX = WEB / "modelo" / "conjunto.onnx"
TOLERANCIA = 1e-5


@pytest.fixture(scope="module")
def conjunto():
    return inferencia.cargar_conjunto()


@pytest.fixture(scope="module")
def sesion():
    return ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def meta():
    return json.loads((WEB / "modelo.json").read_text(encoding="utf-8"))


def con_onnx(sesion, x: np.ndarray) -> tuple[float, np.ndarray]:
    """x: (3, 256, 256) uint8 -> (media, por modelo) como lo hace el navegador (uint8 / 255 en float32)."""
    prob, por_modelo = sesion.run(None, {"imagen": (x[None].astype(np.float32) / 255)})
    return float(prob[0]), por_modelo[0]


def con_pytorch(conjunto, rutas) -> tuple[float, np.ndarray]:
    r = inferencia.pipeline(conjunto, rutas, "cpu")
    return r["prob"], np.array([r["por_modelo"][n] for n in conjunto.nombres])


def comprobar(sesion, conjunto, rutas, umbral):
    prob_o, por_o = con_onnx(sesion, leer_corte(rutas))
    prob_t, por_t = con_pytorch(conjunto, rutas)
    assert abs(prob_o - prob_t) < TOLERANCIA
    assert np.abs(por_o - por_t).max() < TOLERANCIA
    assert (prob_o > umbral) == (prob_t > umbral)


# --------------------------------------------------------------------------- #
# Coherencia del JSON con results/
# --------------------------------------------------------------------------- #

def test_el_json_dice_lo_que_dicen_el_plan_y_el_umbral(meta, conjunto):
    plan = json.loads((REPO / "results/test/plan.json").read_text(encoding="utf-8"))["pesos"]
    assert [p["sha256"] for p in meta["modelo"]["pesos"]] == [p["sha256"] for p in plan]
    assert [p["fichero"] for p in meta["modelo"]["pesos"]] == [p["fichero"] for p in plan]
    assert [p["sha256"] for p in meta["modelo"]["pesos"]] == list(conjunto.sha256.values())
    assert meta["umbral"] == inferencia.leer_umbral() == pytest.approx(0.3635, abs=5e-5)


def test_el_onnx_es_el_del_json(meta):
    datos = ONNX.read_bytes()
    assert hashlib.sha256(datos).hexdigest() == meta["modelo"]["sha256"]
    assert len(datos) == meta["modelo"]["bytes"]
    assert meta["modelo"]["opset"] == 17


def test_las_metricas_del_json_son_las_de_results(meta):
    assert meta["metricas"] == inferencia.metricas_internas()


def test_entrada_y_salidas_del_onnx(sesion):
    entrada, = sesion.get_inputs()
    assert (entrada.name, entrada.shape, entrada.type) == ("imagen", [1, 3, 256, 256], "tensor(float)")
    assert [(s.name, s.shape) for s in sesion.get_outputs()] == [("prob", [1]), ("por_modelo", [1, 5])]


def test_los_ejemplos_de_la_web_son_los_de_la_app(meta):
    for ej in meta["ejemplos"]:
        for fase in FASES:
            copia = (WEB / ej["ficheros"][fase]).read_bytes()
            assert copia == (REPO / "app" / "ejemplos" / f"{ej['corte']}_{fase}.png").read_bytes()
    assert (WEB / "ejemplos" / "LEEME.md").read_text(encoding="utf-8").count("CC BY-NC 4.0") >= 1


# --------------------------------------------------------------------------- #
# Paridad ONNX frente a PyTorch
# --------------------------------------------------------------------------- #

def test_paridad_en_los_ejemplos(sesion, conjunto, meta):
    for ej in meta["ejemplos"]:
        rutas = [WEB / ej["ficheros"][f] for f in FASES]
        comprobar(sesion, conjunto, rutas, meta["umbral"])
        prob, por_modelo = con_onnx(sesion, leer_corte(rutas))
        # y con la referencia guardada en el JSON (la que usa la autocomprobacion del navegador)
        assert abs(prob - ej["referencia"]["prob"]) < TOLERANCIA
        assert np.abs(por_modelo - np.array(ej["referencia"]["por_modelo"])).max() < TOLERANCIA
        assert ("pCR" if prob > meta["umbral"] else "no pCR") == ej["referencia"]["clase"]


@pytest.mark.skipif(config.RAIZ is None, reason="sin el dataset (CI)")
def test_paridad_en_cortes_de_train_de_las_3_cohortes(sesion, conjunto, meta):
    from src.data.carga import COHORTES, rutas_corte
    uc = config.utils()
    samples = uc.cargar_samples()
    cohorte = samples.patient_id.map(uc.cargar_patients().set_index("pid").dataset)
    train = samples[samples.split == "train"]
    elegidos = []
    for c in COHORTES:
        de_la_cohorte = train[cohorte[train.index] == c]
        elegidos.append(de_la_cohorte.sample(17, random_state=7))
    assert sum(len(e) for e in elegidos) == 51
    for fila in itertools.chain.from_iterable(e.itertuples() for e in elegidos):
        comprobar(sesion, conjunto, rutas_corte(fila, config.RAIZ), meta["umbral"])


def test_exportar_de_nuevo_da_un_onnx_equivalente(conjunto, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from src.web import exportar
    ruta = tmp_path / "otro.onnx"
    exportar.exportar_onnx(conjunto, ruta)
    nuevo = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    for base in ("ISPY2-329941_z008", "Breast_MRI_290_z010"):
        x = leer_corte([WEB / "ejemplos" / f"{base}_{f}.png" for f in FASES])
        assert abs(con_onnx(nuevo, x)[0] - con_pytorch(
            conjunto, [WEB / "ejemplos" / f"{base}_{f}.png" for f in FASES])[0]) < TOLERANCIA


# --------------------------------------------------------------------------- #
# Las cifras de las dos paginas (web/datos.json) y los ficheros de prueba del navegador
# --------------------------------------------------------------------------- #

def iguales(a, b, ruta="") -> None:
    """Dos JSON iguales; los decimales con una tolerancia de 1e-9 (no dependen de la version de NumPy)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), ruta
        for k in a:
            iguales(a[k], b[k], f"{ruta}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), ruta
        for i, (x, y) in enumerate(zip(a, b)):
            iguales(x, y, f"{ruta}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), ruta
    else:
        assert a == b, ruta


@pytest.fixture(scope="module")
def datos():
    return json.loads((WEB / "datos.json").read_text(encoding="utf-8"))


def test_datos_json_esta_al_dia(datos, conjunto):
    """web/datos.json es lo que sale hoy de results/ y de la red: si cambia algo, `python -m
    src.web.exportar --solo-datos` lo regenera."""
    pytest.importorskip("sklearn")
    from src.web import datos as datos_web
    ejemplos = [{"corte": e["corte"], "cohorte": e["cohorte"]} for e in json.loads(
        (WEB / "modelo.json").read_text(encoding="utf-8"))["ejemplos"]]
    iguales(json.loads(json.dumps(datos_web.construir(conjunto, ejemplos))), datos)


def test_las_cifras_clave_son_las_de_results(datos):
    test = json.loads((REPO / "results/test/resumen.json").read_text(encoding="utf-8"))
    umbral = json.loads((REPO / "results/umbral/resumen.json").read_text(encoding="utf-8"))["normal"]
    assert datos["test"]["auc"]["auc"] == test["por_paciente"]["auc"]["auc"]
    assert datos["test"]["matriz"] == {k: test["por_paciente"][k] for k in ("VP", "FP", "FN", "VN")}
    assert datos["umbral"] == umbral["umbral_final"] == datos["seleccion"]["umbral"]["valor"]
    # la curva del umbral reproduce la matriz fuera de fold de results/umbral
    assert datos["seleccion"]["umbral"]["en_el_umbral"]["especificidad"] == pytest.approx(
        umbral["por_paciente"]["especificidad"], abs=1e-9)
    assert datos["datos"]["pacientes"] == 1273 and datos["datos"]["cortes"] == 12703
    assert sum(c["pacientes"] for c in datos["datos"]["cohortes"]) == datos["datos"]["pacientes"]
    assert datos["modelo"]["parametros"] == 18329
    assert datos["modelo"]["campo_receptivo"]["px"] == 72        # calculado de la red, no escrito a mano


def test_los_data_v_del_html_existen_en_datos_json(datos):
    """Cada `data-v="ruta"` de las dos paginas tiene dato: una errata saldria como «?» en pantalla."""
    import re
    for pagina in ("index.html", "predecir.html"):
        html = (WEB / pagina).read_text(encoding="utf-8")
        rutas = set(re.findall(r'data-v="([^"]+)"', html))
        assert rutas, pagina
        for ruta in rutas:
            valor = datos
            for clave in ruta.split("."):
                valor = valor[int(clave)] if isinstance(valor, list) else valor[clave]
            assert valor is not None, f"{pagina}: {ruta}"


def test_los_ficheros_de_prueba_del_navegador_estan_al_dia():
    """web/pruebas/*.json son lo que dice Python hoy (los contrasta `python -m src.web.probar_navegador`)."""
    from src.web import casos_lectura, casos_visualizacion
    iguales(json.loads(json.dumps(casos_lectura.generar())),
            json.loads((WEB / "pruebas" / "casos.json").read_text(encoding="utf-8")))
    esperado = {"aviso": "Generado por src/web/casos_visualizacion.py; no se edita a mano.",
                "casos": casos_visualizacion.casos()}
    iguales(json.loads(json.dumps(esperado)),
            json.loads((WEB / "pruebas" / "visualizacion.json").read_text(encoding="utf-8")))
    for caso in json.loads((WEB / "pruebas" / "casos.json").read_text(encoding="utf-8"))["casos"]:
        if "fichero" in caso:
            assert (WEB / "pruebas" / caso["fichero"]).exists(), caso["nombre"]
