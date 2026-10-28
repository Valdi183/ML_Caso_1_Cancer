"""La app (src/inferencia.py): misma prediccion que el pipeline y errores bien gestionados.

Usa los pesos versionados y los cortes de train de app/ejemplos, asi que corre en la CI
sin el dataset. Con el dataset, ademas compara con el pipeline de entrenamiento en
cortes de train elegidos al azar."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src import config, inferencia
from src.data.imagen import FASES

REPO = Path(__file__).resolve().parents[1]
EJEMPLOS = REPO / "app" / "ejemplos"
TOLERANCIA = 1e-6          # frente al pipeline ejecutado en FP32 (CPU, como la app)
TOLERANCIA_TF32 = 1e-3     # frente a probabilidades guardadas en GPU con TF32


@pytest.fixture(scope="module")
def conjunto():
    return inferencia.cargar_conjunto()


def png(array: np.ndarray, modo: str | None = None) -> bytes:
    b = io.BytesIO()
    Image.fromarray(array, modo).save(b, format="PNG")
    return b.getvalue()


def corte_ejemplo(base: str) -> tuple[np.ndarray, dict]:
    datos, nombres = inferencia.corte_de_ejemplo(EJEMPLOS, base)
    return np.stack([inferencia.validar_fase(datos[f], f)[0] for f in FASES]), nombres


def pipeline_antiguo(base: str) -> torch.Tensor:
    """La entrada como la construia el entrenamiento antes de extraer la lectura:
    utils_caso.cargar_fase (float32 / 255) -> np.rint(* 255) -> uint8."""
    fases = []
    for f in FASES:
        with Image.open(EJEMPLOS / f"{base}_{f}.png") as im:
            x = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        fases.append(np.rint(x * 255).astype(np.uint8))
    return torch.from_numpy(np.stack(fases)[None])


# --------------------------------------------------------------------------- #
# Modelo y umbral
# --------------------------------------------------------------------------- #

def test_umbral_es_el_de_validacion_y_no_el_de_youden_de_los_pesos(conjunto):
    oficial = json.loads((REPO / "results/umbral/resumen.json").read_text(encoding="utf-8"))["normal"]
    assert conjunto.umbral == pytest.approx(oficial["umbral_final"])
    assert conjunto.umbral == pytest.approx(0.3635, abs=5e-5)
    assert all(abs(conjunto.umbral - i["umbral"]) > 1e-3 for i in conjunto.info.values())


def test_carga_los_5_pesos_del_plan_en_cpu_y_eval(conjunto):
    plan = json.loads((REPO / "results/test/plan.json").read_text(encoding="utf-8"))
    assert [f"models/{n}.pt" for n in conjunto.nombres] == [p["fichero"] for p in plan["pesos"]]
    assert list(conjunto.sha256.values()) == [p["sha256"] for p in plan["pesos"]]
    assert str(conjunto.dispositivo) == "cpu"
    assert all(not red.training for red in conjunto.redes)


def test_sin_fichero_de_umbral_no_arranca(tmp_path):
    with pytest.raises(inferencia.ErrorModelo, match="umbral"):
        inferencia.cargar_conjunto(ruta_umbral=tmp_path / "no_existe.json")


def test_peso_cambiado_no_arranca(tmp_path):
    plan = json.loads((REPO / "results/test/plan.json").read_text(encoding="utf-8"))
    plan["pesos"][2]["sha256"] = "0" * 64
    falso = tmp_path / "plan.json"
    falso.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(inferencia.ErrorModelo, match="SHA-256"):
        inferencia.cargar_conjunto(plan=falso)


# --------------------------------------------------------------------------- #
# Misma prediccion que el pipeline
# --------------------------------------------------------------------------- #

def test_app_igual_que_el_pipeline_de_test_en_los_ejemplos(conjunto):
    """El conjunto de la app frente a `evaluar_test.predecir` (el que evaluo test), con la
    entrada construida como antes de extraer la lectura."""
    from src.utils.evaluar_test import predecir as predecir_test
    for base in pd.read_csv(EJEMPLOS / "ejemplos.csv").corte:
        x, _ = corte_ejemplo(base)
        app = inferencia.predecir(conjunto, x)
        referencia = predecir_test(conjunto.redes, pipeline_antiguo(base), "cpu")[0]
        assert np.allclose(list(app["por_modelo"].values()), referencia, atol=TOLERANCIA)
        assert app["prob"] == pytest.approx(float(referencia.mean()), abs=TOLERANCIA)
        assert app["clase"] == ("pCR" if app["prob"] > conjunto.umbral else "no pCR")


def test_app_igual_que_lo_que_guardo_el_entrenamiento(conjunto):
    """Para cada ejemplo, el modelo de su fold no vio a la paciente y guardo su probabilidad
    en predicciones_val.csv al entrenar. Ese valor se calculo en GPU con lotes de 128, donde
    cuDNN usa TF32 por defecto (torch.backends.cudnn.allow_tf32): difiere del FP32 de la app
    hasta ~5e-4 (sin TF32, la GPU coincide con la CPU en < 1e-6). Por eso aqui 1e-3."""
    tabla = pd.read_csv(EJEMPLOS / "ejemplos.csv")
    for f in tabla.itertuples():
        nombre = next(n for n in conjunto.nombres if conjunto.info[n]["entrenamiento"]["fold"] == f.fold)
        guardadas = pd.read_csv(REPO / "results" / nombre / "predicciones_val.csv").set_index("sample_id")
        x, _ = corte_ejemplo(f.corte)
        app = inferencia.predecir(conjunto, x)["por_modelo"][nombre]
        assert app == pytest.approx(guardadas.loc[f.corte, "prob"], abs=TOLERANCIA_TF32), f.corte


@pytest.mark.skipif(config.RAIZ is None, reason="sin el dataset (CI)")
def test_app_igual_que_el_pipeline_de_entrenamiento_en_cortes_de_train(conjunto):
    from src.data.carga import cargar_en_memoria, rutas_corte
    from src.training.entrenar import predecir
    uc = config.utils()
    samples = uc.cargar_samples()
    filas = samples[samples.split == "train"].sample(12, random_state=1)
    imagenes = cargar_en_memoria(filas, "prueba")
    referencia = np.mean([predecir(red, imagenes, "cpu") for red in conjunto.redes], axis=0)
    for i, f in enumerate(filas.itertuples()):
        datos = [p.read_bytes() for p in rutas_corte(f, config.RAIZ)]
        x = np.stack([inferencia.validar_fase(d, fase)[0] for d, fase in zip(datos, FASES)])
        assert inferencia.predecir(conjunto, x)["prob"] == pytest.approx(float(referencia[i]), abs=TOLERANCIA)


class Espia(torch.nn.Module):
    """Red de mentira que apunta como estan los interruptores de TF32 al predecir."""
    def __init__(self, fallar=False):
        super().__init__()
        self.vistos, self.fallar = [], fallar

    def forward(self, x):
        self.vistos.append((torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))
        if self.fallar:
            raise RuntimeError("fallo a proposito")
        return x.mean(dim=(1, 2, 3))


def test_probabilidades_sin_tf32_y_restaura_los_valores():
    antes = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    imagenes = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
    try:
        for valores in ((True, True), (False, True)):
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = valores
            espia = Espia()
            inferencia.probabilidades(espia, imagenes, "cpu")
            assert espia.vistos == [(False, False)]
            assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == valores
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(RuntimeError):
            inferencia.probabilidades(Espia(fallar=True), imagenes, "cpu")
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = antes


def test_orden_de_la_defensa_igual_que_la_app(conjunto, capsys):
    """`python -m src.inferencia --pre --early --late` escribe exactamente lo que muestra la app."""
    for base in pd.read_csv(EJEMPLOS / "ejemplos.csv").corte:
        x, _ = corte_ejemplo(base)
        app = inferencia.predecir(conjunto, x)
        rutas = [EJEMPLOS / f"{base}_{f}.png" for f in FASES]
        assert inferencia.pipeline(conjunto, rutas)["prob"] == app["prob"]          # exacto
        inferencia.main(["--pre", str(rutas[0]), "--early", str(rutas[1]), "--late", str(rutas[2])])
        salida = capsys.readouterr().out
        assert f"Probabilidad de pCR: {inferencia.texto_probabilidad(app['prob'])} " in salida
        assert f"Clase: {app['clase']} " in salida


def test_prediccion_reproducible(conjunto):
    x, _ = corte_ejemplo(pd.read_csv(EJEMPLOS / "ejemplos.csv").corte[0])
    assert inferencia.predecir(conjunto, x)["prob"] == inferencia.predecir(conjunto, x)["prob"]


# --------------------------------------------------------------------------- #
# Errores y avisos
# --------------------------------------------------------------------------- #

def test_png_corrupto():
    bueno = png(np.full((256, 256), 100, np.uint8))
    with pytest.raises(inferencia.ErrorEntrada, match="corrupto"):
        inferencia.validar_fase(bueno[:len(bueno) // 2], "PRE")             # truncado
    with pytest.raises(inferencia.ErrorEntrada, match="corrupto"):
        inferencia.validar_fase(inferencia.FIRMA_PNG + b"basura" * 50, "PRE")
    dañado = bytearray(bueno)
    dañado[40] ^= 0xFF                                                         # CRC roto
    with pytest.raises(inferencia.ErrorEntrada, match="corrupto"):
        inferencia.validar_fase(bytes(dañado), "PRE")


def test_no_png_vacio_o_demasiado_grande():
    with pytest.raises(inferencia.ErrorEntrada, match="no es un PNG"):
        inferencia.validar_fase(b"\xff\xd8\xff\xe0" + b"0" * 100, "EARLY")    # cabecera JPEG
    with pytest.raises(inferencia.ErrorEntrada, match="vacio"):
        inferencia.validar_fase(b"", "EARLY")
    with pytest.raises(inferencia.ErrorEntrada, match="maximo"):
        inferencia.validar_fase(inferencia.FIRMA_PNG + b"0" * (inferencia.TAMANO_MAXIMO + 1), "EARLY")


def test_tamano_erroneo_no_se_redimensiona():
    with pytest.raises(inferencia.ErrorEntrada, match="128x128"):
        inferencia.validar_fase(png(np.zeros((128, 128), np.uint8)), "LATE")
    with pytest.raises(inferencia.ErrorEntrada, match="256x300"):
        inferencia.validar_fase(png(np.zeros((300, 256), np.uint8)), "LATE")


def test_valores_no_finitos():
    x = np.full((3, 256, 256), 100.0)
    x[1, 10, 10] = np.nan
    with pytest.raises(inferencia.ErrorEntrada, match="no finitos"):
        inferencia.comprobar_corte(x)
    x[1, 10, 10] = np.inf
    with pytest.raises(inferencia.ErrorEntrada, match="no finitos"):
        inferencia.comprobar_corte(x)


def test_fuera_de_rango_y_forma_erronea(conjunto):
    with pytest.raises(inferencia.ErrorEntrada, match="0-255"):
        inferencia.comprobar_corte(np.full((3, 256, 256), 300.0))
    with pytest.raises(inferencia.ErrorEntrada, match="forma"):
        inferencia.predecir(conjunto, np.zeros((2, 256, 256), np.uint8))


def test_color_y_16_bits_se_rechazan_y_gris_rgb_se_acepta():
    rng = np.random.default_rng(0)
    color = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    with pytest.raises(inferencia.ErrorEntrada, match="color"):
        inferencia.validar_fase(png(color), "PRE")
    with pytest.raises(inferencia.ErrorEntrada, match="modo"):
        inferencia.validar_fase(png(rng.integers(0, 65535, (256, 256)).astype(np.uint16)), "PRE")
    gris = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    x, avisos = inferencia.validar_fase(png(np.repeat(gris[..., None], 3, axis=2)), "PRE")
    assert np.array_equal(x, gris) and avisos


def test_avisos_fases_intercambiadas_y_nombres():
    x = np.zeros((3, 256, 256), np.uint8)
    x[0] += 120                                          # PRE mas brillante que EARLY
    x[1] += 80
    avisos = inferencia.avisos_corte(x, {"PRE": "a_EARLY.png", "EARLY": "a_PRE.png", "LATE": "a_LATE.png"})
    assert any("intercambiadas" in a for a in avisos)
    assert sum("sufijo" in a for a in avisos) == 2


def test_ejemplos_son_de_train():
    if config.RAIZ is None:
        pytest.skip("sin el dataset (CI)")
    samples = config.utils().cargar_samples().set_index("sample_id")
    for corte in pd.read_csv(EJEMPLOS / "ejemplos.csv").corte:
        assert samples.loc[corte, "split"] == "train"


def test_metricas_internas_salen_de_los_json():
    m = inferencia.metricas_internas()
    test = json.loads((REPO / "results/test/resumen.json").read_text(encoding="utf-8"))
    assert m["test"]["por_paciente"]["auc"]["auc"] == test["por_paciente"]["auc"]["auc"]
    assert m["umbral_fuera_de_fold"]["umbral"] == pytest.approx(0.3635, abs=5e-5)
    assert m["validacion"]["auc_medio_epocas"]["media"] == pytest.approx(0.551, abs=1e-3)
