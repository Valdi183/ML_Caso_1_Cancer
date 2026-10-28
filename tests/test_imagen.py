"""La lectura del repo (src/data/imagen.py) da EXACTAMENTE lo mismo que la de antes
(`np.rint(utils_caso.cargar_imagen(f) * 255)`, que usaba cargar_en_memoria)."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src import config
from src.data.carga import rutas_corte
from src.data.imagen import a_entrada, leer_corte, leer_fase

EJEMPLOS = Path(__file__).resolve().parents[1] / "app" / "ejemplos"


def lectura_antigua(origen) -> np.ndarray:
    """utils_caso.cargar_fase (float32 / 255) seguido de lo que hacia cargar_en_memoria."""
    with Image.open(origen) as png:
        x = np.asarray(png.convert("L"), dtype=np.float32) / 255.0
    return np.rint(x * 255).astype(np.uint8)


def png(array: np.ndarray, modo: str = "L") -> io.BytesIO:
    b = io.BytesIO()
    Image.fromarray(array, modo).save(b, format="PNG")
    b.seek(0)
    return b


def test_ida_y_vuelta_exacta_en_los_256_valores():
    v = np.arange(256, dtype=np.uint8)
    assert np.array_equal(np.rint((v.astype(np.float32) / 255.0) * 255).astype(np.uint8), v)


def test_igual_que_la_lectura_antigua_en_png_sintetico():
    x = np.random.default_rng(0).integers(0, 256, (256, 256), dtype=np.uint8)
    assert np.array_equal(leer_fase(png(x)), lectura_antigua(png(x)))
    assert leer_fase(png(x)).dtype == np.uint8


def test_igual_que_la_lectura_antigua_en_los_ejemplos_de_la_app():
    for ruta in sorted(EJEMPLOS.glob("*.png")):
        assert np.array_equal(leer_fase(ruta), lectura_antigua(ruta)), ruta.name


def test_a_entrada_es_la_division_entre_255_de_siempre():
    x = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    assert torch.equal(a_entrada(x), x.float() / 255)
    assert a_entrada(x).dtype == torch.float32


@pytest.mark.skipif(config.RAIZ is None, reason="sin el dataset (CI)")
def test_igual_que_utils_caso_en_cortes_del_dataset():
    uc = config.utils()
    samples = uc.cargar_samples()
    filas = samples[samples.split == "train"].sample(30, random_state=0)
    for f in filas.itertuples():
        antes = np.rint(uc.cargar_imagen(f) * 255).astype(np.uint8)
        assert np.array_equal(leer_corte(rutas_corte(f, config.RAIZ)), antes), f.sample_id
