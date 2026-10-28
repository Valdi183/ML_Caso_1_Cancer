"""Lectura de un corte DCE: la UNICA funcion que convierte PNG en la entrada de la red.

La usan el entrenamiento (`src.data.carga.cargar_en_memoria`) y la app web
(`src.inferencia`), asi que una imagen se lee exactamente igual en los dos sitios:

    PNG -> escala de grises ("L") -> uint8 (256, 256)      leer_fase
    3 fases apiladas PRE, EARLY, LATE -> (3, 256, 256)     leer_corte
    uint8 -> float32 / 255 dentro de torch                  a_entrada

Es la misma operacion que `utils_caso.cargar_fase` (que devuelve directamente
float32 / 255), sin depender de la carpeta del dataset: en el servidor de la app
no hay dataset. tests/test_imagen.py comprueba que da exactamente lo mismo. Ninguna
normalizacion mas: el resto del preprocesado va dentro de la red (src/models/cnn.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image

FASES = ("PRE", "EARLY", "LATE")


def leer_fase(origen: str | Path | BinaryIO) -> np.ndarray:
    """Una fase: (alto, ancho) uint8. `origen` es una ruta o un fichero abierto en binario."""
    with Image.open(origen) as png:
        return np.asarray(png.convert("L"), dtype=np.uint8)


def leer_corte(origenes) -> np.ndarray:
    """(3, alto, ancho) uint8 con las fases en el orden PRE, EARLY, LATE."""
    return np.stack([leer_fase(o) for o in origenes])


def a_entrada(x: torch.Tensor) -> torch.Tensor:
    """De uint8 (0-255) al rango [0, 1] que recibe la red, en float32."""
    return x.float() / 255
