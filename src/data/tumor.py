"""Caja del tumor en las coordenadas de la imagen de 256x256.

La tabla de pacientes trae `sraw`, `eraw` (filas) y `scol`, `ecol` (columnas) del
recorte del tumor, pero en el sistema de coordenadas del volumen ORIGINAL, que
esta girado 90 grados respecto a los PNG y mide `n_xy` pixeles de lado. Por eso en
el EDA la caja "tal cual" no caia sobre la lesion.

Comprobado a ojo en `notebooks/localizacion_tumor.ipynb` (13-10-2026):

    filas    = 256 - ecol*k  ...  256 - scol*k
    columnas =       sraw*k  ...        eraw*k          con k = 256 / n_xy

- n_xy = 256 (84 % de las pacientes): cae sobre la lesion en las tres cohortes.
- n_xy de 128 a 320: con el escalado tambien cae, salvo excepciones.
- n_xy de 384 a 512 (~11 %, casi todas de spy2): NO cae. Esas imagenes debieron
  de generarse con otro recorte o remuestreo, y la caja no es fiable.

La caja es la del tumor en 3D: la misma para todos los cortes de la paciente.
Solo existe en el dataset, NO en la app: en la defensa llegan las imagenes sin
coordenadas. Cualquier uso en el modelo final tiene que funcionar sin ella.
"""

from __future__ import annotations

import pandas as pd

LADO = 256
#: Por encima de este tamano original la caja no cae sobre la lesion
N_XY_MAXIMO_FIABLE = 320


def caja_en_imagen(paciente: pd.Series) -> tuple[int, int, int, int] | None:
    """(fila0, fila1, col0, col1) del tumor en la imagen de 256x256, o None si la
    caja de esa paciente no es fiable (volumen original de mas de 320 px).

    `paciente`: una fila de `utils_caso.cargar_patients()`.
    """
    n_xy = float(paciente["n_xy"])
    if n_xy > N_XY_MAXIMO_FIABLE:
        return None
    k = LADO / n_xy
    f0, f1 = LADO - paciente["ecol"] * k, LADO - paciente["scol"] * k
    c0, c1 = paciente["sraw"] * k, paciente["eraw"] * k

    def recorta(v: float) -> int:
        return int(round(min(max(v, 0), LADO)))

    caja = recorta(f0), recorta(f1), recorta(c0), recorta(c1)
    if caja[1] <= caja[0] or caja[3] <= caja[2]:
        return None
    return caja
