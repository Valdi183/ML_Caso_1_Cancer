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
Por eso solo se usa al ENTRENAR (`--peso-caja`): empuja la atencion de la red
hacia el tumor; en la app la red es la misma y no la necesita.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

LADO = 256
#: Por encima de este tamano original la caja no cae sobre la lesion
N_XY_MAXIMO_FIABLE = 320


def caja_en_imagen(paciente: pd.Series) -> tuple[int, int, int, int] | None:
    """(fila0, fila1, col0, col1) del tumor en la imagen de 256x256, o None si la
    caja de esa paciente no es fiable (volumen original de mas de 320 px).

    `paciente`: una fila de `utils_caso.cargar_patients()`.
    """
    if paciente[["n_xy", "sraw", "eraw", "scol", "ecol"]].isna().any():
        return None
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


# --------------------------------------------------------------------------- #
# Atencion supervisada con la caja (solo al entrenar; --peso-caja)
# --------------------------------------------------------------------------- #

def cajas_por_corte(filas: pd.DataFrame, pacientes: pd.DataFrame) -> torch.Tensor:
    """(N, 4) int16 con la caja de cada corte (la de su paciente); -1 si no la hay.

    Se guardan 4 numeros por corte y la mascara se construye por lote en la GPU
    (`mascaras`): un canal extra en todas las imagenes cargadas costaria 1/3 mas
    de memoria. `pacientes`: tabla de utils_caso indexada por `pid`.
    """
    por_paciente = {}
    for pid in filas.patient_id.unique():
        caja = caja_en_imagen(pacientes.loc[pid])
        por_paciente[pid] = caja if caja is not None else (-1, -1, -1, -1)
    return torch.tensor([por_paciente[p] for p in filas.patient_id], dtype=torch.int16)


def mascaras(cajas: torch.Tensor, lado: int = LADO) -> torch.Tensor:
    """(B, 4) cajas  ->  (B, 1, lado, lado) float con 1 dentro de la caja.

    Las cajas -1 dan una mascara vacia. Tiene la forma de un canal de imagen
    para pasar por `aumentar` junto a las fases y girarse igual que ellas.
    """
    cajas = cajas.long()
    idx = torch.arange(lado, device=cajas.device)
    f0, f1, c0, c1 = cajas.unbind(1)
    en_filas = (idx >= f0[:, None]) & (idx < f1[:, None])
    en_columnas = (idx >= c0[:, None]) & (idx < c1[:, None])
    return (en_filas[:, :, None] & en_columnas[:, None, :]).float().unsqueeze(1)


def en_rejilla(mascara: torch.Tensor, alto: int, ancho: int) -> torch.Tensor:
    """(B, 1, 256, 256)  ->  (B, alto, ancho): 1 en las celdas del mapa final que
    tocan la caja. El umbral 0,5 deshace la interpolacion de la rotacion."""
    return F.adaptive_max_pool2d((mascara > 0.5).float(), (alto, ancho)).squeeze(1)


def masa_en_caja(pesos: torch.Tensor, rejilla: torch.Tensor) -> torch.Tensor:
    """(B,) parte de la atencion (pesos de suma 1) que cae dentro de la caja."""
    return (pesos * rejilla).flatten(1).sum(1)


def perdida_caja(pesos: torch.Tensor, rejilla: torch.Tensor) -> torch.Tensor:
    """Media, sobre los cortes con caja, de -log(1e-6 + masa de atencion en la caja).

    Vale ~0 si toda la atencion cae en la caja y -log(fraccion) si es uniforme.
    Los cortes sin caja (rejilla vacia) no aportan; si no hay ninguno, vale 0.
    """
    con_caja = rejilla.flatten(1).sum(1) > 0
    if not con_caja.any():
        return pesos.new_zeros(())
    return -torch.log(1e-6 + masa_en_caja(pesos[con_caja], rejilla[con_caja])).mean()
