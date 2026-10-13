"""Tests de src/data/carga.py que no necesitan el dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.carga import aplicar_objetivo

FILAS = pd.DataFrame({
    "patient_id": ["A", "A", "B", "C", "C", "D"],
    "pCR": [1, 1, 0, 0, 0, 1],
})
PACIENTES = pd.DataFrame({
    "dataset": ["spy1", "duke", "duke", "spy2"],
    "TripleNeg": [1.0, 0.0, 0.0, np.nan],          # D: subtipo desconocido
    "HER2pos": [0.0, 1.0, 0.0, np.nan],
    "HRposHER2neg": [0.0, 0.0, 1.0, np.nan],
}, index=pd.Index(["A", "B", "C", "D"], name="pid"))


def test_pcr_no_cambia_nada():
    assert aplicar_objetivo(FILAS, "pcr", PACIENTES) is FILAS


def test_cohorte_como_etiqueta():
    salida = aplicar_objetivo(FILAS, "duke", PACIENTES)
    assert salida.pCR.tolist() == [0, 0, 1, 1, 1, 0]
    assert FILAS.pCR.tolist() == [1, 1, 0, 0, 0, 1]          # el original no se toca


@pytest.mark.parametrize("objetivo, esperado", [
    ("triple", [1, 1, 0, 0, 0]),
    ("her2", [0, 0, 1, 0, 0]),
    ("hrpos", [0, 0, 0, 1, 1]),
])
def test_subtipo_como_etiqueta_y_sin_dato_fuera(objetivo, esperado):
    salida = aplicar_objetivo(FILAS, objetivo, PACIENTES)
    assert "D" not in set(salida.patient_id)                  # sin subtipo: se quita
    assert salida.pCR.tolist() == esperado


def test_misma_etiqueta_en_todos_los_cortes_de_una_paciente():
    for objetivo in ("spy1", "triple"):
        salida = aplicar_objetivo(FILAS, objetivo, PACIENTES)
        assert salida.groupby("patient_id").pCR.nunique().eq(1).all()


def test_objetivo_desconocido():
    with pytest.raises(ValueError):
        aplicar_objetivo(FILAS, "edad", PACIENTES)
