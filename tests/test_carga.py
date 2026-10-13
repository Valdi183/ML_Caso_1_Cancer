"""Tests de src/data/carga.py que no necesitan el dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.carga import aplicar_objetivo

FILAS = pd.DataFrame({
    "patient_id": ["A", "A", "B", "C", "C"],
    "pCR": [1, 1, 0, 0, 0],
})
COHORTE_DE = pd.Series({"A": "spy1", "B": "duke", "C": "duke"})


def test_pcr_no_cambia_nada():
    assert aplicar_objetivo(FILAS, "pcr", COHORTE_DE) is FILAS


def test_cohorte_como_etiqueta():
    salida = aplicar_objetivo(FILAS, "duke", COHORTE_DE)
    assert salida.pCR.tolist() == [0, 0, 1, 1, 1]
    assert FILAS.pCR.tolist() == [1, 1, 0, 0, 0]          # el original no se toca


def test_misma_etiqueta_en_todos_los_cortes_de_una_paciente():
    salida = aplicar_objetivo(FILAS, "spy1", COHORTE_DE)
    assert salida.groupby("patient_id").pCR.nunique().eq(1).all()


def test_paciente_sin_cohorte():
    with pytest.raises(ValueError):
        aplicar_objetivo(FILAS, "duke", COHORTE_DE.drop("B"))


def test_objetivo_desconocido():
    with pytest.raises(ValueError):
        aplicar_objetivo(FILAS, "edad", COHORTE_DE)
