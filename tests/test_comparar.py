"""Tests de src/utils/comparar.py (sin dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.comparar import brier, diferencia_pareada


def test_brier():
    assert brier([1, 0], [1, 0]) == 0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    # Predecir la prevalencia q da q (1 - q)
    y = np.array([1, 0, 0, 0])
    assert brier(np.full(4, y.mean()), y) == pytest.approx(0.25 * 0.75)


def pacientes(percentil, pcr):
    return pd.DataFrame({"percentil": percentil, "pcr": pcr}, index=[f"p{i}" for i in range(len(pcr))])


def test_diferencia_pareada_mismo_modelo_es_cero():
    rng = np.random.default_rng(0)
    a = pacientes(rng.random(60), rng.integers(0, 2, 60))
    d = diferencia_pareada(a, a, remuestreos=200, semilla=1)
    assert d["diferencia"] == 0 and d["ic95"] == [0, 0] and d["pacientes"] == 60


def test_diferencia_pareada_perfecto_frente_a_invertido():
    pcr = np.array([0, 1] * 20)
    a = pacientes(1 - pcr + 0.01 * np.arange(40), pcr)      # AUC 0
    b = pacientes(pcr + 0.01 * np.arange(40), pcr)          # AUC 1
    d = diferencia_pareada(a, b, remuestreos=200, semilla=1)
    assert d["diferencia"] == pytest.approx(1.0)
    assert d["ic95"][0] == pytest.approx(1.0)


def test_diferencia_pareada_solo_pacientes_comunes():
    a = pacientes([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
    b = pacientes([0.1, 0.9, 0.2], [0, 1, 0])
    assert diferencia_pareada(a, b, remuestreos=50, semilla=1)["pacientes"] == 3
