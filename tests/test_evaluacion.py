"""Tests de src/utils/evaluacion.py (sin dataset)."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.evaluacion import auc_mismo_hospital

COHORTE = ["A", "A", "A", "B", "B", "B"]
PCR = [1, 0, 0, 1, 1, 0]


def test_constante_dentro_de_cada_cohorte_da_05():
    # Solo conoce el hospital: todos los pares de la misma cohorte empatan
    assert auc_mismo_hospital([0.9, 0.9, 0.9, 0.1, 0.1, 0.1], PCR, COHORTE) == 0.5


def test_perfecta_da_1():
    # Ordena bien dentro de cada hospital aunque B puntue todo por debajo de A
    assert auc_mismo_hospital([0.9, 0.8, 0.7, 0.3, 0.25, 0.2], PCR, COHORTE) == 1.0


def test_media_de_cohortes_ponderada_por_pares():
    rng = np.random.default_rng(0)
    prob, pcr = rng.random(60), rng.integers(0, 2, 60)
    cohorte = np.repeat(["A", "B", "C"], 20)
    pares = [(pcr[cohorte == c] == 1).sum() * (pcr[cohorte == c] == 0).sum() for c in "ABC"]
    aucs = [roc_auc_score(pcr[cohorte == c], prob[cohorte == c]) for c in "ABC"]
    assert math.isclose(auc_mismo_hospital(prob, pcr, cohorte), np.average(aucs, weights=pares))


def test_sin_pares_da_nan():
    assert math.isnan(auc_mismo_hospital([0.5, 0.4], [1, 1], ["A", "B"]))
