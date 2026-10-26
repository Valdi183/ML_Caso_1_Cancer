"""Tests de la regla del umbral (src/utils/umbral.py), con datos sinteticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.umbral import fiabilidad, metricas, umbral_conjunto, umbral_especificidad, umbrales_por_fold


def sinteticos(semilla=0, por_fold=220, desplazamientos=(0, 0, 0, 0, 0)):
    """5 folds con ~30 % de positivos y algo de separacion entre clases. `desplazamientos`
    mueve la escala de probabilidad de cada fold (como pasa entre modelos de folds distintos)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for fold, desplazamiento in enumerate(desplazamientos):
        pcr = rng.random(por_fold) < 0.3
        prob = np.clip(0.25 + desplazamiento + 0.04 * pcr + 0.08 * rng.standard_normal(por_fold), 0, 1)
        filas.append(pd.DataFrame({"fold": fold, "prob": prob, "pcr": pcr.astype(int)}))
    return pd.concat(filas, ignore_index=True)


def test_umbral_de_cada_fold_da_especificidad_090():
    d = sinteticos()
    for fold, t in umbrales_por_fold(d).items():
        de = d[d.fold == fold]
        assert metricas(de.prob, de.pcr, t)["especificidad"] >= 0.90


def test_umbral_con_empates_no_baja_de_090():
    # Muchas probabilidades iguales: prediciendo pCR solo si prob > umbral sigue cumpliendo
    negativos = np.array([0.2] * 50 + [0.3] * 45 + [0.4] * 5)
    t = umbral_especificidad(negativos)
    assert (negativos <= t).mean() >= 0.90


@pytest.mark.parametrize("desplazamientos", [(0, 0, 0, 0, 0), (0, -0.03, 0, 0.07, 0.05)])
def test_umbral_oficial_cumple_en_el_conjunto(desplazamientos):
    # La regla oficial: percentil 90 de todas las pCR=0 juntas. Cumple aunque cambie la escala
    d = sinteticos(semilla=3, desplazamientos=desplazamientos)
    esp = metricas(d.prob, d.pcr, umbral_conjunto(d))["especificidad"]
    assert 0.90 <= esp < 0.91


def test_la_mediana_de_los_t_k_no_basta_si_cambia_la_escala():
    # Por que se cambio la regla: con escalas distintas por fold, la mediana de los t_k
    # deja por debajo los folds desplazados hacia arriba (el 0,839 de la iteracion 10)
    d = sinteticos(semilla=3, desplazamientos=(0, -0.03, 0, 0.07, 0.05))
    t = float(np.median(list(umbrales_por_fold(d).values())))
    assert metricas(d.prob, d.pcr, t)["especificidad"] < 0.90


def test_metricas_y_siempre_no_pcr():
    m = metricas([0.9, 0.1, 0.8, 0.2], [1, 1, 0, 0], 0.5)
    assert (m["VP"], m["FN"], m["FP"], m["VN"]) == (1, 1, 1, 1)
    assert m["sensibilidad"] == m["especificidad"] == m["VPP"] == m["VPN"] == 0.5
    assert m["f1_macro"] == pytest.approx(0.5)
    s = metricas([0.9, 0.1, 0.8, 0.2], [1, 1, 0, 0], None)
    assert s["sensibilidad"] == 0 and s["especificidad"] == 1 and np.isnan(s["VPP"])
    assert s["pcr1"]["f1"] == 0 and s["pcr0"]["f1"] == pytest.approx(2 * 0.5 / 1.5)


def test_fiabilidad_diez_grupos():
    rng = np.random.default_rng(1)
    prob = rng.random(1000)
    f = fiabilidad(prob, rng.random(1000) < prob)
    assert len(f) == 10 and f.pacientes.sum() == 1000
    assert (np.diff(f.prob_media) > 0).all()
