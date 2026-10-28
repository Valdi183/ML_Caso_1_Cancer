"""Metricas descriptivas de src/utils/metricas_extra.py con datos sinteticos."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.utils.metricas_extra import (
    brier_skill,
    calibracion,
    icc,
    permutacion_auc,
    potencia_formula,
    precision_media,
    referencia_azar,
    tabla_umbrales,
    umbral_youden_tabla,
)


def test_brier_skill_cero_con_la_constante():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    assert brier_skill(np.full(500, 0.3), y, 0.3) == pytest.approx(0.0)


def test_brier_skill_positivo_si_mejora_a_la_constante():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 5000)
    y = rng.binomial(1, p)
    assert brier_skill(p, y, y.mean()) > 0.2


def test_pendiente_uno_con_probabilidades_calibradas():
    rng = np.random.default_rng(2)
    p = 1 / (1 + np.exp(-rng.normal(-0.8, 1.0, 20000)))
    y = rng.binomial(1, p)
    c = calibracion(p, y)
    assert c["pendiente"] == pytest.approx(1.0, abs=0.08)
    assert c["intercepto"] == pytest.approx(0.0, abs=0.08)
    assert c["intercepto_pendiente_1"] == pytest.approx(0.0, abs=0.05)


def test_pendiente_menor_que_uno_si_son_demasiado_extremas():
    rng = np.random.default_rng(3)
    z = rng.normal(-0.8, 1.0, 20000)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    extremas = 1 / (1 + np.exp(-2 * z))                 # el doble de seguras de lo que deberian
    assert calibracion(extremas, y)["pendiente"] == pytest.approx(0.5, abs=0.05)


def test_ap_igual_a_la_prevalencia_con_puntuaciones_al_azar():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.3, 20000)
    assert precision_media(rng.uniform(size=20000), y) == pytest.approx(y.mean(), abs=0.02)


def test_permutacion_auc_coincide_con_sklearn_y_no_rechaza_al_azar():
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.3, 200)
    p = rng.uniform(size=200)
    r = permutacion_auc(p, y, permutaciones=2000)
    assert r["auc"] == pytest.approx(roc_auc_score(y, p))
    assert r["p_dos_colas"] > 0.05


def test_permutacion_por_corte_mueve_pacientes_enteras():
    # 60 pacientes con 10 cortes casi identicos: con la señal fuerte, p pequeño, y el
    # AUC por corte es el de sklearn sobre los cortes
    rng = np.random.default_rng(6)
    y_pac = np.r_[np.ones(20), np.zeros(40)].astype(int)
    base = np.where(y_pac == 1, 0.7, 0.3) + rng.normal(0, 0.05, 60)
    grupos = np.repeat(np.arange(60), 10)
    prob = base[grupos] + rng.normal(0, 0.01, 600)
    r = permutacion_auc(prob, y_pac[grupos], grupos, permutaciones=2000)
    assert r["auc"] == pytest.approx(roc_auc_score(y_pac[grupos], prob))
    assert r["p_una_cola"] < 0.01


def test_icc_uno_sin_variacion_dentro_y_cero_sin_variacion_entre():
    grupos = np.repeat(np.arange(50), 8)
    rng = np.random.default_rng(7)
    assert icc(rng.normal(size=50)[grupos], grupos) == pytest.approx(1.0)
    assert icc(rng.normal(size=400), grupos) == pytest.approx(0.0, abs=0.1)


def test_youden_separa_del_todo_datos_separables():
    prob = np.r_[np.linspace(0.1, 0.4, 30), np.linspace(0.6, 0.9, 10)]
    y = np.r_[np.zeros(30), np.ones(10)].astype(int)
    u = umbral_youden_tabla(prob, y)
    assert 0.4 <= u < 0.6


def test_azar_tiene_vpp_igual_a_la_prevalencia():
    y = np.r_[np.ones(30), np.zeros(70)].astype(int)
    r = referencia_azar(y)
    assert r["VPP"] == pytest.approx(0.3)
    assert r["VP"] + r["FN"] == pytest.approx(30)
    assert r["VN"] == pytest.approx(63)


def test_tabla_umbrales_cumple_cada_especificidad():
    rng = np.random.default_rng(8)
    y = rng.binomial(1, 0.3, 1000)
    prob = rng.uniform(size=1000) + 0.1 * y
    filas = tabla_umbrales(prob, y, 0.5)
    for fila, objetivo in zip(filas, (0.95, 0.90, 0.80, 0.70, 0.50)):
        assert fila["especificidad"] >= objetivo
    assert [f["criterio"] for f in filas[-2:]] == ["siempre no pCR", "azar con especificidad 0,90 (esperado)"]


def test_potencia_crece_con_el_auc():
    assert potencia_formula(0.55, 53, 123) < potencia_formula(0.65, 53, 123) < potencia_formula(0.75, 53, 123)
    assert potencia_formula(0.5, 53, 123) == pytest.approx(0.05, abs=0.01)
