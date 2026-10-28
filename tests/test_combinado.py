"""Combinacion clinico + CNN de src/utils/combinado.py con datos sinteticos."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils.combinado import (
    bloque_global,
    diferencia_pareada,
    estandarizar_por_fold,
    logistica,
    logit,
    matriz_clinica,
    razon_verosimilitudes,
)
from src.utils.metricas_extra import oof_clinico


def pacientes_sinteticas(n: int = 1000, senal_cnn: float = 1.0, semilla: int = 0) -> pd.DataFrame:
    """Edad, volumen, subtipo y un logit de CNN independiente de la clinica; pCR depende de
    subtipo, edad y (con peso `senal_cnn`) del logit."""
    rng = np.random.default_rng(semilla)
    subtipo = rng.choice(["HRposHER2neg", "TripleNeg", "HER2pos"], n, p=[0.45, 0.3, 0.25])
    edad, volumen, cnn = rng.normal(50, 10, n), rng.lognormal(2.5, 1, n), rng.normal(-1, 0.5, n)
    z = -1.5 + 1.2 * (subtipo != "HRposHER2neg") - 0.03 * (edad - 50) + senal_cnn * (cnn + 1) / 0.5
    d = pd.DataFrame({"age": edad, "tum_vol": volumen, "HR_HER2_STATUS": subtipo, "logit_cnn": cnn,
                      "pCR": rng.binomial(1, 1 / (1 + np.exp(-z))), "fold": rng.integers(0, 5, n),
                      "dataset": rng.choice(["spy1", "spy2", "duke"], n)})
    d.loc[d.index[:3], ["age", "tum_vol", "HR_HER2_STATUS"]] = np.nan    # faltan, como en el dataset
    return d


def test_logit_invierte_la_sigmoide():
    p = np.array([0.1, 0.5, 0.9])
    assert 1 / (1 + np.exp(-logit(p))) == pytest.approx(p)


def test_estandarizar_por_fold_media_0_desviacion_1_en_cada_fold():
    rng = np.random.default_rng(1)
    fold = np.repeat(np.arange(5), 50)
    x = rng.normal(0, 1, 250) * (fold + 1) + 3 * fold          # otra escala en cada fold
    z = pd.Series(estandarizar_por_fold(x, fold)).groupby(fold)
    assert z.mean().to_numpy() == pytest.approx(np.zeros(5), abs=1e-12)
    assert z.std().to_numpy() == pytest.approx(np.ones(5))


def test_logistica_coincide_con_sklearn_sin_penalizar():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (2000, 3))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + X @ [1.0, -0.5, 0.0]))))
    r = logistica(y, X)
    sk = LogisticRegression(C=np.inf, max_iter=1000).fit(X, y)
    assert r["beta"] == pytest.approx(np.r_[sk.intercept_, sk.coef_[0]], abs=1e-4)
    assert r["ee"].shape == (4,) and (r["ee"] > 0).all()


def test_matriz_clinica_sin_huecos_y_sin_la_categoria_mas_frecuente():
    d = pacientes_sinteticas()
    X = matriz_clinica(d)
    assert not X.isna().any().any()
    assert "HR_HER2_STATUS_HRposHER2neg" not in X and "HR_HER2_STATUS_TripleNeg" in X
    assert X.age.mean() == pytest.approx(0, abs=1e-9)
    assert any(c.startswith("dataset_") for c in matriz_clinica(d, cohorte=True))


def test_razon_verosimilitudes_detecta_la_senal_y_no_el_ruido():
    d = pacientes_sinteticas(senal_cnn=1.0)
    con = razon_verosimilitudes(d.pCR, matriz_clinica(d), d.logit_cnn)
    assert con["p_valor"] < 1e-6 and con["ic95_wald"][0] > 0
    assert con["coeficiente"] == pytest.approx(2.0, abs=0.5)        # 1 / 0,5 en la simulacion
    ruido = razon_verosimilitudes(d.pCR, matriz_clinica(d), np.random.default_rng(3).normal(0, 1, len(d)))
    assert ruido["estadistico_lr"] >= 0 and ruido["p_valor"] > 0.01
    assert ruido["ic95_wald"][0] < 0 < ruido["ic95_wald"][1]


def test_combinacion_anidada_no_usa_las_etiquetas_del_fold_que_predice():
    d = pacientes_sinteticas()
    columnas = (["age", "tum_vol", "logit_cnn"], ["HR_HER2_STATUS"])
    prob, _ = oof_clinico(d, *columnas)
    cambiada = d.copy()
    en_0 = cambiada.fold == 0
    cambiada.loc[en_0, "pCR"] = 1 - cambiada.loc[en_0, "pCR"]
    prob_cambiada, _ = oof_clinico(cambiada, *columnas)
    assert prob[en_0].to_numpy() == pytest.approx(prob_cambiada[en_0].to_numpy())
    assert not np.allclose(prob[~en_0], prob_cambiada[~en_0])


def test_combinar_sube_el_auc_si_la_cnn_tiene_senal_independiente():
    d = pacientes_sinteticas(senal_cnn=1.0)
    _, solo = oof_clinico(d, ["age", "tum_vol"], ["HR_HER2_STATUS"])
    _, combinado = oof_clinico(d, ["age", "tum_vol", "logit_cnn"], ["HR_HER2_STATUS"])
    assert diferencia_pareada(combinado, solo)["folds_a_favor"] == 5


def test_diferencia_pareada():
    r = diferencia_pareada({0: 0.7, 1: 0.6, 2: 0.65}, {0: 0.6, 1: 0.62, 2: 0.6})
    assert r["media"] == pytest.approx(0.13 / 3)
    assert r["folds_a_favor"] == 2 and r["por_fold"][1] == pytest.approx(-0.02)


def test_bloque_global_diferencia_pareada_cero_con_el_mismo_modelo():
    d = pacientes_sinteticas(n=300)
    p = 1 / (1 + np.exp(-d.logit_cnn.to_numpy()))
    r = bloque_global({"clinico": p, "copia": p.copy()}, d.pCR, d.dataset, remuestreos=50)
    dif = r["copia"]["global"]["diferencia_con_clinico"]
    assert dif["valor"] == 0 and dif["ic95"] == [0, 0]
    assert "diferencia_con_clinico" not in r["clinico"]["global"]
    lo, hi = r["clinico"]["mismo_hospital"]["ic95"]
    assert lo <= r["clinico"]["mismo_hospital"]["auc"] <= hi
