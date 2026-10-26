"""Tests de los calculos de src/utils/figuras.py (sin dataset ni GPU)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.utils import figuras


def historial(auc_val, auc_train=None):
    h = pd.DataFrame({"epoca": range(1, len(auc_val) + 1), "auc_val": auc_val,
                      "perdida_train": 0.6, "perdida_val": np.linspace(0.65, 0.6, len(auc_val)),
                      "auc_val_suave": auc_val, "lr": 1e-3})
    if auc_train is not None:
        h["auc_train"] = auc_train
    return h


def test_auc_medio_por_ventana():
    h = historial([0.5] * 15 + [0.7] * 5)
    assert figuras.auc_medio(h) == pytest.approx(0.55)
    assert figuras.auc_medio(h, 15) == pytest.approx(0.5)
    assert np.isnan(figuras.auc_medio(h, columna="auc_val_mismo_hospital"))


def test_en_epoca_sin_columna_o_epoca():
    h = historial([0.5] * 11, auc_train=[0.6] * 11)
    assert figuras.en_epoca(h, "auc_train", 11) == pytest.approx(0.6)
    assert np.isnan(figuras.en_epoca(h, "auc_train", 15))         # paro antes de la 15
    assert np.isnan(figuras.en_epoca(historial([0.5] * 20), "auc_train", 15))   # sin la columna


def test_perdida_constante_normal_es_la_entropia():
    q = 0.294
    assert figuras.perdida_constante(q) == pytest.approx(-(q * np.log(q) + (1 - q) * np.log(1 - q)))
    assert figuras.perdida_constante(q) == pytest.approx(0.606, abs=0.002)


def test_perdida_constante_ponderada_con_n0_n1():
    # Con pos_weight = N0/N1 el optimo es 0,5 y la perdida ln 2 · (wq + 1 - q): el 0,979 de pCR
    q = 0.294
    w = (1 - q) / q
    assert figuras.perdida_constante(q, w) == pytest.approx(np.log(2) * (w * q + 1 - q))
    assert figuras.perdida_constante(q, 2.4) == pytest.approx(0.979, abs=0.003)


def test_sens_esp():
    s, e = figuras.sens_esp([0.9, 0.2, 0.6, 0.1], [1, 1, 0, 0])
    assert (s, e) == (0.5, 0.5)
    s, e = figuras.sens_esp([0.1, 0.2], [0, 0])
    assert np.isnan(s) and e == 1.0


def test_por_paciente():
    pred = pd.DataFrame({"patient_id": ["a", "a", "b"], "prob": [0.2, 0.4, 0.5], "pCR": [1, 1, 0]})
    p = figuras.por_paciente(pred)
    assert p.loc["a", "prob"] == pytest.approx(0.3)
    assert p.loc["a", "cortes"] == 2 and p.loc["a", "pcr"] == 1
    assert np.isnan(p.loc["b", "desviacion"])


def escribir_ejecucion(carpeta, nombre, cfg, auc_val, auc_train=None, elegida=2):
    d = carpeta / nombre
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    historial(auc_val, auc_train).to_csv(d / "historial.csv", index=False)
    metricas = {"mejor_epoca": elegida, "auc_val_media": auc_val[elegida - 1]}
    (d / "metricas.json").write_text(json.dumps(metricas), encoding="utf-8")
    pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "patient_id": ["a", "a", "b", "c"],
                  "pCR": [1, 1, 0, 0], "prob": [0.4, 0.3, 0.2, 0.1]}).to_csv(
        d / "predicciones_val.csv", index=False)


def test_inventario_lee_y_rellena_valores_por_defecto(tmp_path):
    # Antigua (sin filtros ni reducir en config.json) y nueva ponderada
    escribir_ejecucion(tmp_path, "it1_normal_f0", {"fold": 0, "ponderada": False, "parametros": 294129},
                       [0.5] * 15 + [0.6] * 5, auc_train=[0.7] * 20)
    escribir_ejecucion(tmp_path, "it1_ponderada_f0", {"fold": 0, "ponderada": True, "pos_weight": 1.0,
                                                      "filtros": [8, 16, 32], "reducir": 2}, [0.55] * 10)
    inv = figuras.inventario(tmp_path).set_index("nombre")
    assert list(inv.index) == ["it1_normal_f0", "it1_ponderada_f0"]
    a, b = inv.loc["it1_normal_f0"], inv.loc["it1_ponderada_f0"]
    assert a.filtros == "16-32-64-128" and a.reducir == 1 and a.pooling == "media"
    assert a.auc_val_medio == pytest.approx(0.525) and a.auc_val_medio_1_15 == pytest.approx(0.5)
    assert a.auc_train_15 == pytest.approx(0.7) and np.isnan(b.auc_train_15)
    assert a.epocas == 20 and a.epoca_elegida == 2
    # prevalencia de los cortes de validacion 0,5; ponderada con w = 1 es la normal
    assert a.perdida_constante == pytest.approx(np.log(2))
    assert b.perdida_constante == pytest.approx(np.log(2))
    assert "8-16-32 · 128 px · media · ponderada" in b.configuracion
    assert "it2_pequena_reg_f0" in figuras.faltan(tmp_path)


def test_resumen_configuracion():
    inv = pd.DataFrame({"nombre": ["x_f0", "x_f1"], "auc_val_medio": [0.5, 0.6]})
    r = figuras.resumen_configuracion(inv, ["x_f0", "x_f1"])
    assert r["media"] == pytest.approx(0.55) and r["desviacion"] == pytest.approx(np.std([0.5, 0.6], ddof=1))


def test_grupos_bien_formados():
    # Cada configuracion de 5 folds tiene los folds 0-4 en orden, y todo lo agrupado esta en EJECUCIONES
    todas = {n for ns in figuras.EJECUCIONES.values() for n in ns}
    for nombres in figuras.CINCO_FOLDS.values():
        assert [n[-1] for n in nombres] == list("01234")
        assert set(nombres) <= todas
    for nombres in figuras.FOLDS_0_1.values():
        assert [n[-1] for n in nombres] == ["0", "1"] and set(nombres) <= todas
