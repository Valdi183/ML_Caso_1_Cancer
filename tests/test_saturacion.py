"""Saturacion de la zona que mas realza (src/utils/saturacion.py) con cortes sinteticos."""

import numpy as np
import pandas as pd
import pytest

from src import visualizacion as vis
from src.utils.saturacion import resumir, saturacion_corte, zona_que_mas_realza


def corte_sintetico(early_tumor: int = 255, late_tumor: int = 200, semilla: int = 0) -> np.ndarray:
    """Mama (disco) con un tumor pequeño que realza; fondo negro."""
    rng = np.random.default_rng(semilla)
    yy, xx = np.mgrid[:256, :256]
    mama = (yy - 128) ** 2 + (xx - 128) ** 2 < 90 ** 2
    tumor = (yy - 120) ** 2 + (xx - 140) ** 2 < 10 ** 2
    pre = np.where(mama, 60, 0) + rng.integers(0, 10, (256, 256))
    early, late = pre + np.where(mama, 20, 0), pre + np.where(mama, 25, 0)
    early[tumor], late[tumor] = early_tumor, late_tumor
    return np.clip(np.stack([pre, early, late]), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_zona_es_la_de_curva_captacion(semilla):
    x = corte_sintetico(early_tumor=252, late_tumor=240, semilla=semilla)
    zona = zona_que_mas_realza(x)
    curva = vis.curva_captacion(x)
    assert zona.shape[1] == curva["n_pixeles"]
    assert zona.mean(axis=1) == pytest.approx([curva["medias"][f] for f in ("PRE", "EARLY", "LATE")])
    s = saturacion_corte(x)
    assert s["early_250"] == pytest.approx(curva["saturada_early"])
    assert s["late_250"] == pytest.approx(curva["saturada_late"])


def test_tumor_a_255_en_early_cuenta_y_a_252_solo_con_250():
    a255 = saturacion_corte(corte_sintetico(early_tumor=255, late_tumor=200))
    assert a255["early_255"] > 0.5 and a255["late_255"] == 0
    a252 = saturacion_corte(corte_sintetico(early_tumor=252, late_tumor=200))
    assert a252["early_255"] == 0 and a252["early_250"] > 0.5


def test_resumir_por_corte_y_por_paciente():
    tabla = pd.DataFrame({"patient_id": ["a", "a", "a", "b", "b"],
                          "early_255": [1.0, 0.9, 0.1, 0.0, 0.6], "late_255": [0.0] * 5,
                          "early_250": [1.0] * 5, "late_250": [0.0] * 5})
    r = resumir(tabla)
    assert r["early_255"]["fraccion_cortes"] == pytest.approx(3 / 5)
    assert r["early_255"]["fraccion_pacientes_mayoria_de_cortes"] == pytest.approx(1 / 2)   # b: 1 de 2
    assert r["early_y_late_255_fraccion_cortes"] == 0
