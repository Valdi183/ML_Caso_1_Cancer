"""Tests de src/data/tumor.py (sin dataset)."""

from __future__ import annotations

import pandas as pd

from src.data.tumor import caja_en_imagen


def paciente(n_xy=256, sraw=110, eraw=192, scol=59, ecol=144):
    return pd.Series({"n_xy": n_xy, "sraw": sraw, "eraw": eraw, "scol": scol, "ecol": ecol})


def test_giro_de_90_grados():
    # filas = 256 - ecol .. 256 - scol ; columnas = sraw .. eraw
    assert caja_en_imagen(paciente()) == (112, 197, 110, 192)


def test_escalado_por_n_xy():
    # Volumen de 128 px: todo se multiplica por 2
    assert caja_en_imagen(paciente(n_xy=128, sraw=10, eraw=40, scol=20, ecol=50)) == (156, 216, 20, 80)


def test_volumenes_grandes_no_fiables():
    assert caja_en_imagen(paciente(n_xy=384)) is None
    assert caja_en_imagen(paciente(n_xy=512)) is None


def test_caja_dentro_de_la_imagen():
    f0, f1, c0, c1 = caja_en_imagen(paciente(sraw=0, eraw=300, scol=0, ecol=300))
    assert 0 <= f0 < f1 <= 256 and 0 <= c0 < c1 <= 256
