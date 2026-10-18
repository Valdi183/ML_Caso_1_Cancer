"""Tests de la atencion supervisada con la caja del tumor (sin dataset ni GPU)."""

from __future__ import annotations

import math

import pytest
import torch

from src.data.carga import aumentar
from src.data.tumor import en_rejilla, masa_en_caja, mascaras, perdida_caja
from src.models.cnn import CNNpCR

# Caja de 64x64 px en la imagen de 256: 4x4 celdas de un mapa final de 16x16
CAJA = torch.tensor([[64, 128, 32, 96]])


def rejilla_de(cajas: torch.Tensor) -> torch.Tensor:
    return en_rejilla(mascaras(cajas), 16, 16)


def test_mascara_y_rejilla():
    m = mascaras(CAJA)
    assert m.shape == (1, 1, 256, 256) and m.sum() == 64 * 64
    r = rejilla_de(CAJA)
    assert r.shape == (1, 16, 16) and r.sum() == 16
    assert r[0, 4:8, 2:6].all()


def test_celdas_que_tocan_la_caja():
    # Una caja de 1 px cae en una sola celda; si cruza un borde de celda, en dos
    assert rejilla_de(torch.tensor([[20, 21, 20, 21]])).sum() == 1
    assert rejilla_de(torch.tensor([[15, 17, 20, 21]])).sum() == 2


def test_sin_caja_mascara_vacia():
    assert mascaras(torch.tensor([[-1, -1, -1, -1]])).sum() == 0


def test_perdida_atencion_entera_en_la_caja():
    r = rejilla_de(CAJA)
    pesos = r / r.sum()
    assert perdida_caja(pesos, r).item() == pytest.approx(0, abs=1e-4)
    assert masa_en_caja(pesos, r).item() == pytest.approx(1)


def test_perdida_atencion_uniforme():
    r = rejilla_de(CAJA)
    pesos = torch.full((1, 16, 16), 1 / 256)
    assert perdida_caja(pesos, r).item() == pytest.approx(-math.log(16 / 256), rel=1e-4)


def test_cortes_sin_caja_no_aportan():
    cajas = torch.cat([CAJA, torch.tensor([[-1, -1, -1, -1]])])
    r = rejilla_de(cajas)
    pesos = torch.full((2, 16, 16), 1 / 256)
    assert perdida_caja(pesos, r).item() == pytest.approx(-math.log(16 / 256), rel=1e-4)
    assert perdida_caja(pesos[1:], r[1:]).item() == 0


def test_aumentado_mueve_la_mascara_con_la_imagen():
    # Las tres fases son la propia mascara: tras volteos y giro tienen que seguir iguales
    torch.manual_seed(0)
    m = mascaras(CAJA.repeat(8, 1))
    x, m2 = aumentar(torch.cat([m.repeat(1, 3, 1, 1), m], 1)).split([3, 1], dim=1)
    assert torch.equal(x, m2.expand_as(x))
    assert not torch.equal(m2, m)                       # algo se ha movido


def test_con_atencion_igual_que_forward():
    red = CNNpCR(pooling="atencion", filtros=(8, 16, 32), reducir=2).eval()
    torch.nn.init.normal_(red.cabeza[0].atencion.weight)  # atencion no uniforme
    x = torch.rand(2, 3, 256, 256)
    logits, pesos = red.con_atencion(x)
    assert torch.allclose(logits, red(x))
    assert pesos.shape == (2, 16, 16)
    assert torch.allclose(pesos.flatten(1).sum(1), torch.ones(2))


def test_con_atencion_exige_atencion():
    with pytest.raises(ValueError):
        CNNpCR().con_atencion(torch.rand(1, 3, 256, 256))

