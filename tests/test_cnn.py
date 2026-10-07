"""Tests de la red (src/models/cnn.py). No necesitan el dataset ni GPU.

    pytest -q
"""

from __future__ import annotations

import pytest
import torch

from src.models.cnn import POOLINGS, CNNpCR, Preprocesado, cargar, guardar, parametros

LOTE = torch.rand(2, 3, 256, 256)


@pytest.mark.parametrize("realce, sin_late, canales", [
    (False, False, 3), (True, False, 5), (False, True, 2), (True, True, 3),
])
def test_canales_del_preprocesado(realce, sin_late, canales):
    pre = Preprocesado(realce=realce, sin_late=sin_late)
    assert pre.canales == canales
    assert pre(LOTE).shape == (2, canales, 256, 256)


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("realce", [False, True])
def test_un_logit_por_corte(pooling, realce):
    red = CNNpCR(realce=realce, pooling=pooling).eval()
    assert red(LOTE).shape == (2,)


def test_pooling_desconocido():
    with pytest.raises(ValueError):
        CNNpCR(pooling="suma")


def test_parametros_de_la_red_base():
    # Si cambia, los pesos ya entrenados dejan de corresponder a la red por defecto
    assert parametros(CNNpCR()) == 294_129


def test_normalizar_no_depende_del_brillo():
    # Mismo corte con otra escala de intensidad (otro hospital): misma prediccion
    red = CNNpCR(normalizar=True).eval()
    assert torch.allclose(red(LOTE), red(2 * LOTE + 0.1), atol=1e-4)


def test_normalizar_conserva_el_realce():
    # Una sola media y desviacion para las tres fases: EARLY sigue mas brillante
    # que PRE despues de normalizar
    x = torch.zeros(1, 3, 8, 8)
    x[:, 1] = 0.5
    y = Preprocesado(normalizar=True)(x)
    assert (y[:, 1] > y[:, 0]).all()


def test_atencion_empieza_como_la_media():
    torch.manual_seed(0)
    atencion = CNNpCR(pooling="atencion").eval()
    media = CNNpCR(pooling="media").eval()
    media.load_state_dict({k: v for k, v in atencion.state_dict().items() if "atencion" not in k})
    assert torch.allclose(atencion(LOTE), media(LOTE), atol=1e-6)


def test_atencion_aprende():
    red = CNNpCR(pooling="atencion")
    red(LOTE).sum().backward()
    assert red.cabeza[0].atencion.weight.grad.abs().sum() > 0


def test_mapa_de_atencion_suma_uno():
    red = CNNpCR(pooling="atencion").eval()
    with torch.no_grad():
        mapa = red.cabeza[0].pesos_atencion(red.bloques(red.preprocesado(LOTE)))
    assert mapa.shape == (2, 16, 16)
    assert torch.allclose(mapa.sum(dim=(1, 2)), torch.ones(2))


def test_claves_de_la_red_base_sin_cambios():
    # Los .pt entrenados antes de normalizar/pooling solo tienen estas claves de cabeza
    claves = [k for k in CNNpCR().state_dict() if k.startswith("cabeza")]
    assert claves == ["cabeza.3.weight", "cabeza.3.bias"]


@pytest.mark.parametrize("opciones", [{}, {"realce": True, "normalizar": True, "pooling": "atencion"}])
def test_guardar_y_cargar(tmp_path, opciones):
    red = CNNpCR(**opciones).eval()
    ruta = tmp_path / "red.pt"
    guardar(red, ruta, umbral=0.4)
    cargada, info = cargar(ruta)
    assert info == {"umbral": 0.4}
    assert cargada.config == red.config
    assert not cargada.training
    assert torch.allclose(cargada(LOTE), red(LOTE))


def test_carga_pesos_antiguos(tmp_path):
    # Un .pt de antes de normalizar/pooling (config sin esas claves) sigue cargando
    red = CNNpCR().eval()
    config_antigua = {k: v for k, v in red.config.items() if k not in ("normalizar", "pooling")}
    ruta = tmp_path / "antiguo.pt"
    torch.save({"config": config_antigua, "pesos": red.state_dict(), "info": {}}, ruta)
    cargada, _ = cargar(ruta)
    assert cargada.config["pooling"] == "media" and not cargada.config["normalizar"]
    assert torch.allclose(cargada(LOTE), red(LOTE))
