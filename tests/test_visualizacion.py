"""Lo que la app ensena alrededor de la prediccion (src/visualizacion.py) y la interfaz.

Imagenes sinteticas con zonas de cinetica conocida; las cifras se comparan con los JSON
de results/. No hace falta el dataset."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import inferencia
from src import visualizacion as vis

REPO = Path(__file__).resolve().parents[1]

# Zonas de 20x20 px: (fila, columna, PRE, EARLY, LATE) sobre tejido 60/70/70 y fondo 0
ZONAS = {"persistente": (40, 40, 60, 160, 200), "meseta": (40, 120, 60, 160, 165),
         "lavado": (120, 40, 60, 160, 120), "sin_tipo": (120, 120, 60, 255, 255),
         "lavado_saturado": (190, 120, 60, 255, 150)}


def corte_sintetico(zonas=ZONAS) -> np.ndarray:
    x = np.zeros((3, 256, 256), np.uint8)
    x[:, 20:236, 20:236] = np.array([60, 70, 70], np.uint8)[:, None, None]
    for fila, col, *valores in zonas.values():
        x[:, fila:fila + 20, col:col + 20] = np.array(valores, np.uint8)[:, None, None]
    return x


def centro(nombre):
    fila, col = ZONAS[nombre][:2]
    return fila + 10, col + 10


# --------------------------------------------------------------------------- #
# Cinetica
# --------------------------------------------------------------------------- #

def test_suavizar_conserva_lo_constante_y_la_forma():
    img = np.full((256, 256), 77, np.uint8)
    assert np.allclose(vis.suavizar(img), 77)
    assert vis.suavizar(np.zeros((30, 40))).shape == (30, 40)


def test_tipo_curva_umbrales_y_saturacion():
    assert vis.tipo_curva(0.25) == "persistente"
    assert vis.tipo_curva(0.05) == vis.tipo_curva(-0.05) == "meseta"
    assert vis.tipo_curva(-0.25) == "lavado"
    # EARLY saturado: el cambio real es igual o menor; solo se puede afirmar lavado
    assert vis.tipo_curva(-0.25, early_saturado=True) == "lavado"
    assert vis.tipo_curva(0.0, early_saturado=True) is None
    # LATE saturado: el cambio real es igual o mayor; solo se puede afirmar persistente
    assert vis.tipo_curva(0.25, late_saturado=True) == "persistente"
    assert vis.tipo_curva(-0.25, late_saturado=True) is None
    assert vis.tipo_curva(-0.5, True, True) is None


def test_mascara_separa_el_tejido_del_fondo():
    m = vis.mascara_mama(corte_sintetico())
    assert m[128, 128] and not m[5, 5]
    assert vis.mascara_mama(np.zeros((3, 256, 256), np.uint8)).all()      # vacia: todo el corte


def test_mapa_colorea_cada_zona_con_su_tipo():
    x = corte_sintetico()
    copia = x.copy()
    mapa = vis.mapa_captacion(x)
    assert np.array_equal(x, copia)                         # no toca la entrada del modelo
    tipo = mapa["tipo"]
    for nombre, esperado in (("persistente", 0), ("meseta", 1), ("lavado", 2), ("lavado_saturado", 2)):
        assert tipo[centro(nombre)] == esperado, nombre
    assert tipo[centro("sin_tipo")] == -1                   # EARLY y LATE a 255: no se puede medir
    assert tipo[200, 60] == -1                              # tejido que no realza
    assert mapa["rgb"].shape == (256, 256, 3) and mapa["rgb"].dtype == np.uint8
    gris = mapa["rgb"][centro("sin_tipo")]
    assert gris[0] == gris[1] == gris[2]                    # sin color
    assert tuple(mapa["rgb"][centro("lavado")]) != (160, 160, 160)
    assert sum(mapa["fracciones"].values()) == pytest.approx(1)
    assert mapa["n_saturados"] > 0


def test_curva_de_la_zona_que_mas_realza():
    zonas = {"lavado": (100, 100, 60, 200, 150)}           # la que mas realza (+140)
    x = corte_sintetico(zonas)
    curva = vis.curva_captacion(x, fraccion=0.005)          # pocos px: dentro de la zona
    assert curva["medias"] == pytest.approx({"PRE": 60, "EARLY": 200, "LATE": 150})
    assert curva["cambio"] == pytest.approx(-0.25)
    assert curva["tipo"] == "lavado"
    assert curva["saturada_early"] == 0


def test_curva_saturada_sin_tipo():
    x = corte_sintetico({"z": (100, 100, 60, 255, 255)})
    curva = vis.curva_captacion(x, fraccion=0.005)
    assert curva["saturada_early"] == 1 and curva["tipo"] is None


# --------------------------------------------------------------------------- #
# Cifras de confianza
# --------------------------------------------------------------------------- #

def test_cifras_salen_de_los_json():
    c = vis.cifras_confianza()
    test = json.loads((REPO / "results/test/resumen.json").read_text(encoding="utf-8"))
    umbral = json.loads((REPO / "results/umbral/resumen.json").read_text(encoding="utf-8"))["normal"]
    assert c["test"]["auc"] == test["por_paciente"]["auc"]["auc"]
    assert c["test"]["VPP"] == test["por_paciente"]["VPP"]
    assert c["test"]["marcadas"] == test["por_paciente"]["VP"] + test["por_paciente"]["FP"]
    assert c["validacion"]["VPP"] == umbral["por_paciente"]["VPP"]
    assert c["tasa_base"] == pytest.approx(0.294, abs=1e-3)
    # Lo que se dice en la app: ~54 % en test, 4 de cada 10 en validacion y 2 en test
    assert vis.por_ciento(c["test"]["auc"], 0) == "54 %"
    assert vis.de_cada_10(c["validacion"]["VPP"]) == 4 and vis.de_cada_10(c["test"]["VPP"]) == 2


def test_cifras_sin_ficheros(tmp_path):
    assert vis.cifras_confianza(tmp_path) == {"tasa_base": None, "validacion": None, "test": None}


def test_de_cada_10_y_formato():
    assert [vis.de_cada_10(p) for p in (0.04, 0.05, 0.425, 0.2, 0.96)] == [0, 1, 4, 2, 10]
    assert vis.coma(0.36349, 4) == "0,3635" and vis.por_ciento(0.29353) == "29,4 %"


# --------------------------------------------------------------------------- #
# Resultado, errores y graficos
# --------------------------------------------------------------------------- #

def test_resultado_json():
    sha = {"a": "1" * 64, "b": "2" * 64}
    r = {"prob": 0.3, "clase": "no pCR", "umbral": 0.3635, "por_modelo": {"a": 0.2, "b": 0.4},
         "dispositivo": "cpu"}
    d = json.loads(vis.resultado_json(r, sha, {"PRE": "x_PRE.png"}))
    assert d["probabilidad_pcr"] == 0.3 and d["clase"] == "no pCR" and d["umbral"] == 0.3635
    assert d["por_modelo"] == r["por_modelo"] and d["sha256"] == sha
    assert d["huella_conjunto"] == vis.huella_conjunto(sha) and len(d["huella_conjunto"]) == 12
    assert vis.huella_conjunto(sha) != vis.huella_conjunto({"b": sha["b"], "a": sha["a"]})


def test_errores_en_lenguaje_llano():
    import io

    from PIL import Image

    def error(datos, fase):
        with pytest.raises(inferencia.ErrorEntrada) as e:
            inferencia.validar_fase(datos, fase)
        return vis.mensaje_error(str(e.value))

    b = io.BytesIO()
    Image.fromarray(np.zeros((512, 512), np.uint8)).save(b, format="PNG")
    assert error(b.getvalue(), "EARLY") == ("EARLY mide 512×512 px; la app necesita 256×256, el tamaño con "
                                            "el que se entrenó el modelo. Sube el PNG original del dataset, "
                                            "sin redimensionar.")
    assert "vacío" in error(b"", "PRE")
    assert "no es un PNG" in error(b"\xff\xd8\xff\xe0" + b"0" * 100, "LATE")
    assert "dañado" in error(inferencia.FIRMA_PNG + b"basura" * 50, "PRE")
    assert vis.mensaje_error("Otro mensaje") == "Otro mensaje"


def test_svg_bien_formados_y_accesibles():
    escala = vis.svg_escala(0.30, 0.3635, 0.294, [0.2, 0.25, 0.3, 0.35, 0.4], destacado=2)
    curva = vis.svg_curva(vis.curva_captacion(corte_sintetico()))
    for svg in (escala, curva):
        raiz = ET.fromstring(svg)
        assert raiz.get("role") == "img" and raiz.get("aria-label")
    assert "umbral 0,3635" in escala and "tasa base 29,4 %" in escala and "0,3000" in escala
    sin_tasa = vis.svg_escala(1.5, 0.3635, None, [0.1] * 5)                # fuera de 0-1: se recorta
    ET.fromstring(sin_tasa)
    assert "tasa base" not in sin_tasa


# --------------------------------------------------------------------------- #
# La interfaz (Streamlit sin navegador)
# --------------------------------------------------------------------------- #

def test_app_con_un_ejemplo_muestra_la_prediccion_del_pipeline():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Un ejemplo").run()
    assert not at.exception and not at.error
    base = pd.read_csv(REPO / "app/ejemplos/ejemplos.csv").corte[0]
    datos, _ = inferencia.corte_de_ejemplo(REPO / "app" / "ejemplos", base)
    x = np.stack([inferencia.validar_fase(datos[f], f)[0] for f in ("PRE", "EARLY", "LATE")])
    prob = inferencia.predecir(inferencia.cargar_conjunto(), x)["prob"]
    textos = " ".join(str(h.proto.body) for h in at.get("html"))
    assert f'class="prob">{inferencia.texto_probabilidad(prob)}<' in textos
    assert "Sin validez clínica" in textos


def test_imagen_frente_a_clinica(tmp_path):
    assert vis.imagen_frente_a_clinica(tmp_path) is None                  # sin fichero: sin panel
    (tmp_path / "combinado").mkdir()
    (tmp_path / "combinado" / "resumen.json").write_text('{"por_fold": {}}', encoding="utf-8")
    assert vis.imagen_frente_a_clinica(tmp_path) is None                  # claves que faltan: sin panel
    ruta = REPO / "results/combinado/resumen.json"
    if not ruta.exists():
        pytest.skip("aun no hay results/combinado/resumen.json")
    d = json.loads(ruta.read_text(encoding="utf-8"))
    c = vis.imagen_frente_a_clinica()
    assert c["clinica"]["media"] == d["por_fold"]["clinico"]["media"]
    assert c["imagen"]["media"] == d["por_fold"]["cnn"]["media"]
    assert c["combinada"]["media"] == d["por_fold"]["clinico + cnn"]["media"]
    epocas = list(d["cnn_auc_medio_epocas_por_fold"].values())
    assert c["imagen_todas_las_epocas"] == pytest.approx(np.mean(epocas))
    test = json.loads((REPO / "results/test/resumen.json").read_text(encoding="utf-8"))
    assert c["imagen_test"] == test["por_paciente"]["auc"]["auc"]
    # Solo test sin combinado: el panel sigue, sin la linea de test
    (tmp_path / "combinado" / "resumen.json").write_text(ruta.read_text(encoding="utf-8"), encoding="utf-8")
    assert vis.imagen_frente_a_clinica(tmp_path)["imagen_test"] is None
