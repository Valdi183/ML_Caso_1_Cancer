"""Validacion de la entrada: las tres fases son el mismo corte de una RM de mama.

src/inferencia.py (validar_corte) y web/js/png.js (validarCorte), con los umbrales de
web/validacion_entrada.json (src/utils/calibrar_entrada.py, calibrados con train). Los ejemplos
y cortes reales de las 3 cohortes se aceptan; pacientes mezcladas, la misma imagen tres veces e
imagenes ajenas se rechazan; y JS y Python aceptan y rechazan exactamente lo mismo (con Chrome
sin ventana, si esta instalado)."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src import inferencia
from src.data.imagen import FASES

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
EJEMPLOS = REPO / "app" / "ejemplos"
CORTES = ["ISPY1_1103_z017", "ISPY2-884266_z021", "ISPY2-329941_z008", "Breast_MRI_290_z010"]


def ejemplo(corte: str) -> tuple[np.ndarray, dict[str, str]]:
    datos, nombres = inferencia.corte_de_ejemplo(EJEMPLOS, corte)
    return np.stack([inferencia.validar_fase(datos[f], f)[0] for f in FASES]), nombres


def sin_nombres(x):
    return inferencia.validar_corte(x, None)


@pytest.fixture(scope="module")
def ejemplos():
    return {c: ejemplo(c) for c in CORTES}


# --------------------------------------------------------------------------- #
# Lo que debe pasar
# --------------------------------------------------------------------------- #


def test_el_json_es_el_de_la_calibracion():
    u = inferencia.leer_umbrales_entrada()
    assert u["version"] == 1 and u["tejido"] == inferencia.TEJIDO and u["bloque"] == inferencia.BLOQUE
    assert set(u["rasgos"]) == set(inferencia.MENSAJES_RASGOS)
    cal = u["calibracion"]
    assert cal["train_con_umbrales_finales"]["aceptados"] == 1.0  # ningun real de train rechazado
    assert all(f["aceptados"] == 1.0 for f in cal["fold4_con_umbrales_de_folds_0_3"].values())
    assert all(f["rechazados"] == 0 for f in cal["cada_fold_con_umbrales_de_los_otros_cuatro"].values())


def test_los_ejemplos_se_aceptan_con_y_sin_nombres(ejemplos):
    for corte, (x, nombres) in ejemplos.items():
        inferencia.validar_corte(x, nombres)
        inferencia.validar_corte(x, None)
        otros = {f: f"subida_{f.lower()}.png" for f in FASES}  # nombres sin el patron
        avisos = inferencia.validar_corte(x, otros)
        assert any("no siguen el patrón" in a for a in avisos), corte


def test_validar_no_cambia_la_imagen_ni_la_prediccion(ejemplos):
    conjunto = inferencia.cargar_conjunto()
    x, nombres = ejemplos[CORTES[0]]
    antes = x.copy()
    p0 = inferencia.predecir(conjunto, x)["prob"]
    inferencia.validar_corte(x, nombres)
    assert np.array_equal(x, antes) and inferencia.predecir(conjunto, x)["prob"] == p0


@pytest.mark.skipif(
    inferencia.REPO.joinpath(".bdcedl_ruta").exists() is False and not os.environ.get("BDCEDL"),
    reason="sin el dataset",
)
def test_cortes_reales_de_las_3_cohortes_se_aceptan():
    from src import config
    from src.data.carga import cargar_en_memoria

    uc = config.utils()
    s = uc.cargar_samples()
    train = s[s.split == "train"].copy()
    train["cohorte"] = train.patient_id.map(uc.cargar_patients().set_index("pid").dataset)
    muestra = train.groupby("cohorte").sample(40, random_state=1)  # solo train, nunca test
    img = cargar_en_memoria(muestra, "muestra").numpy()
    for x, fila in zip(img, muestra.itertuples()):
        nombres = {f: Path(getattr(fila, f"path_{f.lower()}")).name for f in FASES}
        inferencia.validar_corte(x, nombres)


# --------------------------------------------------------------------------- #
# Lo que debe rechazarse
# --------------------------------------------------------------------------- #


def test_pacientes_mezcladas_por_el_nombre(ejemplos):
    (a, na), (b, nb) = ejemplos[CORTES[0]], ejemplos[CORTES[1]]
    x = np.stack([a[0], b[1], b[2]])
    with pytest.raises(
        inferencia.ErrorEntrada, match="PRE es de ISPY1_1103_z017 y EARLY de ISPY2-884266_z021"
    ):
        inferencia.validar_corte(x, {"PRE": na["PRE"], "EARLY": nb["EARLY"], "LATE": nb["LATE"]})


def test_fase_en_otra_casilla_por_el_nombre(ejemplos):
    x, n = ejemplos[CORTES[0]]
    with pytest.raises(inferencia.ErrorEntrada, match="casilla PRE es la fase EARLY"):
        inferencia.validar_corte(x, {"PRE": n["EARLY"], "EARLY": n["PRE"], "LATE": n["LATE"]})


def test_pacientes_mezcladas_por_el_contenido(ejemplos):
    # Sin nombres utiles, ninguna mezcla de los ejemplos pasa en silencio: se rechaza o da un aviso.
    # La tasa de rechazo medida en la calibracion (fold 4) esta en web/validacion_entrada.json.
    veredictos = []
    for a in CORTES:
        for b in CORTES:
            if a == b:
                continue
            x = np.stack([ejemplos[a][0][0], ejemplos[b][0][1], ejemplos[b][0][2]])
            try:
                avisos = sin_nombres(x)
                veredictos.append("aviso" if any(v.startswith("Aviso") for v in avisos) else "pasa")
            except inferencia.ErrorEntrada:
                veredictos.append("error")
    assert "pasa" not in veredictos and "error" in veredictos, veredictos


def test_la_misma_imagen_tres_veces(ejemplos):
    for x, _ in ejemplos.values():
        with pytest.raises(inferencia.ErrorEntrada, match="falta el contraste"):
            sin_nombres(np.stack([x[1]] * 3))


@pytest.mark.parametrize("ajena", ["ruido", "constante", "degradado", "tablero"])
def test_imagenes_ajenas(ajena):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256]
    x = {
        "ruido": rng.integers(0, 256, (3, 256, 256)),
        "constante": np.full((3, 256, 256), 120),
        "degradado": np.stack([xx * f for f in (0.6, 0.8, 0.85)]),
        "tablero": np.stack([((xx // 16 + yy // 16) % 2) * 150] * 3),
    }[ajena]
    with pytest.raises(inferencia.ErrorEntrada, match="no parecen el mismo corte"):
        sin_nombres(np.clip(x, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# JS (web/js/png.js) frente a Python: los mismos rasgos y el mismo veredicto
# --------------------------------------------------------------------------- #


def _chrome() -> str | None:
    for ruta in (
        os.environ.get("CHROME"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ):
        if ruta and Path(ruta).exists():
            return ruta
    return None


def volcar_dom(pagina: Path, perfil: Path, limite: float = 150) -> str:
    """DOM de `pagina` tras ejecutar su JS, con Chrome sin ventana y un perfil temporal. Chrome a
    veces no se cierra tras volcar el DOM: se lee la salida hasta </html> y se cierra entonces."""
    import threading
    import time

    proc = subprocess.Popen([_chrome(), "--headless=new", "--disable-gpu", "--no-first-run",
                             "--no-default-browser-check", f"--user-data-dir={perfil}",
                             "--allow-file-access-from-files", "--virtual-time-budget=10000", "--dump-dom",
                             pagina.as_uri()], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    trozos = []
    lector = threading.Thread(target=lambda: trozos.extend(iter(lambda: proc.stdout.read1(1 << 16), b"")),
                              daemon=True)
    lector.start()
    fin = time.monotonic() + limite
    while time.monotonic() < fin and proc.poll() is None and b"</html>" not in b"".join(trozos[-3:]):
        time.sleep(0.5)
    proc.kill()
    lector.join(5)
    return b"".join(trozos).decode("utf-8", errors="replace")


def casos_de_paridad(ejemplos) -> list[dict]:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:256, 0:256]
    casos = []

    def caso(nombre, x, nombres=None):
        casos.append(
            {"nombre": nombre, "x": np.clip(np.asarray(x), 0, 255).astype(np.uint8), "nombres": nombres}
        )

    for c, (x, n) in ejemplos.items():
        caso(f"real {c}", x, n)
        caso(f"real {c} sin patron", x, {f: f"{f.lower()} (1).png" for f in FASES})
        caso(f"misma x3 {c}", np.stack([x[1]] * 3))
        caso(f"cambiadas {c}", np.stack([x[1], x[0], x[2]]))
        caso(f"casillas cambiadas {c}", x, {"PRE": n["EARLY"], "EARLY": n["PRE"], "LATE": n["LATE"]})
    for a in CORTES:
        for b in CORTES:
            if a != b:
                xa, xb = ejemplos[a][0], ejemplos[b][0]
                caso(f"mezcla {a}+{b}", np.stack([xa[0], xb[1], xb[2]]))
                caso(f"late de otra {a}+{b}", np.stack([xa[0], xa[1], xb[2]]))
                caso(
                    f"mezcla por nombre {a}+{b}",
                    np.stack([xa[0], xb[1], xb[2]]),
                    {
                        "PRE": ejemplos[a][1]["PRE"],
                        "EARLY": ejemplos[b][1]["EARLY"],
                        "LATE": ejemplos[b][1]["LATE"],
                    },
                )
    caso("ruido", rng.integers(0, 256, (3, 256, 256)))
    caso("constante", np.full((3, 256, 256), 90))
    caso("negro", np.zeros((3, 256, 256)))
    caso("degradado", np.stack([xx * f for f in (0.6, 0.8, 0.85)]))
    disco = (xx - 128) ** 2 + (yy - 128) ** 2 < 70**2
    caso("disco que realza", np.stack([disco * v for v in (90, 150, 160)]))
    return casos


def en_python(caso) -> dict:
    r = inferencia.rasgos_corte(caso["x"])
    try:
        return {"rasgos": r, "avisos": inferencia.validar_corte(caso["x"], caso["nombres"])}
    except inferencia.ErrorEntrada as e:
        return {"rasgos": r, "error": str(e)}


@pytest.mark.skipif(_chrome() is None, reason="sin Chrome/Edge para ejecutar el JS")
def test_js_y_python_aceptan_y_rechazan_lo_mismo(ejemplos, tmp_path):
    casos = casos_de_paridad(ejemplos)
    datos = [
        {"nombre": c["nombre"], "nombres": c["nombres"], "x": base64.b64encode(c["x"].tobytes()).decode()}
        for c in casos
    ]
    umbrales = json.loads((WEB / "validacion_entrada.json").read_text(encoding="utf-8"))
    pagina = tmp_path / "paridad.html"
    pagina.write_text(
        f"""<!doctype html><meta charset="utf-8"><pre id="salida"></pre>
<script src="{(WEB / "js" / "png.js").as_uri()}"></script>
<script>
const CASOS = {json.dumps(datos)}, UMBRALES = {json.dumps(umbrales)};
const salida = CASOS.map((c) => {{
  const b = Uint8Array.from(atob(c.x), (ch) => ch.charCodeAt(0)), n = 256 * 256;
  const fases = [b.slice(0, n), b.slice(n, 2 * n), b.slice(2 * n)];
  const r = Png.rasgosCorte(fases);
  try {{ return {{ rasgos: r, avisos: Png.validarCorte(fases, c.nombres, UMBRALES) }}; }}
  catch (e) {{ return {{ rasgos: r, error: e instanceof Png.ErrorEntrada ? e.message : "JS: " + e }}; }}
}});
document.getElementById("salida").textContent = JSON.stringify(salida);
</script>""",
        encoding="utf-8",
    )
    dom = volcar_dom(pagina, tmp_path / "perfil")
    texto = re.search(r'<pre id="salida">(.*?)</pre>', dom, re.S)
    assert texto and texto.group(1), "el JS no devolvio nada"
    js = json.loads(html.unescape(texto.group(1)))
    assert len(js) == len(casos)
    for caso, j in zip(casos, js):
        p = en_python(caso)
        assert j["rasgos"] == p["rasgos"], caso["nombre"]  # bit a bit
        assert j.get("error") == p.get("error"), caso["nombre"]
        assert j.get("avisos") == p.get("avisos"), caso["nombre"]
    veredictos = [("error" in en_python(c)) for c in casos]
    assert any(veredictos) and not all(veredictos)  # prueba las dos ramas
