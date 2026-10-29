"""Pruebas en navegador real de la web estatica (Chrome y Firefox). No forman parte de `pytest`.

    pip install playwright && python -m playwright install firefox     # una vez
    python -m src.web.probar_navegador [--navegador chrome firefox] [--solo lectura|pagina]

Sirve web/ con http.server (como `python -m http.server` dentro de web/) y comprueba:
  lectura  web/pruebas/lectura.html: js/png.js acepta y rechaza lo mismo que Python (casos.json)
  pagina   web/predecir.html: los 4 ejemplos dan la misma probabilidad que `python -m src.inferencia`
           a 4 decimales, los casos de error salen con el mensaje de la app y la autocomprobacion pasa
Chrome es el instalado en el equipo (canal "chrome"); Firefox, el de Playwright.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

from src import inferencia
from src.data.imagen import FASES

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
EJEMPLOS = WEB / "ejemplos"


def servir():
    class Manejador(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    Manejador.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                                ".mjs": "text/javascript", ".wasm": "application/wasm"}
    servidor = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                               functools.partial(Manejador, directory=str(WEB)))
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    return servidor, f"http://127.0.0.1:{servidor.server_address[1]}"


def lanzar(p, nombre: str):
    if nombre == "chrome":
        return p.chromium.launch(channel="chrome")
    return p.firefox.launch()


def prueba_lectura(pagina, url: str) -> list[str]:
    pagina.goto(f"{url}/pruebas/lectura.html")
    pagina.wait_for_function("window.__resultado !== undefined", timeout=120_000)
    r = pagina.evaluate("window.__resultado")
    if "error" in r:
        return [f"lectura: {r['error']}"]
    for f in r["filas"]:
        if not f["bien"]:
            print(f"    DISTINTO {f['nombre']}\n      Python: {f['python']}\n      JS:     {f['js']}")
    print(f"    lectura: {r['total'] - r['fallos']}/{r['total']} casos iguales que Python")
    return [f"lectura: {r['fallos']} casos distintos"] if r["fallos"] else []


def prueba_visualizacion(pagina, url: str) -> list[str]:
    pagina.goto(f"{url}/pruebas/visualizacion.html")
    pagina.wait_for_function("window.__resultado !== undefined", timeout=180_000)
    r = pagina.evaluate("window.__resultado")
    if "error" in r:
        return [f"visualizacion: {r['error']}"]
    for f in r["filas"]:
        if not f["bien"]:
            print(f"    DISTINTO {f['nombre']}: {f['detalle']}")
    print(f"    mapa y curva: {r['total'] - r['fallos']}/{r['total']} cortes iguales que Python")
    return [f"mapa y curva: {r['fallos']} cortes distintos"] if r["fallos"] else []


def cli(base: str) -> str:
    """El texto de la probabilidad que escribe `python -m src.inferencia`."""
    rutas = [EJEMPLOS / f"{base}_{f}.png" for f in FASES]
    salida = subprocess.run([sys.executable, "-m", "src.inferencia", "--pre", str(rutas[0]), "--early",
                             str(rutas[1]), "--late", str(rutas[2])], capture_output=True, text=True,
                            cwd=REPO, check=True).stdout
    return re.search(r"Probabilidad de pCR: (\S+)", salida)[1]


def prueba_pagina(pagina, url: str) -> list[str]:
    fallos = []
    pagina.goto(f"{url}/predecir.html")
    pagina.wait_for_function("window.__listo === true", timeout=120_000)
    modelo = json.loads((WEB / "modelo.json").read_text(encoding="utf-8"))
    for ej in modelo["ejemplos"]:
        pagina.click(f"[data-ejemplo='{ej['corte']}']")
        pagina.wait_for_function("c => window.__resultado && window.__resultado.corte === c",
                                 arg=ej["corte"], timeout=60_000)
        r = pagina.evaluate("window.__resultado")
        esperado = cli(ej["corte"])
        mostrado = pagina.inner_text("#probabilidad").strip()
        bien = mostrado == esperado and r["texto"] == esperado
        print(f"    {ej['corte']:22s} web {mostrado}  cli {esperado}  clase {r['clase']:6s} "
              f"{r['ms']:.0f} ms  {'OK' if bien else 'DISTINTO'}")
        if not bien:
            fallos.append(f"{ej['corte']}: web {mostrado} frente a {esperado}")
    # autocomprobacion con el pipeline
    pagina.click("summary:has-text('Comprobar con el pipeline')")      # el boton esta en un desplegable
    pagina.click("#comprobar")
    pagina.wait_for_function("window.__comprobacion !== undefined", timeout=120_000)
    c = pagina.evaluate("window.__comprobacion")
    print(f"    autocomprobacion: {c['correctos']}/{c['total']} correctos, "
          f"diferencia maxima {c['maxima']:.2e}")
    if c["correctos"] != c["total"]:
        fallos.append("autocomprobacion con fallos")
    # errores
    from src.web import casos_lectura
    malos = casos_lectura.casos()
    entradas = {
        "no PNG": ("a.jpg", malos["no_png_jpeg"]),
        "128x128": ("a.png", malos["tamano_128"]),
        "color": ("a.png", malos["rgb_color"]),
        "16 bits": ("a.png", malos["gris_16"]),
        "truncado": ("a.png", malos["truncado_mitad"]),
        "vacio": ("a.png", b""),
    }
    ejemplo = {f: (EJEMPLOS / f"ISPY2-329941_z008_{f}.png").read_bytes() for f in FASES}
    for nombre, (fichero, datos) in entradas.items():
        pagina.reload()
        pagina.wait_for_function("window.__listo === true", timeout=120_000)
        subidos = {"PRE": (fichero, datos),
                   "EARLY": ("ISPY2-329941_z008_EARLY.png", ejemplo["EARLY"]),
                   "LATE": ("ISPY2-329941_z008_LATE.png", ejemplo["LATE"])}
        for fase, (nombre_fichero, contenido) in subidos.items():
            pagina.set_input_files(f"#fichero-{fase}", files=[
                {"name": nombre_fichero, "mimeType": "image/png", "buffer": contenido}])
        pagina.click("#predecir")
        pagina.wait_for_selector("#error:not([hidden])", timeout=20_000)
        mensaje = pagina.inner_text("#error").strip()
        try:
            inferencia.validar_fase(datos, "PRE")
            esperado = None
        except inferencia.ErrorEntrada as e:
            esperado = str(e)
        bien = esperado is not None and esperado in mensaje and not pagina.is_visible("#probabilidad")
        print(f"    error {nombre:9s} {'OK' if bien else 'DISTINTO'}: {mensaje[:90]}")
        if not bien:
            fallos.append(f"error {nombre}: «{mensaje}» frente a «{esperado}»")
    return fallos


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--navegador", nargs="+", default=["chrome", "firefox"], choices=["chrome", "firefox"])
    a.add_argument("--solo", choices=["lectura", "visualizacion", "pagina"])
    args = a.parse_args(argv)
    from playwright.sync_api import sync_playwright
    servidor, url = servir()
    total = 0
    with sync_playwright() as p:
        for nombre in args.navegador:
            print(f"== {nombre}")
            navegador = lanzar(p, nombre)
            pagina = navegador.new_page()
            consola = []
            pagina.on("console", lambda m: consola.append(m.text) if m.type == "error" else None)
            pagina.on("pageerror", lambda e: consola.append(str(e)))
            fallos = []
            if args.solo in (None, "lectura"):
                fallos += prueba_lectura(pagina, url)
            if args.solo in (None, "visualizacion"):
                fallos += prueba_visualizacion(pagina, url)
            if args.solo in (None, "pagina"):
                fallos += prueba_pagina(pagina, url)
            for linea in consola:
                print(f"    consola: {linea}")
            fallos += [f"consola: {linea}" for linea in consola]
            print(f"  -> {'TODO BIEN' if not fallos else 'FALLOS: ' + '; '.join(fallos)}")
            total += len(fallos)
            navegador.close()
    servidor.shutdown()
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
