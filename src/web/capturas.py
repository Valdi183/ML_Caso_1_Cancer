"""Capturas de la web a 1440, 1024 y 390 px en Chrome y Firefox, para revisarlas a ojo.

    python -m src.web.capturas CARPETA [--paginas index predecir] [--navegador chrome firefox]

Para cada pagina y anchura guarda una captura de pagina completa y avisa de errores de consola y de
scroll horizontal (la pagina no debe desbordar en ninguna anchura). Es una herramienta de desarrollo:
no forma parte de `pytest`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.web.probar_navegador import lanzar, servir

ANCHOS = (1440, 1024, 390)
ESPERA = {"index": "window.__listo === true", "predecir": "window.__listo === true"}


def capturar(pagina, ruta: Path, ancho: int, alto: int) -> None:
    """Captura de pagina completa. Chrome no pasa de 16.384 px por captura (a partir de ahi repite el
    contenido), asi que las paginas largas se capturan en tramos y se cosen."""
    if alto <= 15000:
        pagina.screenshot(path=str(ruta), full_page=True)
        return
    import io

    from PIL import Image
    tramo = 8000
    lienzo = Image.new("RGB", (ancho, alto))
    for y in range(0, alto, tramo):
        zona = {"x": 0, "y": y, "width": ancho, "height": min(tramo, alto - y)}
        trozo = pagina.screenshot(full_page=True, clip=zona)
        lienzo.paste(Image.open(io.BytesIO(trozo)).convert("RGB"), (0, y))
    lienzo.save(ruta)


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("carpeta", type=Path)
    a.add_argument("--paginas", nargs="+", default=["index", "predecir"], choices=["index", "predecir"])
    a.add_argument("--navegador", nargs="+", default=["chrome", "firefox"], choices=["chrome", "firefox"])
    a.add_argument("--ejemplo", default="ISPY2-329941_z008", help="ejemplo que se pulsa en predecir")
    args = a.parse_args(argv)
    args.carpeta.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    servidor, url = servir()
    problemas = 0
    with sync_playwright() as p:
        for nombre in args.navegador:
            navegador = lanzar(p, nombre)
            for pagina_nombre in args.paginas:
                for ancho in ANCHOS:
                    pagina = navegador.new_page(viewport={"width": ancho, "height": 900})
                    consola = []
                    pagina.on("console", lambda m, c=consola: c.append(m.text) if m.type == "error" else None)
                    pagina.on("pageerror", lambda e, c=consola: c.append(str(e)))
                    pagina.goto(f"{url}/{pagina_nombre}.html")
                    pagina.wait_for_function(ESPERA[pagina_nombre], timeout=120_000)
                    if pagina_nombre == "predecir":
                        pagina.click(f"[data-ejemplo='{args.ejemplo}']")
                        pagina.wait_for_function("window.__resultado && window.__resultado.corte",
                                                 timeout=60_000)
                    pagina.wait_for_timeout(3200)           # que acabe la transicion de la portada
                    desborde = pagina.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                    ruta = args.carpeta / f"{pagina_nombre}_{ancho}_{nombre}.png"
                    alto = pagina.evaluate("document.documentElement.scrollHeight")
                    capturar(pagina, ruta, ancho, alto)
                    estado = "OK" if desborde <= 0 and not consola else "REVISAR"
                    print(f"{nombre:8s} {pagina_nombre:9s} {ancho:5d} px  alto {alto:6d}  "
                          f"desborde {desborde:4d}  "
                          f"consola {len(consola)}  {estado}  {ruta.name}")
                    for linea in consola:
                        print(f"           consola: {linea}")
                    problemas += int(desborde > 0) + len(consola)
                    pagina.close()
            navegador.close()
    servidor.shutdown()
    return 1 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
