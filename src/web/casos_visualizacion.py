"""Casos de prueba del mapa y la curva de captacion en JS: lo que dice Python (src/visualizacion.py).

    python -m src.web.casos_visualizacion       # escribe web/pruebas/visualizacion.json

24 cortes: los 4 ejemplos de train con las tres fases en las 6 ordenes posibles (las
permutaciones dan cortes con EARLY menos brillante que PRE, LATE mas que EARLY, etc., que el
ejemplo original no cubre). Para cada uno: la curva de captacion y, del mapa, las fracciones, los
recuentos y el SHA-256 de `tipo` (int8) y de `rgb` (uint8). La pagina web/pruebas/visualizacion.html
calcula lo mismo con web/js/visualizacion.js y lo compara.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from src import visualizacion as vis
from src.data.imagen import FASES, leer_corte

REPO = Path(__file__).resolve().parents[2]
EJEMPLOS = REPO / "web" / "ejemplos"
SALIDA = REPO / "web" / "pruebas" / "visualizacion.json"


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def casos() -> list[dict]:
    with open(EJEMPLOS / "ejemplos.csv", encoding="utf-8", newline="") as f:
        cortes = [r["corte"] for r in csv.DictReader(f)]
    lista = []
    for corte in cortes:
        x = leer_corte([EJEMPLOS / f"{corte}_{f}.png" for f in FASES])
        for orden in itertools.permutations(range(3)):
            y = x[list(orden)]
            curva, mapa = vis.curva_captacion(y), vis.mapa_captacion(y)
            lista.append({
                "corte": corte, "orden": list(orden),
                "curva": {"medias": curva["medias"], "cambio": curva["cambio"],
                          "n_pixeles": curva["n_pixeles"],
                          "tipo": curva["tipo"], "saturada_early": curva["saturada_early"],
                          "saturada_late": curva["saturada_late"]},
                "mapa": {"fracciones": mapa["fracciones"], "n_coloreados": mapa["n_coloreados"],
                         "n_saturados": mapa["n_saturados"], "n_mama": mapa["n_mama"],
                         "sha256_tipo": huella(mapa["tipo"]), "sha256_rgb": huella(mapa["rgb"])}})
    return lista


def main() -> None:
    datos = {"aviso": "Generado por src/web/casos_visualizacion.py; no se edita a mano.", "casos": casos()}
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{SALIDA.relative_to(REPO)}: {len(datos['casos'])} casos")


if __name__ == "__main__":
    main()
