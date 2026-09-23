r"""Localiza el dataset BreastDCEDL sin rutas escritas a mano.

El dataset NO vive en el repositorio: son 1,27 GB en 38.109 ficheros. Cada
maquina lo guarda en un sitio distinto (el portatil en `C:\bdcedl`, el equipo
del laboratorio donde haya disco rapido), asi que ningun script debe llevar una
ruta fija dentro. Se resuelve en este orden:

    1. la variable de entorno  BDCEDL
    2. el fichero  .bdcedl_ruta  junto a este modulo (lo escribe configurar.ps1)
    3. rutas habituales: ./breastdcedl, C:/bdcedl, ~/bdcedl, D:/bdcedl

Uso desde cualquier script o cuaderno del repositorio:

    import config
    uc = config.utils()            # utils_caso.py ya importable
    samples = uc.cargar_samples()

`config.RAIZ` es la carpeta del dataset ya validada.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FICHERO_RUTA = AQUI / ".bdcedl_ruta"

#: Ficheros que tienen que existir para considerar valida una carpeta.
TESTIGOS = ("metadata/samples.csv", "utils_caso.py")


def _es_valida(carpeta: Path) -> bool:
    return all((carpeta / t).exists() for t in TESTIGOS)


def _candidatas() -> list[Path]:
    rutas: list[Path] = []

    if entorno := os.environ.get("BDCEDL"):
        rutas.append(Path(entorno))

    if FICHERO_RUTA.exists():
        guardada = FICHERO_RUTA.read_text(encoding="utf-8").strip()
        if guardada:
            rutas.append(Path(guardada))

    rutas += [
        AQUI / "breastdcedl",
        AQUI.parent / "breastdcedl",
        Path("C:/bdcedl"),
        Path("D:/bdcedl"),
        Path.home() / "bdcedl",
        Path("/content/breastdcedl"),          # Google Colab
    ]
    return rutas


def raiz_datos() -> Path:
    """Devuelve la carpeta del dataset, o explica como arreglarlo si no esta."""
    for candidata in _candidatas():
        try:
            resuelta = candidata.expanduser().resolve()
        except OSError:                        # ruta invalida en esta maquina
            continue
        if _es_valida(resuelta):
            return resuelta

    raise FileNotFoundError(
        "No encuentro el dataset BreastDCEDL.\n"
        "Descargalo y apunta a el de una de estas formas:\n"
        # Ojo: "\b" es el caracter de retroceso, no un escape invalido, asi que
        # Python no avisa. Las contrabarras de rutas de Windows van dobladas.
        "  python descargar_datos.py --destino C:\\bdcedl\n"
        "  setx BDCEDL C:\\bdcedl          (y abre una terminal nueva)\n"
        f"  echo C:\\bdcedl > {FICHERO_RUTA}\n"
        "Buscado en:\n  " + "\n  ".join(str(c) for c in _candidatas())
    )


def fijar_ruta(carpeta: Path | str) -> Path:
    """Guarda la ruta en `.bdcedl_ruta` para que no haya que repetirla."""
    resuelta = Path(carpeta).expanduser().resolve()
    if not _es_valida(resuelta):
        raise FileNotFoundError(f"{resuelta} no contiene el dataset")
    FICHERO_RUTA.write_text(str(resuelta), encoding="utf-8")
    return resuelta


def utils():
    """Importa `utils_caso.py` desde la carpeta del dataset y lo devuelve."""
    raiz = str(raiz_datos())
    if raiz not in sys.path:
        sys.path.insert(0, raiz)

    # El analizador del editor marcara este import como no resuelto, y no se
    # equivoca: `utils_caso` vive en la carpeta del dataset, fuera del proyecto,
    # y la linea anterior la añade al sys.path en tiempo de EJECUCION. Ningun
    # analisis estatico puede saber que devolvera raiz_datos(), que lee el disco.
    # En ejecucion funciona; comprobado con cargar_samples() y cargar_imagen().
    import utils_caso  # type: ignore[import-not-found]
    return utils_caso


try:
    RAIZ: Path | None = raiz_datos()
except FileNotFoundError:                      # aun sin descargar; no rompas el import
    RAIZ = None
