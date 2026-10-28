"""Prediccion del modelo final para la app web, sin Streamlit (testeable).

    from src import inferencia
    conjunto = inferencia.cargar_conjunto()                    # una vez al arrancar
    pre, avisos_pre = inferencia.validar_fase(bytes_pre, "PRE")
    ...
    r = inferencia.predecir(conjunto, np.stack([pre, early, late]))

Modelo: el conjunto de los 5 pesos del congelado, en el orden de results/test/plan.json,
comprobando que el SHA-256 de cada uno coincide con el plan (los mismos pesos que se
evaluaron en test). Probabilidad de un corte = media de las 5 sigmoides, como en
`src.utils.evaluar_test`. Umbral: `umbral_final` de results/umbral/resumen.json (0,3635),
nunca `info["umbral"]` de los .pt (es el de Youden de cada fold).

La imagen se lee con `src.data.imagen` (la misma funcion que el entrenamiento) y pasa por
`probabilidades` (la misma que usa entrenar.py para validar). No hay mas preprocesado:
el resto va dentro de la red. No hace falta la carpeta del dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.data.carga import lotes
from src.data.imagen import FASES, a_entrada, leer_fase
from src.models.cnn import cargar, parametros

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "results" / "test" / "plan.json"
UMBRAL = REPO / "results" / "umbral" / "resumen.json"
RESULTADOS = REPO / "results"
LADO = 256
TAMANO_MAXIMO = 2 * 1024 * 1024          # bytes por fase
FIRMA_PNG = b"\x89PNG\r\n\x1a\n"
MODOS_GRIS = ("L",)                      # como los PNG del dataset (8 bits, gris)
MODOS_CONVERTIBLES = ("LA", "RGB", "RGBA", "P")


class ErrorEntrada(ValueError):
    """Entrada no valida: el mensaje se le muestra tal cual al usuario."""


class ErrorModelo(RuntimeError):
    """El modelo no puede arrancar (falta un fichero o un peso no es el del plan)."""


# --------------------------------------------------------------------------- #
# Prediccion por lotes (la misma que usa entrenar.py en validacion)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def probabilidades(red, imagenes: torch.Tensor, dispositivo) -> np.ndarray:
    """Probabilidad por corte, en el mismo orden que las filas (sin desordenar).
    `imagenes`: (N, 3, 256, 256) uint8."""
    red.eval()
    salidas = [torch.sigmoid(red(a_entrada(imagenes[i].to(dispositivo)))).cpu()
               for i in lotes(len(imagenes), 128, desordenar=False)]
    return torch.cat(salidas).numpy()


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

@dataclass
class Conjunto:
    redes: list
    nombres: list[str]
    sha256: dict[str, str]
    info: dict[str, dict]
    umbral: float
    dispositivo: torch.device
    parametros: int
    config_red: dict = field(default_factory=dict)


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def leer_umbral(ruta: Path = UMBRAL) -> float:
    """`umbral_final` de results/umbral/resumen.json. Sin valor por defecto."""
    if not ruta.exists():
        raise ErrorModelo(f"Falta {ruta.name} ({ruta}): sin el umbral fijado en validacion "
                          "la app no arranca.")
    try:
        umbral = float(json.loads(ruta.read_text(encoding="utf-8"))["normal"]["umbral_final"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise ErrorModelo(f"{ruta} no tiene normal.umbral_final valido: {e}") from e
    if not 0 < umbral < 1:
        raise ErrorModelo(f"Umbral fuera de (0, 1): {umbral}")
    return umbral


def cargar_conjunto(plan: Path = PLAN, ruta_umbral: Path = UMBRAL, dispositivo: str = "cpu") -> Conjunto:
    """Los 5 pesos del plan, comprobados por SHA-256, en eval y en CPU."""
    umbral = leer_umbral(ruta_umbral)
    if not plan.exists():
        raise ErrorModelo(f"Falta {plan}: es la lista de pesos (con su SHA-256) del modelo final.")
    pesos = json.loads(plan.read_text(encoding="utf-8"))["pesos"]
    redes, nombres, huellas, infos = [], [], {}, {}
    for p in pesos:
        ruta = REPO / p["fichero"]
        if not ruta.exists():
            raise ErrorModelo(f"Falta el peso {p['fichero']}.")
        huella = sha256(ruta)
        if huella != p["sha256"]:
            raise ErrorModelo(f"{p['fichero']} no coincide con el SHA-256 de plan.json: "
                              "no es el modelo evaluado.")
        red, info = cargar(ruta, dispositivo)
        red.eval()
        redes.append(red)
        nombres.append(ruta.stem)
        huellas[ruta.stem] = huella
        infos[ruta.stem] = info
    return Conjunto(redes=redes, nombres=nombres, sha256=huellas, info=infos, umbral=umbral,
                    dispositivo=torch.device(dispositivo), parametros=parametros(redes[0]),
                    config_red=dict(redes[0].config))


# --------------------------------------------------------------------------- #
# Validacion de la entrada
# --------------------------------------------------------------------------- #

def validar_fase(datos: bytes, fase: str) -> tuple[np.ndarray, list[str]]:
    """Comprueba un PNG subido y lo lee con la funcion del entrenamiento.
    Devuelve (256, 256) uint8 y avisos que no bloquean. Lanza ErrorEntrada si no vale.
    Solo trabaja con los bytes en memoria: ni rutas ni ejecucion de nada."""
    if not isinstance(datos, (bytes, bytearray)) or len(datos) == 0:
        raise ErrorEntrada(f"{fase}: el fichero esta vacio.")
    if len(datos) > TAMANO_MAXIMO:
        raise ErrorEntrada(f"{fase}: ocupa {len(datos) / 1024**2:.1f} MB; el maximo es "
                           f"{TAMANO_MAXIMO / 1024**2:.0f} MB.")
    if not bytes(datos[:8]) == FIRMA_PNG:
        raise ErrorEntrada(f"{fase}: no es un PNG (la cabecera no es la de PNG).")
    avisos = []
    try:
        with Image.open(io.BytesIO(datos)) as png:
            if png.format != "PNG":
                raise ErrorEntrada(f"{fase}: el formato es {png.format}, no PNG.")
            # El tamaño se lee de la cabecera, antes de descomprimir nada
            if png.size != (LADO, LADO):
                raise ErrorEntrada(f"{fase}: mide {png.size[0]}x{png.size[1]} px; tiene que ser "
                                   f"{LADO}x{LADO} (no se redimensiona).")
            png.verify()                                    # estructura y CRC de cada bloque
        with Image.open(io.BytesIO(datos)) as png:
            png.load()                                      # descomprime: detecta truncados
            modo = png.mode
            if modo not in MODOS_GRIS:
                if modo not in MODOS_CONVERTIBLES:
                    raise ErrorEntrada(f"{fase}: modo de imagen {modo}; se espera un PNG en escala de "
                                       "grises de 8 bits, como los del dataset.")
                rgb = np.asarray(png.convert("RGB"))
                if not ((rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2])).all():
                    raise ErrorEntrada(f"{fase}: la imagen tiene color; se espera escala de grises.")
                avisos.append(f"{fase}: PNG en modo {modo} (gris guardado como {modo}); se convierte a "
                              "escala de grises igual que al entrenar.")
    except ErrorEntrada:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise ErrorEntrada(f"{fase}: el PNG esta corrupto o no se puede leer ({type(e).__name__}).") from e

    x = leer_fase(io.BytesIO(datos))                         # la funcion del entrenamiento
    if x.shape != (LADO, LADO):
        raise ErrorEntrada(f"{fase}: tras leerlo mide {x.shape}; se esperaba ({LADO}, {LADO}).")
    return x, avisos


def comprobar_corte(x) -> np.ndarray:
    """Ultima comprobacion antes de la red: (3, 256, 256), finito, entero en 0-255."""
    x = np.asarray(x)
    if x.shape != (3, LADO, LADO):
        raise ErrorEntrada(f"El corte tiene forma {x.shape}; se esperaba (3, {LADO}, {LADO}).")
    if x.dtype == np.uint8:
        return x
    if not np.issubdtype(x.dtype, np.number):
        raise ErrorEntrada(f"El corte no es numerico ({x.dtype}).")
    if not np.isfinite(x).all():
        raise ErrorEntrada("El corte tiene valores no finitos (NaN o infinito).")
    if x.min() < 0 or x.max() > 255 or not np.array_equal(x, np.round(x)):
        raise ErrorEntrada("El corte tiene valores fuera de 0-255 o no enteros: no es una imagen de 8 bits.")
    return x.astype(np.uint8)


_SUFIJO = re.compile(r"_(PRE|EARLY|LATE)\.png$", re.IGNORECASE)


def avisos_corte(x: np.ndarray, nombres: dict[str, str] | None = None) -> list[str]:
    """Avisos que NO bloquean: fases posiblemente intercambiadas o nombres que no cuadran."""
    avisos = []
    pre, early = float(x[0].mean()), float(x[1].mean())
    if early <= pre:
        avisos.append(f"La media de EARLY ({early:.1f}) no supera a la de PRE ({pre:.1f}): al inyectar el "
                      "contraste el tejido se ilumina, asi que puede que las fases esten intercambiadas.")
    for fase, nombre in (nombres or {}).items():
        sufijo = _SUFIJO.search(nombre or "")
        if sufijo and sufijo.group(1).upper() != fase:
            avisos.append(f"El fichero subido como {fase} se llama «{nombre}» "
                          f"(sufijo _{sufijo.group(1).upper()}).")
    return avisos


# --------------------------------------------------------------------------- #
# Prediccion
# --------------------------------------------------------------------------- #

def predecir(conjunto: Conjunto, x) -> dict:
    """Probabilidad de pCR de un corte (3, 256, 256) con el conjunto de 5 y su clase."""
    x = comprobar_corte(x)
    inicio = time.perf_counter()
    imagenes = torch.from_numpy(np.ascontiguousarray(x[None]))
    por_modelo = np.array([probabilidades(red, imagenes, conjunto.dispositivo)[0] for red in conjunto.redes])
    prob = float(por_modelo.mean())
    segundos = time.perf_counter() - inicio
    if not np.isfinite(prob):
        raise ErrorModelo("La red ha devuelto un valor no finito.")
    es_pcr = prob > conjunto.umbral
    return {"prob": prob, "pcr": es_pcr, "clase": "pCR" if es_pcr else "no pCR", "umbral": conjunto.umbral,
            "por_modelo": dict(zip(conjunto.nombres, por_modelo.tolist())),
            "segundos": segundos, "dispositivo": str(conjunto.dispositivo)}


# --------------------------------------------------------------------------- #
# Metricas internas (leidas de results/, no escritas a mano)
# --------------------------------------------------------------------------- #

def _leer(ruta: Path) -> dict | None:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def metricas_internas(resultados: Path = RESULTADOS) -> dict:
    """Validacion (5 folds), umbral fuera de fold y test, de los JSON de results/.
    Lo que falte vuelve como None (la app lo indica; no inventa numeros)."""
    def claves(m, *nombres):
        return {k: m[k] for k in nombres}

    r: dict = {"validacion": None, "umbral_fuera_de_fold": None, "test": None}
    comparacion = _leer(resultados / "comparacion" / "resumen.json")
    if comparacion and "congelado" in comparacion:
        c = comparacion["congelado"]
        r["validacion"] = {"auc_medio_epocas": c["auc_medio"], "auc_epoca_elegida": c["auc_elegido"],
                           "auc_global_fuera_de_fold": c["auc_global"], "por_cohorte": c["por_cohorte"],
                           "ejecuciones": c["ejecuciones"]}
    umbral = _leer(resultados / "umbral" / "resumen.json")
    if umbral and "normal" in umbral:
        u = umbral["normal"]
        metricas = ("sensibilidad", "especificidad", "VPP", "VPN", "f1_macro")
        r["umbral_fuera_de_fold"] = {"umbral": u["umbral_final"], "criterio": umbral.get("criterio"),
                                     "pacientes": u["pacientes"],
                                     "por_paciente": claves(u["por_paciente"], *metricas),
                                     "por_corte": claves(u["cortes_sueltos"], "n", *metricas)}
    test = _leer(resultados / "test" / "resumen.json")
    if test:
        metricas = ("sensibilidad", "especificidad", "VPP", "VPN", "f1_macro")
        pp, pc = test["por_paciente"], test["corte_suelto"]
        r["test"] = {"pacientes": test["pacientes"],
                     "por_paciente": {"auc": pp["auc"], **claves(pp, *metricas)},
                     "por_corte": {"auc": pc["auc"], **claves(pc, *metricas)}}
    return r


def corte_de_ejemplo(carpeta: Path, base: str) -> tuple[dict[str, bytes], dict[str, str]]:
    """Bytes y nombres de las tres fases de un ejemplo de app/ejemplos (solo esa carpeta)."""
    datos, nombres = {}, {}
    for fase in FASES:
        ruta = carpeta / f"{base}_{fase}.png"
        if ruta.parent.resolve() != carpeta.resolve():
            raise ErrorEntrada("Ejemplo no valido.")
        datos[fase], nombres[fase] = ruta.read_bytes(), ruta.name
    return datos, nombres
