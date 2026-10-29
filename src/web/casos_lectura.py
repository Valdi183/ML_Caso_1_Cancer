"""Casos de prueba de la lectura del PNG: lo que la web (JS) tiene que aceptar y rechazar
exactamente igual que la app (`src.inferencia.validar_fase`).

    python -m src.web.casos_lectura          # escribe web/pruebas/casos/ y casos.json

Cada caso es un fichero PNG (o unos bytes que la pagina construye, si son muy grandes) y
lo que dice Python de el: el mensaje de error, o los avisos y el SHA-256 de la fase leida
(256 x 256 uint8). La pagina web/pruebas/lectura.html lee los mismos bytes con
web/js/png.js y compara. Los PNG raros (2 y 4 bits, 16 bits, paleta, entrelazado, los
cinco filtros, CRC roto, zlib truncado...) se escriben a mano con `png_crudo`, porque PIL
no sabe guardar la mayoria. Ninguno sale del dataset salvo un ejemplo de app/ejemplos.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from src import inferencia
from src.data.imagen import FASES

REPO = Path(__file__).resolve().parents[2]
SALIDA = REPO / "web" / "pruebas"
EJEMPLOS = REPO / "app" / "ejemplos"
FASE = "PRE"            # todos los casos sueltos se validan como PRE (el mensaje la nombra)


# --------------------------------------------------------------------------- #
# Escritor de PNG crudo
# --------------------------------------------------------------------------- #

def bloque(tipo: bytes, datos: bytes, crc: int | None = None) -> bytes:
    crc = zlib.crc32(tipo + datos) if crc is None else crc
    return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", crc & 0xFFFFFFFF)


def ihdr(ancho: int, alto: int, prof: int, tipo: int, entrelazado: int = 0, compresion: int = 0,
         filtro: int = 0) -> bytes:
    return bloque(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, prof, tipo, compresion, filtro,
                                       entrelazado))


CANALES = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ADAM7 = ((0, 0, 8, 8), (0, 4, 8, 8), (4, 0, 8, 4), (0, 2, 4, 4), (2, 0, 4, 2), (0, 1, 2, 2), (1, 0, 2, 1))


def empaquetar(muestras: np.ndarray, prof: int) -> list[bytes]:
    """(alto, ancho, canales) -> una fila de bytes por linea, sin el byte de filtro."""
    filas = []
    for fila in muestras.reshape(muestras.shape[0], -1).astype(np.uint32):
        if prof == 16:
            filas.append(fila.astype(">u2").tobytes())
        elif prof == 8:
            filas.append(fila.astype(np.uint8).tobytes())
        else:
            bits = np.unpackbits(fila.astype(np.uint8)[:, None], axis=1)[:, 8 - prof:]
            filas.append(np.packbits(bits.reshape(-1)).tobytes())
    return filas


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def filtrar(filas: list[bytes], bpp: int, tipos: tuple[int, ...]) -> bytes:
    """Aplica el filtro PNG de cada fila (tipos en ciclo) y antepone su byte."""
    salida, anterior = bytearray(), bytes(len(filas[0]) if filas else 0)
    for i, fila in enumerate(filas):
        t = tipos[i % len(tipos)]
        f = bytearray()
        for x, v in enumerate(fila):
            a = fila[x - bpp] if x >= bpp else 0
            b = anterior[x]
            c = anterior[x - bpp] if x >= bpp else 0
            pred = (0, a, b, (a + b) // 2, paeth(a, b, c))[t]
            f.append((v - pred) & 0xFF)
        salida += bytes([t]) + f
        anterior = fila
    return bytes(salida)


def datos_imagen(muestras: np.ndarray, prof: int, tipo: int, entrelazado: bool, tipos_filtro) -> bytes:
    bpp = max(1, CANALES[tipo] * prof // 8)
    if not entrelazado:
        return filtrar(empaquetar(muestras, prof), bpp, tipos_filtro)
    partes = b""
    for fila0, col0, dfila, dcol in ADAM7:
        sub = muestras[fila0::dfila, col0::dcol]
        if sub.size:
            partes += filtrar(empaquetar(sub, prof), bpp, tipos_filtro)
    return partes


def png_crudo(muestras: np.ndarray, prof: int, tipo: int, *, entrelazado: bool = False,
              tipos_filtro=(0,), antes_idat: list[bytes] = (), despues_idat: list[bytes] = (),
              trozos_idat: int = 1, zlib_datos: bytes | None = None, cabecera: bytes | None = None,
              fin: bytes | None = None) -> bytes:
    if muestras.ndim == 2:
        muestras = muestras[..., None]
    alto, ancho = muestras.shape[:2]
    z = zlib_datos if zlib_datos is not None else zlib.compress(
        datos_imagen(muestras, prof, tipo, entrelazado, tipos_filtro), 6)
    trozo = -(-len(z) // trozos_idat)
    idats = b"".join(bloque(b"IDAT", z[i:i + trozo]) for i in range(0, len(z), trozo))
    return (inferencia.FIRMA_PNG
            + (cabecera if cabecera is not None else ihdr(ancho, alto, prof, tipo, int(entrelazado)))
            + b"".join(antes_idat) + idats + b"".join(despues_idat)
            + (fin if fin is not None else bloque(b"IEND", b"")))


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #

def _gris(semilla: int = 0, lado: int = 256) -> np.ndarray:
    """Degradado con ruido: comprime bien y recorre los 256 valores."""
    rng = np.random.default_rng(semilla)
    y, x = np.mgrid[0:lado, 0:lado]
    base = (x + y) * 255 // (2 * lado - 2)
    return np.clip(base + rng.integers(-6, 7, (lado, lado)), 0, 255).astype(np.uint8)


def _paleta(colores: np.ndarray) -> bytes:
    return bloque(b"PLTE", colores.astype(np.uint8).tobytes())


def casos() -> dict[str, bytes]:
    """nombre -> bytes del fichero."""
    g = _gris()
    g16 = g.astype(np.uint16) * 257 + np.random.default_rng(1).integers(0, 200, g.shape).astype(np.uint16)
    rng = np.random.default_rng(2)
    gris_rgb = np.repeat(g[..., None], 3, axis=2)
    color = gris_rgb.copy()
    color[100, 100] = (10, 200, 30)
    alfa = rng.integers(0, 256, g.shape, dtype=np.uint8)
    bueno = png_crudo(g, 8, 0)
    i_idat = bueno.index(b"IDAT") - 4
    largo_idat = struct.unpack(">I", bueno[i_idat:i_idat + 4])[0]
    z = zlib.compress(datos_imagen(g[..., None], 8, 0, False, (0,)), 6)
    paleta_gris = np.repeat(np.arange(256)[:, None], 3, axis=1)
    paleta_color = paleta_gris.copy()
    paleta_color[200:] = (255, 0, 0)                            # solo indices 200-255 con color

    c: dict[str, bytes] = {}
    c["ejemplo_dataset"] = (EJEMPLOS / "ISPY2-329941_z008_PRE.png").read_bytes()
    c["gris_8"] = bueno
    for t in range(5):
        c[f"filtro_{t}"] = png_crudo(g, 8, 0, tipos_filtro=(t,))
    c["filtros_mezclados"] = png_crudo(g, 8, 0, tipos_filtro=(0, 1, 2, 3, 4, 4, 2))
    c["entrelazado"] = png_crudo(g, 8, 0, entrelazado=True, tipos_filtro=(4, 1, 2, 3, 0))
    c["gris_4"] = png_crudo(g >> 4, 4, 0, tipos_filtro=(1, 4))
    c["gris_2"] = png_crudo(g >> 6, 2, 0, tipos_filtro=(3,))
    c["gris_2_entrelazado"] = png_crudo(g >> 6, 2, 0, entrelazado=True, tipos_filtro=(4, 2))
    c["gris_1"] = png_crudo(g >> 7, 1, 0)
    c["gris_16"] = png_crudo(g16, 16, 0)
    c["rgb_gris"] = png_crudo(gris_rgb, 8, 2, tipos_filtro=(4, 3))
    c["rgb_color"] = png_crudo(color, 8, 2)
    c["rgb16_gris_byte_alto"] = png_crudo(np.repeat(g16[..., None], 3, axis=2), 16, 2, tipos_filtro=(4,))
    rgb16_bajo = np.stack([g16, (g16 & 0xFF00) | 7, (g16 & 0xFF00) | 99], axis=2)
    c["rgb16_distinto_byte_bajo"] = png_crudo(rgb16_bajo, 16, 2)
    c["rgba_gris"] = png_crudo(np.dstack([gris_rgb, alfa]), 8, 6, tipos_filtro=(2, 4))
    c["rgba_color"] = png_crudo(np.dstack([color, alfa]), 8, 6)
    c["rgba16_gris"] = png_crudo(np.dstack([np.repeat(g16[..., None], 3, axis=2), g16]), 16, 6)
    c["la_8"] = png_crudo(np.dstack([g, alfa]), 8, 4, tipos_filtro=(1, 2, 3, 4))
    c["la_16"] = png_crudo(np.dstack([g16, g16]), 16, 4, entrelazado=True)
    c["paleta_8_gris"] = png_crudo(255 - g, 8, 3, antes_idat=[_paleta(paleta_gris)], tipos_filtro=(4,))
    c["paleta_8_color_sin_usar"] = png_crudo(g // 2, 8, 3, antes_idat=[_paleta(paleta_color)])
    c["paleta_8_color_usado"] = png_crudo(g, 8, 3, antes_idat=[_paleta(paleta_color)])
    c["paleta_4_gris"] = png_crudo(g >> 4, 4, 3, antes_idat=[_paleta(paleta_gris[::17])], tipos_filtro=(2,))
    c["paleta_1_entrelazado"] = png_crudo(g >> 7, 1, 3, entrelazado=True,
                                          antes_idat=[_paleta(np.array([[20] * 3, [230] * 3]))])
    c["paleta_corta_indices_fuera"] = png_crudo(g, 8, 3, antes_idat=[_paleta(paleta_gris[:100])])
    c["paleta_sin_plte"] = png_crudo(g, 8, 3)
    c["gris_con_trns"] = png_crudo(g, 8, 0, antes_idat=[bloque(b"tRNS", struct.pack(">H", 128))])
    c["con_gama_e_iccp_falso"] = png_crudo(g, 8, 0, antes_idat=[bloque(b"gAMA", struct.pack(">I", 100000)),
                                                                bloque(b"sRGB", b"\x00")])
    c["bloque_desconocido"] = png_crudo(g, 8, 0, antes_idat=[bloque(b"abCd", b"cualquier cosa")],
                                        despues_idat=[bloque(b"tEXt", b"Comment\x00hola")])
    c["idat_en_trozos"] = png_crudo(g, 8, 0, trozos_idat=7)
    c["datos_tras_iend"] = bueno + b"basura al final"
    c["tamano_128"] = png_crudo(_gris(lado=128), 8, 0)
    c["tamano_256x300"] = png_crudo(np.zeros((300, 256), np.uint8), 8, 0)
    c["tamano_0"] = inferencia.FIRMA_PNG + ihdr(0, 0, 8, 0) + bloque(b"IDAT", zlib.compress(b"")) + \
        bloque(b"IEND", b"")
    c["dimensiones_enormes"] = inferencia.FIRMA_PNG + ihdr(20000, 20000, 8, 0) + \
        bloque(b"IDAT", zlib.compress(b"\x00")) + bloque(b"IEND", b"")
    c["truncado_mitad"] = bueno[:len(bueno) // 2]
    c["firma_y_basura"] = inferencia.FIRMA_PNG + b"basura" * 50
    c["no_png_jpeg"] = b"\xff\xd8\xff\xe0" + b"0" * 100
    c["vacio"] = b""
    roto = bytearray(bueno)
    roto[i_idat + 8 + largo_idat] ^= 0xFF                                       # CRC del IDAT
    c["crc_idat_roto"] = bytes(roto)
    roto = bytearray(bueno)
    roto[8 + 8 + 13] ^= 0xFF                                                    # CRC del IHDR
    c["crc_ihdr_roto"] = bytes(roto)
    c["crc_iend_roto"] = bueno[:-4] + b"\x00\x00\x00\x00"
    c["sin_iend"] = bueno[:-12]
    fin_datos = i_idat + 8 + largo_idat
    c["truncado_tras_datos_idat"] = bueno[:fin_datos]
    c["truncado_en_crc_idat_1"] = bueno[:fin_datos + 1]
    c["truncado_en_crc_idat_3"] = bueno[:fin_datos + 3]
    c["truncado_tras_idat_y_8_de_otro"] = bueno[:fin_datos + 4 + 5]
    c["truncado_en_ihdr"] = bueno[:20]
    c["truncado_tras_ihdr"] = bueno[:33]
    c["solo_firma"] = inferencia.FIRMA_PNG
    c["filtro_invalido"] = png_crudo(g, 8, 0, zlib_datos=zlib.compress(
        b"".join(bytes([5 if i == 40 else 0]) + bytes(r) for i, r in enumerate(g)), 6))
    c["zlib_truncado"] = png_crudo(g, 8, 0, zlib_datos=z[:len(z) // 2])
    c["zlib_sin_adler"] = png_crudo(g, 8, 0, zlib_datos=z[:-4])
    c["zlib_adler_roto"] = png_crudo(g, 8, 0, zlib_datos=z[:-4] + bytes(4))
    c["zlib_datos_de_mas"] = png_crudo(g, 8, 0, zlib_datos=zlib.compress(
        datos_imagen(g[..., None], 8, 0, False, (0,)) + bytes(500)))
    c["zlib_corrupto"] = png_crudo(g, 8, 0, zlib_datos=z[:2] + bytes([0xFF] * 40) + z[42:])
    c["idat_partido_por_otro_bloque"] = (bueno[:i_idat] + bloque(b"IDAT", z[:len(z) // 2])
                                         + bloque(b"tEXt", b"a\x00b") + bloque(b"IDAT", z[len(z) // 2:])
                                         + bloque(b"IEND", b""))
    c["nombre_de_bloque_invalido"] = png_crudo(g, 8, 0, antes_idat=[bloque(b"a#cd", b"x")])
    c["ihdr_corto"] = inferencia.FIRMA_PNG + bloque(b"IHDR", b"\x00\x00\x01\x00" * 2 + b"\x08") + \
        bloque(b"IDAT", z) + bloque(b"IEND", b"")
    c["ihdr_no_es_el_primero"] = (inferencia.FIRMA_PNG + bloque(b"gAMA", struct.pack(">I", 45455))
                                  + bueno[8:])
    c["metodo_filtro_1"] = png_crudo(g, 8, 0, cabecera=ihdr(256, 256, 8, 0, filtro=1))
    c["metodo_compresion_1"] = png_crudo(g, 8, 0, cabecera=ihdr(256, 256, 8, 0, compresion=1))
    c["entrelazado_valor_2"] = png_crudo(g, 8, 0, entrelazado=True, cabecera=ihdr(256, 256, 8, 0, 2))
    c["profundidad_invalida"] = png_crudo(g, 8, 0, cabecera=ihdr(256, 256, 4, 2))
    c["profundidad_invalida_128"] = png_crudo(_gris(lado=128), 8, 0, cabecera=ihdr(128, 128, 4, 2))
    c["tipo_color_invalido"] = png_crudo(g, 8, 0, cabecera=ihdr(256, 256, 8, 5))
    c["bloque_mas_largo_que_el_fichero"] = bueno[:33] + struct.pack(">I", 10**6) + bueno[37:]
    return c


# Ficheros grandes: la pagina los construye (firma PNG + ceros) para no versionar 2 MB.
GRANDES = {"grande_2mb_mas_1": inferencia.TAMANO_MAXIMO + 1, "grande_2_25mb_empate": 2_359_296}


def bytes_grande(n: int) -> bytes:
    return inferencia.FIRMA_PNG + bytes(n - len(inferencia.FIRMA_PNG))


def resultado(datos: bytes, fase: str = FASE) -> dict:
    """Lo que dice la app de unos bytes: el error, o los avisos y la huella de la fase."""
    try:
        x, avisos = inferencia.validar_fase(datos, fase)
    except inferencia.ErrorEntrada as e:
        return {"error": str(e)}
    return {"avisos": avisos, "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            "media": float(x.mean())}


def casos_corte() -> list[dict]:
    """Avisos del corte completo (fases intercambiadas, nombres que no cuadran)."""
    base = "ISPY2-329941_z008"
    normal = {f: f"{base}_{f}.png" for f in FASES}
    lista = [
        {"nombre": "corte_bien", "ficheros": normal, "nombres": normal},
        {"nombre": "corte_nombres_cruzados", "ficheros": normal,
         "nombres": {"PRE": f"{base}_EARLY.png", "EARLY": f"{base}_pre.PNG", "LATE": "sin_sufijo.png"}},
        {"nombre": "corte_pre_y_early_intercambiadas",
         "ficheros": {"PRE": normal["EARLY"], "EARLY": normal["PRE"], "LATE": normal["LATE"]},
         "nombres": normal},
    ]
    for caso in lista:
        x = np.stack([inferencia.validar_fase((EJEMPLOS / caso["ficheros"][f]).read_bytes(), f)[0]
                      for f in FASES])
        caso["avisos"] = inferencia.avisos_corte(x, caso["nombres"])
    return lista


def generar() -> dict:
    todos = casos()
    lista = [{"nombre": n, "fichero": f"casos/{n}.png", **resultado(d)} for n, d in todos.items()]
    lista += [{"nombre": n, "grande": tam, **resultado(bytes_grande(tam))} for n, tam in GRANDES.items()]
    return {"fase": FASE, "casos": lista, "corte": casos_corte()}


def main() -> None:
    carpeta = SALIDA / "casos"
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("*.png"):
        viejo.unlink()
    for nombre, datos in casos().items():
        (carpeta / f"{nombre}.png").write_bytes(datos)
    esperado = generar()
    (SALIDA / "casos.json").write_text(json.dumps(esperado, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in esperado["casos"]:
        print(f"{c['nombre']:32s} {c.get('error') or ('OK ' + ' | '.join(c['avisos']))}")


if __name__ == "__main__":
    main()
