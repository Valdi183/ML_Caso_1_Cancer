"""Lo que la app ENSEÑA alrededor de la prediccion, sin Streamlit (testeable).

    from src import visualizacion as vis
    mapa = vis.mapa_captacion(x)        # x: (3, 256, 256) uint8, PRE, EARLY, LATE
    curva = vis.curva_captacion(x)
    cifras = vis.cifras_confianza()     # leidas de results/, no escritas a mano

Nada de esto entra en la prediccion: el modelo recibe las tres fases tal cual
(`src.inferencia`). El mapa y la curva de captacion son descriptivos, para mirar el
corte, y dependen del momento en que se adquirio LATE, que cambia segun el hospital.

Cinetica (por pixel en el mapa, por zona en la curva): cambio relativo de EARLY a LATE,
(LATE - EARLY) / EARLY. Mas de +10 %: persistente (sigue subiendo); entre -10 % y +10 %:
meseta; menos de -10 %: lavado. Las tres fases comparten ventana de normalizacion
(GUIA, B2), asi que sus intensidades son comparables entre si. El tumor suele saturar
EARLY (255): ahi el cambio medido es solo una cota (ver `tipo_curva`), y lo que no se
puede afirmar se queda sin tipo y sin color.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from src import config

CAMBIO_MESETA = 0.10          # |cambio relativo EARLY -> LATE| por debajo: meseta
FRACCION_ZONA = 0.02          # la curva: el 2 % de pixeles de la mama que mas realzan
PERCENTIL_REALCE = 90         # el mapa colorea el 10 % de la mama que mas realza...
REALCE_MINIMO = 20            # ...si ademas sube al menos 20 niveles de gris (de 255)
SATURADO = 250                # EARLY o LATE desde aqui: saturado, sin tipo
OPACIDAD = 0.6                # del color sobre EARLY en el mapa

TIPOS = ("persistente", "meseta", "lavado")
COLORES = {"persistente": "#5B8FD4", "meseta": "#D9B648", "lavado": "#D2595A"}
_PCT = f"{100 * CAMBIO_MESETA:.0f} %"
DESCRIPCION = {"persistente": f"sigue subiendo de EARLY a LATE (más de +{_PCT})",
               "meseta": f"se mantiene de EARLY a LATE (entre −{_PCT} y +{_PCT})",
               "lavado": f"baja de EARLY a LATE (más de −{_PCT})"}


# --------------------------------------------------------------------------- #
# Cinetica del contraste
# --------------------------------------------------------------------------- #

def suavizar(img: np.ndarray) -> np.ndarray:
    """Filtro binomial 5x5 (casi gaussiano, sigma ~1 px), solo para ver. float32."""
    nucleo = np.array([1, 4, 6, 4, 1], np.float32) / 16
    y = np.pad(np.asarray(img, np.float32), 2, mode="edge")
    y = sum(w * y[i:i + y.shape[0] - 4, :] for i, w in enumerate(nucleo))
    return sum(w * y[:, i:i + y.shape[1] - 4] for i, w in enumerate(nucleo))


def umbral_otsu(img: np.ndarray) -> float:
    """Umbral de Otsu en 0-255: el que mejor separa el fondo del tejido."""
    hist = np.bincount(np.clip(np.asarray(img), 0, 255).astype(np.int64).ravel(), minlength=256)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        varianza = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    return float(np.nanargmax(np.where(np.isfinite(varianza), varianza, np.nan))) \
        if np.isfinite(varianza).any() else 0.0


def mascara_mama(x: np.ndarray) -> np.ndarray:
    """Tejido frente a fondo: EARLY suavizado por encima del umbral de Otsu calculado sobre
    log(1 + intensidad). En lineal, Otsu separa el tumor brillante del resto; en logaritmo,
    el fondo oscuro del tejido. Si sale vacia, todo el corte."""
    early = suavizar(x[1])
    escala = 255 / np.log(256)
    umbral = np.expm1(umbral_otsu(np.log1p(np.clip(early, 0, 255)) * escala) / escala)
    mascara = early > umbral
    return mascara if mascara.any() else np.ones_like(mascara)


def cambio_relativo(early, late):
    """(LATE - EARLY) / EARLY, con EARLY de al menos 1 para no dividir por cero."""
    early = np.asarray(early, np.float32)
    return (np.asarray(late, np.float32) - early) / np.maximum(early, 1)


def tipo_curva(cambio: float, early_saturado: bool = False, late_saturado: bool = False) -> str | None:
    """Tipo por el cambio relativo de EARLY a LATE. Con una fase saturada el cambio medido es
    una cota: con EARLY a 255 el real es igual o MENOR (EARLY real >= 255), asi que solo se
    puede afirmar lavado; con LATE a 255, igual o MAYOR, y solo se puede afirmar persistente.
    Si no se puede afirmar nada, None."""
    if early_saturado and late_saturado:
        return None
    if cambio > CAMBIO_MESETA:
        return None if early_saturado else "persistente"
    if cambio < -CAMBIO_MESETA:
        return None if late_saturado else "lavado"
    return None if early_saturado or late_saturado else "meseta"


def mapa_captacion(x: np.ndarray) -> dict:
    """EARLY en grises con color, segun el tipo de curva, en los pixeles que mas realzan.

    Devuelve `rgb` (256, 256, 3) uint8, `tipo` (256, 256) con -1 (sin color) o el indice
    en TIPOS, `fracciones` de cada tipo sobre los pixeles coloreados y los recuentos."""
    x = np.asarray(x)
    pre, early, late = (suavizar(f) for f in x)
    mama = mascara_mama(x)
    realce = early - pre
    corte = max(REALCE_MINIMO, float(np.percentile(realce[mama], PERCENTIL_REALCE)))
    realza = mama & (realce >= corte)
    sat_early, sat_late = x[1] >= SATURADO, x[2] >= SATURADO

    # La misma regla que tipo_curva, vectorizada (con saturacion, el cambio es una cota)
    cambio = cambio_relativo(early, late)
    sube, baja = cambio > CAMBIO_MESETA, cambio < -CAMBIO_MESETA
    tipo = np.full(x.shape[1:], -1, np.int8)
    tipo[realza & sube & ~sat_early] = 0
    tipo[realza & ~sube & ~baja & ~sat_early & ~sat_late] = 1
    tipo[realza & baja & ~sat_late] = 2
    con_tipo = tipo >= 0
    saturado = realza & ~con_tipo

    gris = np.repeat(x[1][..., None].astype(np.float32), 3, axis=2)
    rgb = gris.copy()
    for i, nombre in enumerate(TIPOS):
        color = np.array([int(COLORES[nombre][j:j + 2], 16) for j in (1, 3, 5)], np.float32)
        sel = tipo == i
        rgb[sel] = (1 - OPACIDAD) * gris[sel] + OPACIDAD * color
    n = int(con_tipo.sum())
    return {"rgb": np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "tipo": tipo,
            "fracciones": {t: (float((tipo == i).sum()) / n if n else 0.0) for i, t in enumerate(TIPOS)},
            "n_coloreados": n, "n_saturados": int(saturado.sum()), "n_mama": int(mama.sum())}


def curva_captacion(x: np.ndarray, fraccion: float = FRACCION_ZONA) -> dict:
    """Intensidad media en PRE, EARLY y LATE de la zona que mas realza: el `fraccion` de los
    pixeles de la mama con mayor EARLY - PRE (suavizado para elegirlos; las medias, del
    PNG sin suavizar). El tipo sale del cambio relativo de esas medias."""
    x = np.asarray(x)
    mama = mascara_mama(x)
    realce = (suavizar(x[1]) - suavizar(x[0]))[mama]
    k = max(1, int(round(fraccion * realce.size)))
    zona = np.zeros(realce.size, bool)
    zona[np.argsort(realce, kind="stable")[-k:]] = True
    fases = [f[mama][zona].astype(np.float64) for f in x]
    medias = {nombre: float(f.mean()) for nombre, f in zip(("PRE", "EARLY", "LATE"), fases)}
    cambio = float(cambio_relativo(medias["EARLY"], medias["LATE"]))
    sat_early, sat_late = float((fases[1] >= SATURADO).mean()), float((fases[2] >= SATURADO).mean())
    return {"medias": medias, "cambio": cambio, "n_pixeles": k,
            "tipo": tipo_curva(cambio, sat_early > 0.5, sat_late > 0.5),
            "saturada_early": sat_early, "saturada_late": sat_late}


# --------------------------------------------------------------------------- #
# Cifras para "¿Cuanto fiarse?" (leidas de results/, no escritas a mano)
# --------------------------------------------------------------------------- #

def _leer(ruta: Path) -> dict | None:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def cifras_confianza(resultados: Path = config.RESULTADOS) -> dict:
    """Tasa base, validacion fuera de fold con el umbral final y test, por paciente.
    Lo que falte vuelve como None: la app lo dice y no inventa numeros."""
    r: dict = {"tasa_base": None, "validacion": None, "test": None}
    umbral = _leer(resultados / "umbral" / "resumen.json")
    if umbral and "normal" in umbral:
        u, pac = umbral["normal"]["por_paciente"], umbral["normal"]["pacientes"]
        r["tasa_base"] = pac["pcr"] / pac["n"]
        r["validacion"] = {"pacientes": pac["n"], "pcr": pac["pcr"], "marcadas": u["VP"] + u["FP"],
                           "VP": u["VP"], "VPP": u["VPP"], "sensibilidad": u["sensibilidad"],
                           "especificidad": u["especificidad"], "auc": umbral["normal"].get("auc_global")}
    test = _leer(resultados / "test" / "resumen.json")
    if test:
        pp = test["por_paciente"]
        r["test"] = {"pacientes": test["pacientes"]["n"], "pcr": test["pacientes"]["pcr"],
                     "auc": pp["auc"]["auc"], "ic95": pp["auc"]["ic95"], "marcadas": pp["VP"] + pp["FP"],
                     "VP": pp["VP"], "VPP": pp["VPP"], "sensibilidad": pp["sensibilidad"],
                     "especificidad": pp["especificidad"]}
        if test.get("tasa_pcr_train") is not None:
            r["tasa_base"] = test["tasa_pcr_train"]
    return r



def imagen_frente_a_clinica(resultados: Path = config.RESULTADOS) -> dict | None:
    """AUC de validacion (media +- desviacion de los 5 folds) de las variables clinicas solas,
    la imagen sola y las dos combinadas, de results/combinado/resumen.json (solo validacion;
    lo genera `src.utils.combinado`). Si falta el fichero o alguna clave, None: la app no
    muestra el panel. `imagen_test` es el AUC por paciente de la imagen en test (de
    results/test/resumen.json; None si falta): solo la imagen se evaluo en test."""
    d = _leer(resultados / "combinado" / "resumen.json")
    if not d:
        return None
    test = _leer(resultados / "test" / "resumen.json")
    try:
        imagen_test = float(test["por_paciente"]["auc"]["auc"]) if test else None
    except (KeyError, TypeError, ValueError):
        imagen_test = None
    try:
        f = d["por_fold"]

        def fila(clave):
            return {"media": float(f[clave]["media"]), "desviacion": float(f[clave]["desviacion_ddof0"])}

        dif = d["diferencias_por_fold"]["clinico + cnn - clinico"]
        ic = d["global"]["clinico + cnn"]["global"]["diferencia_con_clinico"]["ic95"]
        epocas = [float(v) for v in d["cnn_auc_medio_epocas_por_fold"].values()]
        return {"clinica": fila("clinico"), "imagen": fila("cnn"), "combinada": fila("clinico + cnn"),
                "imagen_todas_las_epocas": float(np.mean(epocas)), "imagen_test": imagen_test,
                "mejora": float(dif["media"]), "folds_a_favor": int(dif["folds_a_favor"]),
                "mejora_global_ic95": [float(v) for v in ic],
                "pacientes": int(d["pacientes"])}
    except (KeyError, TypeError, ValueError):
        return None


def de_cada_10(p: float) -> int:
    """Una proporcion como «n de cada 10», redondeada al entero mas cercano."""
    return int(np.floor(p * 10 + 0.5))


# --------------------------------------------------------------------------- #
# Resultado y mensajes
# --------------------------------------------------------------------------- #

def huella_conjunto(sha256: dict[str, str]) -> str:
    """Version del modelo: SHA-256 de los SHA-256 de los 5 pesos, en su orden, abreviado."""
    return hashlib.sha256("".join(sha256.values()).encode()).hexdigest()[:12]


def resultado_json(r: dict, sha256: dict[str, str], nombres: dict[str, str] | None = None) -> str:
    """El resultado de `inferencia.predecir` como JSON para descargar."""
    return json.dumps({
        "probabilidad_pcr": r["prob"], "clase": r["clase"], "umbral": r["umbral"],
        "regla": "pCR si probabilidad > umbral",
        "por_modelo": r["por_modelo"],
        "sha256": sha256, "huella_conjunto": huella_conjunto(sha256),
        "ficheros": nombres or {}, "dispositivo": r["dispositivo"],
        "aviso": "Uso educativo. Sin validez clinica.",
    }, ensure_ascii=False, indent=2)


_ERRORES = (
    (r"mide (\d+)x(\d+) px", lambda f, m: f"{f} mide {m[1]}×{m[2]} px; la app necesita 256×256, el tamaño "
                                          "con el que se entrenó el modelo. Sube el PNG original del "
                                          "dataset, sin redimensionar."),
    (r"vacio", lambda f, m: f"El fichero de {f} está vacío. Vuelve a exportarlo y súbelo otra vez."),
    (r"ocupa ([\d.]+) MB", lambda f, m: f"{f} ocupa {m[1].replace('.', ',')} MB y el máximo es 2 MB. Los "
                                        "PNG del dataset ocupan mucho menos: comprueba que es el fichero "
                                        "correcto."),
    (r"no es un PNG|no PNG", lambda f, m: f"{f} no es un PNG. Sube el fichero .png de esa fase (no JPEG "
                                          "ni DICOM)."),
    (r"tiene color", lambda f, m: f"{f} tiene color; la app necesita una imagen en escala de grises. "
                                  "Sube el PNG original de la fase, sin colorear."),
    (r"modo de imagen (\S+);", lambda f, m: f"{f} no es un PNG de grises de 8 bits (modo {m[1]}). Sube "
                                            "el PNG original del dataset."),
    (r"corrupto", lambda f, m: f"{f} está dañado o incompleto y no se puede leer. Descárgalo de nuevo."),
)


def mensaje_error(texto: str) -> str:
    """El error de `inferencia.validar_fase` en lenguaje llano: que falla y como arreglarlo.
    Si no reconoce el mensaje, lo devuelve tal cual."""
    fase = re.match(r"(PRE|EARLY|LATE): ", texto)
    for patron, hacer in _ERRORES:
        m = re.search(patron, texto)
        if fase and m:
            return hacer(fase[1], m)
    return texto


# --------------------------------------------------------------------------- #
# Graficos en SVG (sin dependencias; la app los inserta tal cual)
# --------------------------------------------------------------------------- #

TEXTO, TEXTO_2, LINEA = "#E3E5E8", "#A0A6AD", "#5C6066"
BARRA_NO, BARRA_SI = "#3D4045", "#55595F"


def coma(v: float, decimales: int = 2) -> str:
    return f"{v:.{decimales}f}".replace(".", ",").replace("-", "−")


def por_ciento(v: float, decimales: int = 1) -> str:
    return f"{coma(100 * v, decimales)} %"


def svg_escala(prob: float, umbral: float, tasa_base: float | None, por_modelo: list[float],
               destacado: int | None = None) -> str:
    """Escala horizontal de 0 a 1: el corte (triangulo), los 5 modelos (puntos; `destacado`,
    el que no vio a la paciente, en aro), el umbral (linea continua) y la tasa base
    (discontinua). Cada marca tiene forma y rotulo propios: nada depende solo del color."""
    x0, x1 = 20, 420

    def x(p):
        return x0 + (x1 - x0) * min(max(float(p), 0.0), 1.0)

    xp, xu = x(prob), x(umbral)
    p = [f'<rect x="{x0}" y="60" width="{xu - x0:.1f}" height="20" fill="{BARRA_NO}"/>',
         f'<rect x="{xu:.1f}" y="60" width="{x1 - xu:.1f}" height="20" fill="{BARRA_SI}"/>',
         f'<text x="{x0 + 6}" y="74" fill="{TEXTO}" font-size="13">no predice pCR</text>',
         f'<text x="{x1 - 6}" y="74" fill="{TEXTO}" font-size="13" text-anchor="end">predice pCR</text>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        p.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="80" y2="85" stroke="{LINEA}"/>')
        p.append(f'<text x="{x(t):.1f}" y="140" fill="{TEXTO_2}" font-size="13" text-anchor="middle">'
                 f'{coma(t, 2).rstrip("0").rstrip(",") if t not in (0, 1) else int(t)}</text>')
    if tasa_base is not None:
        xb = x(tasa_base)
        p.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="54" y2="104" stroke="{TEXTO_2}" stroke-width="1.5" '
                 'stroke-dasharray="4 3"/>')
        p.append(f'<text x="{xb - 5:.1f}" y="118" fill="{TEXTO_2}" font-size="13" text-anchor="end">'
                 f'tasa base {por_ciento(tasa_base)}</text>')
    p.append(f'<line x1="{xu:.1f}" x2="{xu:.1f}" y1="54" y2="104" stroke="{TEXTO}" stroke-width="2"/>')
    p.append(f'<text x="{xu + 5:.1f}" y="118" fill="{TEXTO}" font-size="13">umbral {coma(umbral, 4)}</text>')
    for i, v in enumerate(por_modelo):
        if i == destacado:
            continue
        p.append(f'<circle cx="{x(v):.1f}" cy="94" r="3.5" fill="{TEXTO_2}" fill-opacity="0.85"/>')
    if destacado is not None:
        p.append(f'<circle cx="{x(por_modelo[destacado]):.1f}" cy="94" r="6" fill="none" stroke="{TEXTO}" '
                 'stroke-width="2"/>')
    p.append(f'<path d="M{xp - 8:.1f},40 L{xp + 8:.1f},40 L{xp:.1f},56 Z" fill="{TEXTO}"/>')
    xt = min(max(xp, x0 + 36), x1 - 36)
    p.append(f'<text x="{xt:.1f}" y="32" fill="{TEXTO}" font-size="13" font-weight="700" '
             f'text-anchor="middle">este corte</text>')
    descripcion = (f"Escala de probabilidad de 0 a 1. Este corte: {coma(prob, 4)}. "
                   f"Umbral: {coma(umbral, 4)}. "
                   + (f"Tasa base: {por_ciento(tasa_base)}. " if tasa_base is not None else "")
                   + "Modelos: " + "; ".join(coma(v, 3) for v in por_modelo) + ".")
    return (f'<svg viewBox="0 0 440 150" width="100%" role="img" aria-label="{descripcion}" '
            f'xmlns="http://www.w3.org/2000/svg" style="font-family:inherit">{"".join(p)}</svg>')


def svg_curva(curva: dict) -> str:
    """Curva de captacion de la zona que mas realza: PRE, EARLY y LATE (0-255), con el tipo
    escrito al final de la linea y el techo de 255 marcado."""
    medias, tipo = curva["medias"], curva["tipo"]
    y0, y1 = 22, 202

    def y(v):
        return y1 - (y1 - y0) * float(v) / 255

    xs = {"PRE": 78, "EARLY": 188, "LATE": 298}
    sub = {"PRE": "antes", "EARLY": "temprana", "LATE": "tardía"}
    p = []
    for t in (0, 64, 128, 192):
        p.append(f'<line x1="52" x2="320" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{LINEA}" '
                 'stroke-width="0.6"/>')
        p.append(f'<text x="44" y="{y(t) + 4:.1f}" fill="{TEXTO_2}" font-size="11" '
                 f'text-anchor="end">{t}</text>')
    p.append(f'<line x1="52" x2="320" y1="{y(255):.1f}" y2="{y(255):.1f}" stroke="{TEXTO_2}" '
             'stroke-dasharray="4 3"/>')
    p.append(f'<text x="44" y="{y(255) + 4:.1f}" fill="{TEXTO_2}" font-size="11" '
             'text-anchor="end">255</text>')
    p.append(f'<text x="326" y="{y(255) + 4:.1f}" fill="{TEXTO_2}" font-size="11">techo</text>')
    p.append(f'<text x="14" y="{(y0 + y1) / 2:.0f}" fill="{TEXTO_2}" font-size="11" text-anchor="middle" '
             f'transform="rotate(-90 14 {(y0 + y1) / 2:.0f})">intensidad media</text>')
    color = COLORES.get(tipo, TEXTO_2)
    puntos = " ".join(f"{xs[f]},{y(v):.1f}" for f, v in medias.items())
    trazo = "" if tipo else ' stroke-dasharray="6 4"'
    p.append(f'<polyline points="{puntos}" fill="none" stroke="{color}" stroke-width="3"{trazo}/>')
    for f, v in medias.items():
        p.append(f'<circle cx="{xs[f]}" cy="{y(v):.1f}" r="5" fill="{color}" stroke="#2B2D30" '
                 'stroke-width="1.5"/>')
        p.append(f'<text x="{xs[f]}" y="{min(y(v) + 20, y1 - 4):.1f}" fill="{TEXTO}" font-size="12" '
                 f'text-anchor="middle">{v:.0f}</text>')
        p.append(f'<text x="{xs[f]}" y="224" fill="{TEXTO}" font-size="13" font-weight="700" '
                 f'text-anchor="middle">{f}</text>')
        p.append(f'<text x="{xs[f]}" y="240" fill="{TEXTO_2}" font-size="11" '
                 f'text-anchor="middle">{sub[f]}</text>')
    yl = y(medias["LATE"])
    p.append(f'<text x="326" y="{yl + 18:.1f}" fill="{TEXTO}" font-size="13" font-weight="700">'
             f'{tipo or "sin tipo"}</text>')
    p.append(f'<text x="326" y="{yl + 34:.1f}" fill="{TEXTO_2}" font-size="12">'
             f'{"+" if curva["cambio"] >= 0 else "−"}{por_ciento(abs(curva["cambio"]), 0)}</text>')
    descripcion = (f"Curva de captación. PRE {medias['PRE']:.0f}, EARLY {medias['EARLY']:.0f}, LATE "
                   f"{medias['LATE']:.0f}. Cambio de EARLY a LATE {por_ciento(curva['cambio'], 0)}: "
                   f"{tipo or 'sin tipo'}.")
    return (f'<svg viewBox="0 0 420 250" width="100%" role="img" aria-label="{descripcion}" '
            f'xmlns="http://www.w3.org/2000/svg" style="font-family:inherit">{"".join(p)}</svg>')
