"""App web: probabilidad de pCR para un corte DCE (PRE, EARLY, LATE).

    streamlit run app/app.py

Solo interfaz: la prediccion (carga del modelo, validacion de la entrada y prediccion)
esta en `src/inferencia.py`, y lo que se ensena alrededor (mapa y curva de captacion,
escala del resultado, cifras de confianza) en `src/visualizacion.py`; los dos se
testean (tests/test_app.py, tests/test_visualizacion.py). El preprocesado es el del
entrenamiento: `src.data.imagen` lee el PNG y el resto va dentro de la red. El mapa y la
curva son descriptivos: no entran en la prediccion.
"""

from __future__ import annotations

import base64
import html
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                 # Streamlit solo añade app/ al path
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src import config, inferencia  # noqa: E402
from src import visualizacion as vis  # noqa: E402
from src.data.imagen import FASES  # noqa: E402

EJEMPLOS = Path(__file__).resolve().parent / "ejemplos"
COHORTES = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}
SUBTITULO = {"PRE": "antes del contraste", "EARLY": "fase temprana", "LATE": "fase tardía"}

st.set_page_config(page_title="pCR desde RM DCE", layout="wide")

# Solo lo que el tema (.streamlit/config.toml) no cubre: las piezas en HTML propias
# (visor, escala, banda de aviso, pasos) y un foco de teclado bien visible.
st.html("""<style>
:focus-visible { outline: 2px solid #E3E5E8 !important; outline-offset: 2px; }
.aviso { background: #C8A24A; color: #1E1F21; padding: .7rem 1rem; border-radius: 6px; margin: 0 0 1rem; }
.pasos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: .25rem 0 .5rem; }
.pasos div { border-top: 2px solid #5C6066; padding-top: .5rem; }
.pasos b { display: block; font-size: 1.05rem; }
.pasos span, .secundario { color: #A0A6AD; }
.visor { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; background: #1F2124;
         border: 1px solid #1F2124; border-radius: 6px; overflow: hidden; }
.visor figure { margin: 0; background: #34373B; }
.visor .rotulo { padding: .4rem .6rem; font-size: .9rem; }
.visor .rotulo span { color: #A0A6AD; }
.visor img { display: block; width: 100%; aspect-ratio: 1; }
.visor .vacio { aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
                color: #A0A6AD; padding: 1rem; text-align: center; }
.visor figcaption { padding: .3rem .6rem .5rem; font-size: .8rem; color: #A0A6AD; overflow-wrap: anywhere; }
.leyenda { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin: .6rem 0 .2rem; font-size: .9rem; }
.leyenda i { display: inline-block; width: .9rem; height: .9rem; border-radius: 2px; margin-right: .35rem;
             vertical-align: -2px; }
.prob { font-size: 3.4rem; font-weight: 700; line-height: 1.05; }
.frase { font-size: 1.15rem; margin: .3rem 0 .2rem; }
.fiarse { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; }
.fiarse b.cifra { display: block; font-size: 1.8rem; }
.barra { display: grid; grid-template-columns: minmax(10rem, 22rem) 1fr 7.5rem; gap: .8rem;
         align-items: center; margin: .35rem 0; }
.barra div { background: #3D4045; height: .8rem; border-radius: 3px; }
.barra i { display: block; height: 100%; background: #A0A6AD; border-radius: 3px; }
.nota-barra { margin: -.2rem 0 .5rem; font-size: .85rem; }
@media (max-width: 640px) { .barra { grid-template-columns: 1fr 6.5rem; } .barra div { display: none; } }
@media (max-width: 800px) { .visor, .pasos, .fiarse { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .pasos, .fiarse { grid-template-columns: 1fr; } }
</style>""")


@st.cache_resource(show_spinner="Cargando los 5 modelos...")
def modelo() -> inferencia.Conjunto:
    return inferencia.cargar_conjunto()


@st.cache_data
def metricas() -> dict:
    return inferencia.metricas_internas()


@st.cache_data
def cifras() -> dict:
    return vis.cifras_confianza()


@st.cache_data
def tabla_ejemplos() -> pd.DataFrame:
    return pd.read_csv(EJEMPLOS / "ejemplos.csv")


def png_uri(x: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(x).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def tres(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else vis.coma(v, 3)


def visor(paneles: list[tuple[str, str, np.ndarray | None, str]]) -> str:
    """Tira continua de paneles: (fase, subtitulo, imagen o None, pie)."""
    figuras = []
    for fase, subtitulo, img, pie in paneles:
        cuerpo = (f'<img src="{png_uri(img)}" alt="{html.escape(fase)}: {html.escape(subtitulo)}">'
                  if img is not None else '<div class="vacio">Sin imagen</div>')
        figuras.append(f'<figure><div class="rotulo"><b>{fase}</b> <span>{subtitulo}</span></div>'
                       f'{cuerpo}<figcaption>{html.escape(pie)}</figcaption></figure>')
    return f'<div class="visor" role="group" aria-label="Visor del corte">{"".join(figuras)}</div>'


# --------------------------------------------------------------------------- #
# Cabecera
# --------------------------------------------------------------------------- #
st.title("Predicción de pCR desde una RM DCE previa al tratamiento")
st.html('<div class="aviso" role="note"><b>Uso educativo. Sin validez clínica.</b> No es un dispositivo '
        'médico: no sirve para diagnosticar ni para decidir tratamientos. Es un trabajo académico sobre '
        'el dataset BreastDCEDL.</div>')
st.html('<p>La <b>pCR</b> (respuesta patológica completa) es que, tras la quimioterapia, no quede tumor '
        'invasivo. Esta app estima su probabilidad con una resonancia tomada <b>antes</b> de empezar.</p>'
        '<div class="pasos">'
        '<div><b>1. Sube las imágenes</b><span>En la barra lateral: los tres PNG de un corte (PRE, '
        'EARLY y LATE) o un ejemplo.</span></div>'
        '<div><b>2. Míralas</b><span>Las tres fases del contraste y un mapa de cómo capta cada zona.'
        '</span></div>'
        '<div><b>3. Interpreta el resultado</b><span>La probabilidad frente al umbral, y cuánto '
        'fiarse de ella.</span></div></div>')

try:
    conjunto = modelo()
except inferencia.ErrorModelo as e:
    st.error(f"La app no puede arrancar: {e}")
    st.stop()

fold_de = {conjunto.info[n]["entrenamiento"]["fold"]: i for i, n in enumerate(conjunto.nombres)}

# --------------------------------------------------------------------------- #
# Entrada (barra lateral)
# --------------------------------------------------------------------------- #
datos: dict[str, bytes] = {}
nombres: dict[str, str] = {}
ejemplo = None
with st.sidebar:
    st.header("Entrada")
    origen = st.radio("¿Qué quieres analizar?", ["Mis imágenes", "Un ejemplo"], horizontal=True)
    if origen == "Mis imágenes":
        st.caption("Tres PNG en escala de grises, de 256×256 px y hasta 2 MB cada uno: las tres fases "
                   "de un mismo corte. La app no redimensiona ni convierte nada.")
        for fase in FASES:
            subido = st.file_uploader(f"{fase} · {SUBTITULO[fase]}", type=["png"], key=fase,
                                      help=f"En el dataset, el fichero que acaba en _{fase}.png")
            if subido is not None:
                datos[fase], nombres[fase] = subido.getvalue(), subido.name
    else:
        tabla = tabla_ejemplos()
        st.caption("Pacientes de **train**: 4 de los 5 modelos las vieron al entrenar, así que su "
                   "resultado es optimista. Sirven para probar la app, no para medirla.")
        elegido = st.session_state.setdefault("ejemplo", tabla.corte[0])
        columnas = st.columns(2)
        for i, f in enumerate(tabla.itertuples()):
            with columnas[i % 2]:
                st.image(str(EJEMPLOS / f"{f.corte}_EARLY.png"), width="stretch")
                st.caption(f"**{i + 1}.** {COHORTES.get(f.cohorte, f.cohorte)} · pCR real: "
                           f"{'sí' if f.pCR else 'no'}" + (" · **viendo**" if f.corte == elegido else ""))
                if st.button(f"Ver ejemplo {i + 1}", key=f"ej{i}", width="stretch"):
                    st.session_state.ejemplo = elegido = f.corte
                    st.rerun()
        ejemplo = tabla.set_index("corte").loc[elegido]
        datos, nombres = inferencia.corte_de_ejemplo(EJEMPLOS, elegido)

# --------------------------------------------------------------------------- #
# Visor, curva y resultado
# --------------------------------------------------------------------------- #
st.subheader("Visor")
x = None
if len(datos) < 3:
    faltan = [f for f in FASES if f not in datos]
    st.html(visor([(f, SUBTITULO[f], None, "pendiente" if f in faltan else nombres[f]) for f in FASES]
                  + [("Captación", "mapa descriptivo", None, "")]))
    lista = faltan[0] if len(faltan) == 1 else f"{', '.join(faltan[:-1])} y {faltan[-1]}"
    st.html(f'<p class="secundario">{"Falta" if len(faltan) == 1 else "Faltan"} {lista}. '
            f'{"Súbelo" if len(faltan) == 1 else "Súbelos"} en la barra lateral o elige un ejemplo.</p>')
else:
    fases, errores, avisos = {}, [], []
    for fase in FASES:
        try:
            fases[fase], avisos_fase = inferencia.validar_fase(datos[fase], fase)
            avisos += avisos_fase
        except inferencia.ErrorEntrada as e:
            errores.append(vis.mensaje_error(str(e)))
    for e in errores:
        st.error(e)
    if not errores:
        x = np.stack([fases[f] for f in FASES])
        avisos += inferencia.avisos_corte(x, nombres)

if x is not None:
    for a in avisos:
        st.warning(a)
    mapa, curva = vis.mapa_captacion(x), vis.curva_captacion(x)
    st.html(visor([(f, SUBTITULO[f], fases[f], nombres.get(f, "")) for f in FASES]
                  + [("Captación", "sobre EARLY", mapa["rgb"], "descriptivo: el modelo no lo usa")]))
    leyenda = "".join(f'<span><i style="background:{vis.COLORES[t]}"></i><b>{t}</b>: {vis.DESCRIPCION[t]} '
                      f'({vis.por_ciento(mapa["fracciones"][t], 0)})</span>' for t in vis.TIPOS)
    st.html(f'<div class="leyenda">{leyenda}<span><i style="background:#FFFFFF"></i><b>blanco sin '
            f'color</b>: satura (255) y su tipo no se puede medir</span></div>'
            f'<p class="secundario">Mapa de captación: el {100 - vis.PERCENTIL_REALCE} % de la mama que más '
            f'realza de PRE a EARLY, coloreado por cómo cambia de EARLY a LATE (suavizado para verlo). '
            f'<b>Es descriptivo: el modelo no lo usa.</b> El momento en que se toma LATE cambia según el '
            f'hospital, así que el tipo no es comparable entre cohortes.</p>')

    try:
        r = inferencia.predecir(conjunto, x)
    except (inferencia.ErrorEntrada, inferencia.ErrorModelo) as e:
        st.error(str(e))
        st.stop()

    izquierda, derecha = st.columns([1, 1.25], gap="large")
    with izquierda:
        st.subheader("Curva de captación")
        st.markdown(vis.svg_curva(curva), unsafe_allow_html=True)
        sat = curva["saturada_early"]
        nota = (f" EARLY satura en el {vis.por_ciento(sat, 0)} de esa zona: el cambio medido es una cota, "
                "y solo se afirma un tipo si la cota basta." if sat > 0.1 else "")
        st.html(f'<p class="secundario">Intensidad media del {vis.por_ciento(vis.FRACCION_ZONA, 0)} de '
                f'píxeles de la mama que más realza de PRE a EARLY ({curva["n_pixeles"]} px). '
                f'Tipo por el cambio de EARLY a LATE: más de +10 % persistente, ±10 % meseta, menos de '
                f'−10 % lavado.{nota} <b>Descriptivo: el modelo no lo usa</b>, y el momento de LATE '
                f'cambia según el hospital.</p>')

    with derecha:
        st.subheader("Resultado")
        prob, umbral = r["prob"], r["umbral"]
        destacado = fold_de.get(int(ejemplo["fold"])) if ejemplo is not None else None
        if prob > umbral:
            frase = "Está por encima del umbral: el modelo predice pCR."
        else:
            frase = "Está por debajo del umbral: el modelo no predice pCR."
        cerca = (" Está muy cerca del umbral: otro corte de la misma paciente podría dar la otra clase."
                 if abs(prob - umbral) < 0.02 else "")
        st.html(f'<div class="secundario">Probabilidad de pCR de este corte</div>'
                f'<div class="prob">{inferencia.texto_probabilidad(prob)}</div>'
                f'<p class="frase"><b>{frase}</b>{cerca}</p>')
        st.markdown(vis.svg_escala(prob, umbral, cifras()["tasa_base"], list(r["por_modelo"].values()),
                                destacado), unsafe_allow_html=True)
        st.html('<p class="secundario">▼ este corte (media de los 5 modelos) · ● cada modelo · │ umbral: '
                'por encima se predice pCR · ┆ tasa base: proporción de pacientes con pCR en train'
                + (' · ○ el modelo que no vio a esta paciente' if destacado is not None else '') + '</p>')
        if ejemplo is not None:
            nombre = conjunto.nombres[destacado]
            st.html(f'<p><b>Resultado real: {"pCR" if ejemplo["pCR"] else "no pCR"}</b> '
                    f'({COHORTES.get(ejemplo["cohorte"], ejemplo["cohorte"])}). La paciente es de train: '
                    f'solo el modelo de su fold ({int(ejemplo["fold"])}, <code>{nombre}</code>) no la '
                    f'vio, y le '
                    f'da {vis.coma(r["por_modelo"][nombre], 4)}.</p>')
        st.html(f'<p class="secundario">Umbral {vis.coma(umbral, 4)} · modelo '
                f'<code>{vis.huella_conjunto(conjunto.sha256)}</code> (SHA-256 de los 5 pesos) · '
                f'{r["dispositivo"].upper()} · {r["segundos"] * 1000:.0f} ms</p>')
        st.download_button("Descargar el resultado (JSON)", vis.resultado_json(r, conjunto.sha256, nombres),
                           file_name="resultado_pcr.json", mime="application/json")

# --------------------------------------------------------------------------- #
# ¿Cuánto fiarse?
# --------------------------------------------------------------------------- #
st.subheader("¿Cuánto fiarse?")
c = cifras()
v, t = c["validacion"], c["test"]
if t and v:
    st.html(
        '<div class="fiarse">'
        f'<div><b class="cifra">{vis.por_ciento(t["auc"], 0)}</b>Si se toma una paciente que alcanzó pCR y '
        f'otra que no, el modelo da más probabilidad a la primera el {vis.por_ciento(t["auc"], 0)} de las '
        f'veces (test, {t["pacientes"]} pacientes; margen {vis.por_ciento(t["ic95"][0], 0)} a '
        f'{vis.por_ciento(t["ic95"][1], 0)}). <b>Al azar sería el 50 %.</b></div>'
        f'<div><b class="cifra">{vis.de_cada_10(v["VPP"])} de cada 10</b>'
        'De las pacientes que marca como pCR, '
        f'unas {vis.de_cada_10(v["VPP"])} de cada 10 lo son en validación ({v["VP"]} de {v["marcadas"]}) y '
        f'{vis.de_cada_10(t["VPP"])} de cada 10 en test ({t["VP"]} de {t["marcadas"]}). Sin modelo, '
        f'{vis.de_cada_10(c["tasa_base"])} de cada 10 alcanzan pCR.</div>'
        f'<div><b class="cifra">{t["VP"]} de {t["pcr"]}</b>Casi nunca avisa: en test marcó {t["VP"]} de las '
        f'{t["pcr"]} pacientes que sí alcanzaron pCR. El umbral es exigente a propósito: predecir pCR en '
        f'quien no la alcanzará es el error más grave.</div></div>')
    st.html('<p><b>No sirve para ninguna decisión clínica.</b> En test apenas se distingue del azar. Además, '
            'el umbral se fijó por paciente (media de sus ~10 cortes) y aquí se mira un solo corte, así '
            'que el resultado es algo más ruidoso.</p>')
else:
    st.warning("Faltan results/umbral/resumen.json o results/test/resumen.json: sin ellos la app no "
               "puede decir cuánto fiarse.")

# Imagen frente a variables clinicas: solo si la comparacion existe en results/combinado/
comparacion = vis.imagen_frente_a_clinica()
if comparacion:
    st.subheader("Imagen frente a variables clínicas")
    filas_html = []
    for clave, nombre in (("clinica", "Variables clínicas solas (subtipo, edad y volumen)"),
                          ("imagen", "Imagen sola (esta red)"), ("combinada", "Las dos combinadas")):
        w = comparacion[clave]
        ancho = max(0.0, min(1.0, (w["media"] - 0.5) / 0.3)) * 100          # escala de 0,5 a 0,8
        filas_html.append(f'<div class="barra"><span>{nombre}</span><div><i style="width:{ancho:.0f}%"></i>'
                          f'</div><b>{vis.coma(w["media"], 3)} ± {vis.coma(w["desviacion"], 3)}</b></div>')
        if clave == "imagen":
            filas_html.append('<p class="secundario nota-barra">optimista: época elegida con la propia '
                              'validación; sin elegir, '
                              f'{vis.coma(comparacion["imagen_todas_las_epocas"], 3)}</p>')
    ic = comparacion["mejora_global_ic95"]
    en_test = comparacion["imagen_test"]
    st.html("".join(filas_html)
            + f'<p class="secundario">AUC por paciente en validación, media ± desviación de los 5 folds '
              f'({comparacion["pacientes"]} pacientes; la barra empieza en 0,5, el azar).</p>'
            + (f'<p>Solo el modelo de imagen se evaluó en test (AUC {vis.coma(en_test, 3)}).</p>'
               if en_test is not None else "")
            + '<p><b>Lectura:</b> las variables clínicas ordenan a las pacientes bastante mejor que la '
              f'imagen. Añadir la imagen sube el AUC en {vis.coma(comparacion["mejora"], 3)} (mejora en '
              f'{comparacion["folds_a_favor"]} de 5 folds), pero el margen de esa mejora '
              f'({vis.coma(ic[0], 3)} a {vis.coma(ic[1], 3)}) incluye el cero. Es solo una comparación: la '
              f'app no pide datos clínicos ni los usa.</p>')

# --------------------------------------------------------------------------- #
# Desplegables
# --------------------------------------------------------------------------- #
n_parametros = f"{conjunto.parametros:,}".replace(",", ".")
with st.expander("Cómo funciona"):
    st.markdown(
        "**Las fases.** Una resonancia con contraste toma la misma imagen varias veces: **PRE** antes de "
        "inyectar el contraste, **EARLY** poco después y **LATE** unos minutos más tarde. El tumor suele "
        "captar el contraste antes y con más fuerza que el tejido sano. Las tres son el mismo corte en "
        "tres momentos, no los colores de una foto.\n\n"
        f"**La red.** Una red convolucional propia, entrenada desde cero (nada preentrenado), con tres "
        f"bloques de 8, 16 y 32 filtros y {n_parametros} parámetros. Recibe las tres fases juntas, reduce la "
        "imagen a 128 px y resume el corte con la media de lo que detecta.\n\n"
        f"**El conjunto.** Se entrenaron {len(conjunto.redes)} redes iguales, cada una sin ver una quinta "
        "parte de las pacientes. La probabilidad es la media de las cinco.\n\n"
        f"**El umbral ({vis.coma(conjunto.umbral, 4)}).** Se fijó en validación para que, de las pacientes "
        "sin pCR, como mucho 1 de cada 10 se marque como pCR (especificidad ≥ 0,90). El error grave es "
        "predecir pCR en quien no la alcanzará: podría llevar a quitarle un tratamiento que sí necesita.")
    arquitectura = sorted(config.FIGURAS.glob("*arquitectura*.png"))
    if arquitectura:
        st.image(str(arquitectura[0]), caption="Arquitectura de la red", width="stretch")

m = metricas()
with st.expander("Cómo se validó"):
    val = m["validacion"]
    texto = ("**Separación por paciente.** Las 1.097 pacientes de train se repartieron en 5 grupos "
             "(folds) con la columna `fold` del dataset: todos los cortes de una paciente van al mismo "
             "grupo, así que ningún modelo se valida con una paciente que vio al entrenar.\n\n")
    if val:
        cohortes = ", ".join(f"{COHORTES.get(k, k)} {vis.coma(w['auc'], 3)} ({w['pacientes']} pacientes)"
                             for k, w in val["por_cohorte"].items() if isinstance(w, dict) and "auc" in w)
        texto += ("**Validación (5 folds).** AUC por paciente "
                  f"{vis.coma(val['auc_medio_epocas']['media'], 3)}"
                  f" ± {vis.coma(val['auc_medio_epocas']['desviacion'], 3)} (media de todas las épocas, la "
                  "estimación sin sesgo); en la época elegida "
                  f"{vis.coma(val['auc_epoca_elegida']['media'], 3)}, "
                  "algo inflado porque esa época se eligió con la propia validación."
                  + (f" Por hospital, con todas las pacientes fuera de fold juntas: {cohortes}."
                     if cohortes else "") + "\n\n")
    if t:
        texto += (f"**Test, una sola vez.** Con todo decidido antes (modelo, umbral y SHA-256 de cada peso "
                  f"en `results/test/plan.json`), {t['pacientes']} pacientes nunca vistas: AUC "
                  f"{vis.coma(t['auc'], 3)} [{vis.coma(t['ic95'][0], 3)}-{vis.coma(t['ic95'][1], 3)}]. "
                  "Coincide con la validación y queda por debajo de la meta del caso (0,64).")
    st.markdown(texto)
    roc = config.FIGURAS / "test_roc.png"
    if roc.exists():
        st.image(str(roc), caption="Curva ROC en test, con el punto del umbral", width=520)

with st.expander("Detalles técnicos"):
    st.write(f"Conjunto de {len(conjunto.redes)} modelos (uno por fold), cada uno con {n_parametros} "
             "parámetros, en modo evaluación y sin gradientes. Los SHA-256 coinciden con "
             "`results/test/plan.json` (comprobado al arrancar).")
    st.dataframe(pd.DataFrame([{"peso": f"models/{n}.pt", "sha256": conjunto.sha256[n],
                                "época elegida": conjunto.info[n].get("epoca"),
                                "fold": conjunto.info[n].get("entrenamiento", {}).get("fold")}
                               for n in conjunto.nombres]), hide_index=True)
    if x is not None:
        st.write(f"Dispositivo de inferencia: **{r['dispositivo']}**. Tiempo de respuesta: "
                 f"{r['segundos'] * 1000:.0f} ms. Probabilidad sin redondear: {r['prob']!r}.")
        st.dataframe(pd.DataFrame({"modelo": list(r["por_modelo"]),
                                   "probabilidad": [f"{p:.4f}" for p in r["por_modelo"].values()]}),
                     hide_index=True)
    entrenamiento = dict(conjunto.info[conjunto.nombres[0]].get("entrenamiento", {}))
    entrenamiento.pop("fold", None)
    st.write("Arquitectura (`src/models/cnn.py`):", conjunto.config_red)
    st.write("Entrenamiento (igual en los 5; de `info` de los pesos):", entrenamiento)

    u, mt = m["umbral_fuera_de_fold"], m["test"]
    if val:
        st.markdown(f"**Validación, 5 folds** (`results/comparacion/resumen.json`): AUC por paciente "
                    f"medio de todas las épocas {val['auc_medio_epocas']['media']:.3f} ± "
                    f"{val['auc_medio_epocas']['desviacion']:.3f}; en la época elegida "
                    f"{val['auc_epoca_elegida']['media']:.3f} ± "
                    f"{val['auc_epoca_elegida']['desviacion']:.3f}; "
                    f"global fuera de fold {val['auc_global_fuera_de_fold']['auc']:.3f} "
                    f"[{val['auc_global_fuera_de_fold']['ic95'][0]:.3f}-"
                    f"{val['auc_global_fuera_de_fold']['ic95'][1]:.3f}].")
    filas = []
    if u:
        filas += [{"conjunto": f"validación fuera de fold, por paciente ({u['pacientes']['n']})", "AUC": None,
                   **u["por_paciente"]},
                  {"conjunto": f"validación fuera de fold, por corte ({u['por_corte']['n']})", "AUC": None,
                   **{k: w for k, w in u["por_corte"].items() if k != "n"}}]
    if mt:
        pp, pc = mt["por_paciente"], mt["por_corte"]
        filas += [{"conjunto": f"test, por paciente ({mt['pacientes']['n']})", "AUC": pp["auc"]["auc"],
                   **{k: w for k, w in pp.items() if k != "auc"}},
                  {"conjunto": f"test, por corte ({mt['pacientes']['cortes']})", "AUC": pc["auc"]["auc"],
                   **{k: w for k, w in pc.items() if k != "auc"}}]
    if filas:
        st.markdown(f"**Métricas internas con el umbral {conjunto.umbral:.4f}** "
                    "(`results/umbral/resumen.json`, `results/test/resumen.json`; test evaluado una sola "
                    "vez):")
        st.dataframe(pd.DataFrame(filas).set_index("conjunto").map(tres))

st.caption("Imágenes derivadas de BreastDCEDL (Fridman et al., 2025, arXiv:2506.12190), licencia "
           "CC BY-NC 4.0. CNN propia entrenada desde cero; código en el repositorio del proyecto.")
