"""App web: probabilidad de pCR para un corte DCE (PRE, EARLY, LATE).

    streamlit run app/app.py

Solo interfaz: la logica (carga del modelo, validacion de la entrada y prediccion) esta
en `src/inferencia.py`, que se testea en tests/test_app.py. El preprocesado es el del
entrenamiento: `src.data.imagen` lee el PNG y el resto va dentro de la red.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                 # Streamlit solo añade app/ al path
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import inferencia  # noqa: E402
from src.data.imagen import FASES  # noqa: E402

EJEMPLOS = Path(__file__).resolve().parent / "ejemplos"

st.set_page_config(page_title="pCR desde RM DCE", layout="wide")


@st.cache_resource(show_spinner="Cargando los 5 modelos...")
def modelo() -> inferencia.Conjunto:
    return inferencia.cargar_conjunto()


@st.cache_data
def metricas() -> dict:
    return inferencia.metricas_internas()


def porcentaje(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def mapa_realce(x: np.ndarray) -> np.ndarray:
    """EARLY - PRE solo para VER (no entra en la prediccion), en gris de -0,1 a 0,4."""
    d = (x[1].astype(np.float32) - x[0].astype(np.float32)) / 255
    return np.clip((d + 0.1) / 0.5, 0, 1)


# --------------------------------------------------------------------------- #
st.title("Predicción de pCR desde una RM DCE previa al tratamiento")
st.error("**Uso educativo. Sin validez clínica.** No es un dispositivo médico: no sirve para "
         "diagnosticar ni para decidir tratamientos. Trabajo académico sobre el dataset BreastDCEDL.")

try:
    conjunto = modelo()
except inferencia.ErrorModelo as e:
    st.error(f"La app no puede arrancar: {e}")
    st.stop()

with st.sidebar:
    st.header("Entrada")
    origen = st.radio("Corte", ["Subir PNG", "Usar un ejemplo"], horizontal=True)
    st.caption("Tres PNG en escala de grises, 256×256, máximo 2 MB cada uno: las tres fases del "
               "contraste de un mismo corte. No se redimensiona ni se convierte nada más.")

datos: dict[str, bytes] = {}
nombres: dict[str, str] = {}
etiqueta_real = None
if origen == "Subir PNG":
    columnas = st.columns(3)
    for col, fase in zip(columnas, FASES):
        subido = col.file_uploader(fase, type=["png"], key=fase,
                                   help=f"Fichero de la fase {fase} (sufijo _{fase}.png en el dataset)")
        if subido is not None:
            datos[fase], nombres[fase] = subido.getvalue(), subido.name
else:
    tabla = pd.read_csv(EJEMPLOS / "ejemplos.csv")
    opciones = {f"{f.corte} ({f.cohorte}, pCR real = {f.pCR})": f for f in tabla.itertuples()}
    elegido = opciones[st.selectbox("Ejemplo (pacientes de train)", list(opciones))]
    datos, nombres = inferencia.corte_de_ejemplo(EJEMPLOS, elegido.corte)
    etiqueta_real = int(elegido.pCR)
    st.info("Los ejemplos son de **train**: 4 de los 5 modelos vieron a esa paciente al entrenar, así "
            "que su predicción es optimista. Sirven para probar la app, no para medirla.")

if len(datos) < 3:
    st.write("Sube las tres fases (PRE, EARLY y LATE) para obtener la predicción.")
else:
    fases, errores, avisos = {}, [], []
    for fase in FASES:
        try:
            fases[fase], avisos_fase = inferencia.validar_fase(datos[fase], fase)
            avisos += avisos_fase
        except inferencia.ErrorEntrada as e:
            errores.append(str(e))
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    x = np.stack([fases[f] for f in FASES])
    avisos += inferencia.avisos_corte(x, nombres)
    for a in avisos:
        st.warning(a)

    imagenes = st.columns(4)
    for col, fase in zip(imagenes, FASES):
        col.image(fases[fase], caption=f"{fase} ({nombres.get(fase, '')})", clamp=True, width="stretch")
    imagenes[3].image(mapa_realce(x), caption="EARLY − PRE (solo visualización)", clamp=True, width="stretch")

    try:
        r = inferencia.predecir(conjunto, x)
    except (inferencia.ErrorEntrada, inferencia.ErrorModelo) as e:
        st.error(str(e))
        st.stop()

    st.subheader("Resultado")
    a, b, c, d = st.columns(4)
    a.metric("Probabilidad de pCR", f"{r['prob']:.4f}")
    b.metric("Clase", r["clase"])
    c.metric("Umbral", f"{r['umbral']:.4f}")
    d.metric("Tiempo de respuesta", f"{r['segundos'] * 1000:.0f} ms", help=f"Dispositivo: {r['dispositivo']}")
    if etiqueta_real is not None:
        st.caption(f"Etiqueta real del ejemplo: pCR = {etiqueta_real}.")
    st.caption(f"Se predice pCR si la probabilidad supera el umbral. Dispositivo de inferencia: "
               f"**{r['dispositivo']}**. Probabilidad = media de las 5 sigmoides:")
    st.dataframe(pd.DataFrame({"modelo": list(r["por_modelo"]),
                               "probabilidad": [f"{v:.4f}" for v in r["por_modelo"].values()]}),
                 hide_index=True)

st.warning("**El umbral se eligió por paciente** (probabilidad media de sus ~10 cortes, especificidad "
           "≥ 0,90 fuera de fold), pero la app recibe **un solo corte**. Con un corte suelto el "
           "resultado es algo más ruidoso; abajo están las métricas por corte.")

# --------------------------------------------------------------------------- #
with st.expander("Modelo: pesos, SHA-256 y parámetros de entrenamiento"):
    n_parametros = f"{conjunto.parametros:,}".replace(",", ".")
    st.write(f"Conjunto de {len(conjunto.redes)} modelos (uno por fold), cada uno con {n_parametros} "
             "parámetros. Los SHA-256 coinciden con `results/test/plan.json` (comprobado al arrancar).")
    st.dataframe(pd.DataFrame([{"peso": f"models/{n}.pt", "sha256": conjunto.sha256[n],
                                "época elegida": conjunto.info[n].get("epoca"),
                                "fold": conjunto.info[n].get("entrenamiento", {}).get("fold")}
                               for n in conjunto.nombres]), hide_index=True)
    entrenamiento = dict(conjunto.info[conjunto.nombres[0]].get("entrenamiento", {}))
    entrenamiento.pop("fold", None)
    st.write("Arquitectura (`src/models/cnn.py`):", conjunto.config_red)
    st.write("Entrenamiento (igual en los 5; de `info` de los pesos):", entrenamiento)

m = metricas()
with st.expander("Métricas internas (leídas de results/)", expanded=False):
    v, u, t = m["validacion"], m["umbral_fuera_de_fold"], m["test"]
    if v:
        st.markdown(f"**Validación, 5 folds** (`results/comparacion/resumen.json`): AUC por paciente "
                    f"medio de todas las épocas {v['auc_medio_epocas']['media']:.3f} ± "
                    f"{v['auc_medio_epocas']['desviacion']:.3f}; en la época elegida "
                    f"{v['auc_epoca_elegida']['media']:.3f} ± {v['auc_epoca_elegida']['desviacion']:.3f} "
                    f"(inflado por elegirla con la propia validación); global fuera de fold "
                    f"{v['auc_global_fuera_de_fold']['auc']:.3f} "
                    f"[{v['auc_global_fuera_de_fold']['ic95'][0]:.3f}-{v['auc_global_fuera_de_fold']['ic95'][1]:.3f}].")
    else:
        st.write("Falta results/comparacion/resumen.json.")
    filas = []
    if u:
        filas += [{"conjunto": f"validación fuera de fold, por paciente ({u['pacientes']['n']})", "AUC": None,
                   **u["por_paciente"]},
                  {"conjunto": f"validación fuera de fold, por corte ({u['por_corte']['n']})", "AUC": None,
                   **{k: w for k, w in u["por_corte"].items() if k != "n"}}]
    if t:
        pp, pc = t["por_paciente"], t["por_corte"]
        filas += [{"conjunto": f"test, por paciente ({t['pacientes']['n']})", "AUC": pp["auc"]["auc"],
                   **{k: w for k, w in pp.items() if k != "auc"}},
                  {"conjunto": f"test, por corte ({t['pacientes']['cortes']})", "AUC": pc["auc"]["auc"],
                   **{k: w for k, w in pc.items() if k != "auc"}}]
    if filas:
        st.markdown(f"**Con el umbral {conjunto.umbral:.4f}** (`results/umbral/resumen.json`, "
                    "`results/test/resumen.json`; test evaluado una sola vez):")
        st.dataframe(pd.DataFrame(filas).set_index("conjunto").map(porcentaje))
    else:
        st.write("Faltan los JSON del umbral y de test.")

st.caption("Imágenes derivadas de BreastDCEDL (Fridman et al., 2025, arXiv:2506.12190), licencia "
           "CC BY-NC 4.0. CNN propia entrenada desde cero; código en el repositorio del proyecto.")
