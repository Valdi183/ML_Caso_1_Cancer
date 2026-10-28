App web del proyecto (Streamlit): recibe un corte (PRE, EARLY, LATE), lo pasa por el
conjunto de los 5 modelos y muestra la probabilidad de pCR y la clase con el umbral 0,3635.

    streamlit run app/app.py        # desde la raiz del repositorio

- `app.py`: solo la interfaz. La logica de la prediccion esta en `src/inferencia.py`
  (testeada en `tests/test_app.py`). Lo que se enseña alrededor de la prediccion esta en
  `src/visualizacion.py` (testeado en `tests/test_visualizacion.py`): el mapa y la curva de
  captacion, la escala del resultado, las cifras de "¿Cuanto fiarse?" (leidas de
  `results/umbral/` y `results/test/`), los mensajes de error y el JSON descargable.
- `ejemplos/`: cuatro cortes de pacientes de train, con su atribucion y licencia.
- `requirements.txt`: dependencias del servidor (PyTorch de CPU).

Regla: la app NO preprocesa las imagenes por su cuenta. Lee cada PNG con
`src.data.imagen.leer_fase`, la misma funcion que el entrenamiento, y el resto del
preprocesado va dentro de la red (`src/models/cnn.py`). Asi las 5 muestras privadas de la
defensa se tratan exactamente igual que al entrenar. Detalle y despliegue: README de la
raiz, seccion "App".

El mapa y la curva son descriptivos: el modelo no los usa. El tipo de curva sale del cambio
de EARLY a LATE: mas de +10 %, persistente; ±10 %, meseta; menos de -10 %, lavado. Donde
EARLY o LATE saturan, solo se afirma el tipo que la cota permite. El momento de LATE varia
segun el hospital.

Si existe `results/combinado/resumen.json`, la app muestra el AUC de validacion de la
clinica sola, la imagen sola y las dos combinadas, solo como comparacion: no pide datos
clinicos. Si existe `docs/figuras/arquitectura.png`, muestra el esquema de la red.

El tema (gris de visor y Atkinson Hyperlegible desde Google Fonts) esta en
`.streamlit/config.toml`. La app escribe la probabilidad con coma decimal;
`python -m src.inferencia` la imprime con punto, con los mismos digitos.
