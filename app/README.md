App web del proyecto (Streamlit): recibe un corte (PRE, EARLY, LATE), lo pasa por el
conjunto de los 5 modelos y muestra la probabilidad de pCR y la clase con el umbral 0,3635.

    streamlit run app/app.py        # desde la raiz del repositorio

- `app.py`: solo la interfaz. La logica esta en `src/inferencia.py` (testeada en
  `tests/test_app.py`).
- `ejemplos/`: cuatro cortes de pacientes de train, con su atribucion y licencia.
- `requirements.txt`: dependencias del servidor (PyTorch de CPU).

Regla: la app NO preprocesa las imagenes por su cuenta. Lee cada PNG con
`src.data.imagen.leer_fase`, la misma funcion que el entrenamiento, y el resto del
preprocesado va dentro de la red (`src/models/cnn.py`). Asi las 5 muestras privadas de la
defensa se tratan exactamente igual que al entrenar. Detalle y despliegue: README de la
raiz, seccion "App".
