Demo web del proyecto (pendiente): carga un corte (PRE, EARLY, LATE), lo pasa por
la red y muestra la probabilidad de pCR. Se desplegara con URL publica
(Streamlit / Hugging Face Spaces), como pide el caso.

Regla: la app NO preprocesa las imagenes por su cuenta. Lee el corte con
`utils_caso.cargar_imagen` y lo pasa a la red cargada con `src.models.cnn.cargar`,
que ya lleva dentro el preprocesado del entrenamiento. Asi las 5 muestras privadas
de la defensa se tratan exactamente igual que al entrenar.
