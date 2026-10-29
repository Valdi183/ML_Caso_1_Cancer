# Cortes de ejemplo de la app

Cuatro cortes (PRE, EARLY, LATE) de **cuatro pacientes de train** del caso BreastDCEDL,
copiados sin modificar desde `dataset/train/<paciente>/`. Ninguno es de test ni de la
validación privada. `ejemplos.csv` indica la paciente, la cohorte, la etiqueta pCR y el
fold de validación de cada uno.

Como son de train, 4 de los 5 modelos del conjunto vieron a esa paciente al entrenar: su
predicción es optimista y solo sirve para probar la app. El modelo cuyo fold de validación
es el de la paciente no la vio; `tests/test_app.py` comprueba que su probabilidad en la app
coincide con la que guardó el entrenamiento (`results/<ejecución>/predicciones_val.csv`).

## Atribución y licencia

Imágenes desidentificadas derivadas de **BreastDCEDL** (Fridman et al., 2025,
arXiv:2506.12190), construido a partir de las colecciones públicas I-SPY1, I-SPY2 y Duke
Breast Cancer MRI. Licencia **CC BY-NC 4.0**
(<https://creativecommons.org/licenses/by-nc/4.0/>): uso docente y de investigación, sin
fines comerciales, conservando esta atribución. No se permite reidentificación,
diagnóstico ni recomendación terapéutica.
