# Decisiones del modelo y su justificación

Complementa a [`iteraciones.md`](iteraciones.md) (hipótesis, resultado y decisión de
cada iteración). Aquí está lo que condiciona el trabajo entero: qué pide el caso, qué
modelo se entrega y por qué, y los diagnósticos anteriores a la iteración 1.

## Qué pide el caso (indicaciones del profesor, correo del 14-10-2026)

- **CNN propia desde cero que predice pCR con las 3 fases** (PRE, EARLY, LATE: tres
  instantes del contraste, no RGB). Evaluación por paciente (media de sus cortes),
  umbral elegido en validación, test una sola vez.
- Las variables de `patients.csv` son **comparación, no modelo principal**: en la
  clínica no se tienen todas y la app solo recibe imágenes.
- La app recibe **una muestra (un corte) por paciente** (guía del caso, C7): nada de
  2,5D ni de predicción que necesite varios cortes de la misma paciente.
- Referencia: imagen pretratamiento con test independiente publica AUC 0,70-0,80; con
  cortes 2D y 3 cohortes, **0,60-0,70 en test es buen trabajo**. El IC del AUC en test
  es de ±0,08.
- Rúbrica: EDA; partición por paciente con **5 folds, media ± desviación**;
  arquitectura justificada (fases no como RGB); reproducible (semilla, parada, época
  elegida); **curvas de AUC de train y de validación** y su diferencia en la época
  elegida; **pérdida normal frente a ponderada** y su efecto en la sensibilidad; umbral
  y agregación elegidos en validación con **criterio clínico explícito** (qué error es
  peor); test una vez con matriz de confusión, AUC con IC, precisión/recall/F1 por
  clase, F1 macro y comparación con «siempre no pCR»; resultados por cohorte,
  calibración, casos fallados con imágenes; bibliografía, limitaciones y app con el
  mismo preprocesado.

## Modelo final

**Configuración** (de `it6_3bloques`, congelada en la iteración 7):
`--aumentado --dropout 0.7 --wd 1e-3 --reducir 2 --filtros 8,16,32`, pérdida normal,
semilla 42 y el resto de `config.DEFECTO`. Tres bloques convolucionales (8-16-32),
resumen por media, imagen reducida a 128 px dentro de la red, 18.329 parámetros. El
preprocesado va dentro de la red (`src/models/cnn.py`), así que la app no puede
preprocesar distinto que el entrenamiento.

**Ejecuciones:** `it6_3bloques_f0`, `it6_3bloques_f1`, `it8_3bloques_f2`,
`it8_3bloques_f3`, `it8_3bloques_f4` (`results/` y `models/`).

**Validación (5 folds):** AUC medio de todas las épocas 0,551 ± 0,011; AUC de la
época elegida 0,589 ± 0,020 (inflado por elegir la época con la propia validación).
Por cohorte, en la época elegida: spy2 0,586 ± 0,020, spy1 0,576 ± 0,098, Duke
0,531 ± 0,077.

**Por qué este y no otro:** desde la iteración 2 todo lo probado (capacidad,
resolución, realce, normalización, resumen, atención libre y supervisada con la caja
del tumor) queda en un AUC medio de ~0,545, con diferencias del tamaño del ruido. Este
es el más pequeño, el que menos memoriza y el de menor desviación entre folds.

**Pérdida normal frente a ponderada:** empatan en AUC. Con `pos_weight = N0/N1 = 2,4`
la ponderación suma ln 2,4 al logit y lleva la tasa base a 0,5: desplaza las
probabilidades sin cambiar el orden. Por eso a 0,5 cambia la sensibilidad (normal
0,01; ponderada 0,42 ± 0,26) y no el AUC. Se queda la **normal**, cuyas
probabilidades están calibradas, y el punto de corte lo da el umbral.

**Umbral (criterio fijado; valor pendiente):** por paciente, agregando por la media
de sus cortes, el umbral que da **especificidad ≥ 0,90** sobre las predicciones fuera
de fold de los 5 folds. Criterio clínico: el error grave es el falso positivo
(predecir pCR en quien no la alcanzará podría llevar a desescalar un tratamiento que
sí necesita).

**Predicción final:** el conjunto de los 5 modelos (media de sus probabilidades).
Ninguna validación lo ve entero (cada modelo ha visto en train los otros 4 folds),
así que **solo se puede evaluar en test**.

**Avisos para el informe:**
- El umbral se elige por paciente (media de ~10 cortes) pero la app recibe un solo
  corte: la probabilidad de un corte es más ruidosa y el umbral no se transfiere
  exactamente.
- El conjunto solo se evalúa en test, y el umbral sale de las predicciones de modelos
  individuales, no del conjunto.
- Duke es el 24 % del test (42 de 176) y su AUC en validación ronda 0,5: bajará el
  AUC global.

## Diagnósticos anteriores a la iteración 1

- **EDA** (`notebooks/eda.ipynb`): variables clínicas (subtipo + edad + volumen) dan
  AUC 0,69; solo la cohorte, 0,55; medidas globales de realce, ~0,5. Cortes de una
  misma paciente muy correlacionados (0,81). LATE−EARLY depende del protocolo de cada
  cohorte. Las imágenes son la mama entera con orientación variable.
- **Primer entrenamiento** (`results/ponderada_f0`, red base de 4 bloques 16-32-64-128,
  294.129 parámetros, pérdida ponderada): AUC por paciente 0,58 en la época 1, parada
  en la 9; la pérdida de validación nunca baja de la de predecir siempre 0,5 (0,979) y
  desde la época 4 memoriza. Las variantes del fold 0 (`results/ponderada_*`: aumentado,
  lr 3e-4, realce, normalizar, pooling máximo, atención) dan mejor AUC 0,51-0,59 y AUC
  medio por época 0,48-0,54.
- **Simulación:** un modelo sin señal da 0,56-0,58 quedándose con la mejor de 9-25
  épocas (±0,04 por época con ~220 pacientes). De ahí que se compare el AUC medio de
  todas las épocas y no el pico.
- **Comprobación del circuito** (`results/objduke_normal_f0`): la misma red aprende a
  reconocer la cohorte Duke con AUC 0,97 en la época 1 y 0,998 en la 13. El código
  funciona; el ~0,5 en pCR es falta de señal, no un error. La red capta el hospital
  trivialmente, así que en pCR puede usarlo como atajo.
- **Subtipo tumoral** (`results/obj{triple,her2,hrpos}_normal_f0`): la red tampoco lo
  aprende desde la imagen (mejor AUC 0,57, media por época 0,52-0,53; la pérdida de
  validación nunca baja de la de un modelo constante). Con la mama entera a 256 px la
  red ve el hospital pero no el tumor.
- **Caja del tumor** (`notebooks/localizacion_tumor.ipynb`, `src/data/tumor.py`):
  `sraw/eraw/scol/ecol` sí marcan el tumor, girados 90 grados y en la escala del
  volumen original; fiable para `n_xy ≤ 320` (~89 % de pacientes). La caja no llega a
  la app, así que solo sirve para entrenar o diagnosticar (iteración 7).

## Referencias para el informe

- Autores de BreastDCEDL (Fridman et al.): AUC 0,72 en test con un ViT (Duke 0,54,
  I-SPY1 0,68, I-SPY2 0,78). Otros trabajos solo con imagen: ~0,70-0,75; con
  variables clínicas, 0,70-0,75.
