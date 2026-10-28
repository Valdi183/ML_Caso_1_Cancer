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
  cortes 2D y 3 cohortes, **0,60-0,70 en test es buen trabajo**, y su modelo de
  referencia da **0,64 en test**: es la meta realista. El IC del AUC en test es de ±0,08.
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
Por cohorte, media ± desviación de los 5 folds, cada uno en su época elegida: spy2
0,586 ± 0,020, spy1 0,576 ± 0,098, Duke 0,531 ± 0,077 (el AUC global fuera de fold por
cohorte, con todas las pacientes juntas, está en la fila 9 de `iteraciones.md`).

**Por qué este y no otro:** desde la iteración 2 todo lo probado (capacidad,
resolución, realce, normalización, resumen, atención libre y supervisada con la caja
del tumor) queda en un AUC medio de ~0,545, con diferencias del tamaño del ruido. Este
es el más pequeño, el que menos memoriza y el de menor desviación entre folds.

**Comparación con los otros candidatos** (iteración 9, `python -m src.utils.comparar`,
`results/comparacion/resumen.json`; solo validación): con los 5 folds, la atención
supervisada con la caja da AUC medio 0,539 ± 0,020 frente a 0,551 ± 0,011 del congelado
(pareado, −0,012 ± 0,026; AUC global fuera de fold −0,017 [−0,057 a +0,024]) y
mismo hospital por época −0,020 en los folds 2-4. La regla fijada antes (sustituirlo solo
con AUC medio ≥ 0,581 y mismo hospital no menor) no se cumple: se queda el congelado. La
ponderada empata (+0,002 ± 0,010 de AUC medio).

**Pérdida normal frente a ponderada:** empatan en AUC. Con `pos_weight = N0/N1 = 2,4`
la ponderación suma ln 2,4 al logit y lleva la tasa base a 0,5: desplaza las
probabilidades sin cambiar el orden. Por eso a 0,5 cambia la sensibilidad (normal
0,01; ponderada 0,42 ± 0,26) y no el AUC. Se queda la **normal**, cuyas
probabilidades están calibradas, y el punto de corte lo da el umbral.

**Umbral:** por paciente, agregando por la media de sus cortes, con el criterio de
**especificidad ≥ 0,90** sobre las predicciones fuera de fold de los 5 folds. Criterio
clínico: el error grave es el falso positivo (predecir pCR en quien no la alcanzará
podría llevar a desescalar un tratamiento que sí necesita). Regla: el **percentil 90**
(sin interpolar, `method="higher"`) de la probabilidad por paciente de **todas las
pacientes pCR=0 fuera de fold juntas**; se predice pCR si la probabilidad supera el
umbral (`python -m src.utils.umbral`, campo `umbral_final` de
`results/umbral/resumen.json`). Es el que cumple el criterio sobre el conjunto y no se
elige mirando la sensibilidad.

**Valor: 0,3635.** Fuera de fold y por paciente: especificidad 0,901, sensibilidad
0,177, VPP 0,425, VPN 0,725, F1 macro 0,527 («siempre no pCR»: 0,414); con un corte
suelto, especificidad 0,891 y sensibilidad 0,187. Los umbrales de cada fold por
separado van de 0,307 a 0,405 y, con el umbral final, la especificidad por fold va de
0,77 a 0,99: es la incertidumbre del umbral. La regla inicial (mediana de los umbrales
de cada fold, 0,3393) daba especificidad 0,839 por esa diferencia de escala entre folds;
se corrigió el procedimiento para cumplir el criterio fijado, sin cambiar el criterio
(iteración 10 en `iteraciones.md`).

**Predicción final:** el conjunto de los 5 modelos (media de sus probabilidades).
Ninguna validación lo ve entero (cada modelo ha visto en train los otros 4 folds),
así que **solo se puede evaluar en test**.

**Avisos para el informe:**
- El umbral se elige por paciente (media de ~10 cortes) pero la app recibe un solo
  corte: la probabilidad de un corte es más ruidosa y el umbral no se transfiere
  exactamente (en validación, por corte: especificidad 0,891 frente a 0,901 por paciente).
- El conjunto solo se evalúa en test, y el umbral sale de las predicciones de modelos
  individuales, no del conjunto.
- Duke es el 24 % del test (42 de 176) y su AUC en validación ronda 0,5: bajará el
  AUC global.

## Resultado en test (una sola vez)

Evaluado una vez con todo decidido antes en validación (plan con el SHA-256 de cada peso
en `results/test/plan.json`; `python -m src.utils.evaluar_test`, que se niega a repetirse;
resultados en `results/test/resumen.json` y figuras `docs/figuras/test_*.png`). Conjunto
de los 5 modelos, media de cortes por paciente, umbral 0,3635. 176 pacientes (53 pCR).

| | Por paciente | Corte suelto (la app) | Siempre no pCR | Fridman et al. 2025 (preentrenado) |
|---|---|---|---|---|
| AUC | 0,544 [0,457-0,629] | 0,545 [0,464-0,624] | 0,5 | 0,72 |
| Sensibilidad | 0,038 [0,000-0,094] | 0,049 | 0 | 0,27 |
| Especificidad | 0,935 [0,887-0,976] | 0,925 | 1 | 0,95 |
| VPP / VPN | 0,200 / 0,693 | 0,220 / 0,693 | — / 0,699 | |
| F1 pCR=1 / pCR=0 | 0,063 / 0,796 | 0,080 / 0,792 | 0 / 0,823 | |
| F1 macro | 0,430 [0,382-0,484] | 0,436 | 0,411 | |
| Exactitud | 0,665 | 0,661 | 0,699 | |

Matriz por paciente: VP 2, FP 8, FN 51, VN 115. Por cohorte (AUC): spy1 0,351
[0,182-0,528] (35 pacientes), spy2 0,565 [0,447-0,675] (99), Duke 0,650 [0,430-0,866]
(42). AUC entre pares del mismo hospital 0,552 [0,457-0,648]. Brier 0,2104, igual que el
de un modelo constante con la tasa de pCR de train (0,2105). Los 5 modelos por separado
dan 0,507-0,577 (0,543 ± 0,027); el conjunto, 0,544.

**Frente a las referencias:** el AUC de test queda unos **0,10 por debajo de la meta
del profesor (0,64 en test)**, justo por encima del límite superior de su IC
[0,457-0,629]; esa meta tiene su propia incertidumbre (±0,08), así que la diferencia no es
concluyente, pero en test el modelo no se distingue del azar. Fridman et al. 2025
(0,72 en el mismo test) usa un modelo **preentrenado**: es una cota superior, no
comparable con una CNN desde cero. El 0,544 coincide con la estimación sin sesgo de
validación (AUC medio de todas las épocas, 0,551 ± 0,011).

## Métricas descriptivas (después de test; no cambian nada)

`python -m src.utils.metricas_extra` (`results/metricas_extra/resumen.json`, figuras
`docs/figuras/metricas_*.png`; fila 12 de `iteraciones.md`). Lo de test sale de las
predicciones guardadas; no se vuelve a pasar ninguna imagen por la red.

- **En test el modelo no se distingue del azar**: AUC frente a 0,5 por permutación,
  p = 0,35 por paciente y 0,31 por corte (permutando entre pacientes); AP 0,309 frente a
  0,301 de un modelo sin señal; Brier skill score 0,000. En validación fuera de fold hay
  algo de señal: AP 0,365 [0,322-0,419] frente a 0,294.
- **Potencia baja**: con 53 pCR y 123 no pCR, si el AUC real fuera 0,55 el IC excluiría
  0,5 solo el 18 % de las veces (fórmula y simulación coinciden). Test no puede separar
  0,55 de 0,5.
- **Calibración**: pendiente 0,45 fuera de fold y 0,72 en test, con IC muy anchos; las
  probabilidades apenas ordenan, así que la pendiente queda lejos de 1.
- **Sensibilidad en test**: los 5 modelos discrepan mucho en escala (sobre el umbral,
  de 2 a 56 pacientes) y su media comprime algo la dispersión, así que el conjunto solo
  supera el umbral en 10 pacientes. Es el riesgo anotado en la fila 10.
- **Un corte frente a la paciente**: ICC 0,92 entre los cortes de una paciente; la app,
  con un corte, verá casi la probabilidad media de la paciente (15,5 % de pacientes tienen
  cortes a los dos lados del umbral).
- **Modelo clínico** (subtipo + edad + volumen, solo validación): 0,692 ± 0,032 por fold,
  sin cambios frente al EDA; global fuera de fold 0,686 [0,652-0,719]; entre pares del
  mismo hospital 0,679, así que no viene de la cohorte; +0,141 ± 0,041 sobre la CNN en
  cada fold. En esta versión del dataset, `tum_vol` de Duke es continuo (no 5 categorías)
  pero más pequeño (mediana 6,1 frente a 14,4-14,7); quitar el volumen baja el AUC a 0,669.

## Nota numérica: GPU con TF32 frente a la app en CPU

Las probabilidades guardadas (`predicciones_val.csv` de cada ejecución y
`results/test/predicciones_test*.csv`) se calcularon en la RTX 3060 Ti con lotes de 128.
Con ese tamaño de lote cuDNN usa TF32 en las convoluciones, que PyTorch activa por defecto
(`torch.backends.cudnn.allow_tf32`). La app calcula en FP32 en CPU, como la referencia de
sus tests. Comprobado en el fold 3 de validación: GPU con lotes de 128 reproduce lo guardado
(diferencia < 3e-8); CPU frente a lo guardado, hasta 4,2e-4; GPU sin TF32 frente a CPU,
< 3e-7. El modelo y sus pesos no cambian; solo el redondeo al predecir. En test, una
paciente está a 1,9e-4 del umbral (y 3 cortes a menos de 5e-4), así que su clase podría
salir distinta en FP32. No se recalcula nada de test.

## Diagnósticos anteriores a la iteración 1

- **EDA** (`notebooks/eda.ipynb`): variables clínicas (subtipo + edad + volumen) dan
  AUC 0,69 (0,692 ± 0,032: regresión logística, media ± desviación de los AUC por fold
  con la columna `fold`, solo pacientes de train; no es un AUC de test ni el global
  fuera de fold que se usa con la CNN); solo la cohorte, 0,55; medidas globales de realce, ~0,5. Cortes de una
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

AUC publicados para predecir pCR desde la RM previa (lista del correo del profesor):

- **Fridman et al. 2025** (autores de BreastDCEDL), mismo test: AUC 0,72 con un ViT
  (Duke 0,54, I-SPY1 0,68, I-SPY2 0,78). Usa un modelo **preentrenado**: cota superior,
  no comparable con una CNN desde cero. Solo I-SPY2 con clínica: 0,85; en el subgrupo
  HR+/HER2−: 0,94. Comprobado en el artículo (arXiv 2506.12190v3): pasan las imágenes a
  8 bits para usar modelos preentrenados y fusionan las tres fases como RGB (ellos mismos
  lo citan como limitación). En métodos el test es de 176 pacientes y en la tabla de 177;
  por cohorte, I-SPY1 35, I-SPY2 99 y Duke 43 (en nuestro test, 42).
- **Braman:** 0,74.
- **Cain:** 0,71 [0,58-0,83].
- **Liu:** 0,79; con clínica, 0,86.
- **Joo:** 0,89; solo clínica, 0,83.
- **Referencia del profesor:** 0,64 en test. Es la meta realista para este trabajo
  (cortes 2D, 3 cohortes, CNN desde cero).
