# Registro de iteraciones del modelo

Cada iteración prueba **una idea** con uno o dos modelos en el fold 0, mira las curvas
(AUC por paciente de entrenamiento y validación, pérdida) y decide el siguiente cambio.
Lo que funcione se confirma después con los 5 folds; el test se usa una sola vez al final.

**Cómo se lee un resultado** (criterios fijados antes de ver los datos):

- *Sano*: AUC de train y de validación suben juntos; la diferencia en la época elegida es pequeña.
- *Memoriza*: train sube hacia 1 y validación se queda plana o baja.
- *No aprende*: train tampoco sube; el problema es de optimización o de capacidad, no de sobreajuste.
- La validación tiene ~220 pacientes: diferencias de AUC menores de ~0,04 son ruido.

**Antes de la iteración 1** (resumen, detalle en `CLAUDE.md`): la red base con pérdida
ponderada y sus variantes (aumentado, lr 3e-4, realce, normalizar, pooling máximo y
atención) se quedan en AUC de validación ≈ 0,5–0,59 en el fold 0. La parada temprana
cortaba en la época 9 tras un pico de suerte en la época 1, y no se registraba el AUC de
entrenamiento, así que no se podía distinguir entre memorizar y no aprender.

| Iteración | Hipótesis | Modelos | Resultado (fold 0) | Decisión |
|---|---|---|---|---|
| 1 | Con el entrenamiento corregido (mínimo 15 épocas, época elegida por AUC suavizado) y viendo el AUC de train, sabremos si la red base memoriza o no aprende. Además, normal frente a ponderada | `it1_normal_f0`, `it1_ponderada_f0` | Ninguno de los dos: validación ≈ 0,58-0,59 en las épocas 1-4 y luego **baja** hasta ≈ 0,50; train sube despacio (0,61 → 0,72 en 15 épocas). Pérdida de validación mínima 0,602 (normal) frente a 0,604 de un modelo constante. Normal: sensibilidad a 0,5 = 0 (predice siempre «no pCR»); ponderada: oscila entre 0,02 y 0,80. AUC igual (0,59 vs 0,58) | Patrón de sobreajuste temprano: lo que se generaliza se aprende al principio y después aprende rasgos de las pacientes de entrenamiento. Iteración 2: frenar eso (red pequeña + regularización; lr más bajo). Se sigue con pérdida normal (probabilidades interpretables; el umbral se elige aparte) |
| 2 | Si el sobreajuste es por exceso de capacidad o por pasos grandes, (A) una red 4 veces más pequeña con dropout 0,7 y weight decay 1e-3, o (B) la red base con lr 1e-4, mantendrán el AUC de validación subiendo o estable más épocas | `it2_pequena_reg_f0`, `it2_lr1e-4_f0` | **A (pequeña + regularización)**: la validación ya **no cae**: AUC medio 0,566 en las épocas 1-5 y 0,566 en las 11-15 (iteración 1: 0,581 → 0,509); train sube más despacio (0,69 en la 15). Época elegida 2 (val 0,61, probablemente un pico de suerte). **B (lr 1e-4)**: peor; validación 0,535 → 0,492 y train igual que en la iteración 1 (0,74) | La capacidad sí influye y el lr no: se sigue desde A. Mejora pequeña (+0,02 de AUC medio, dentro del ruido); lo que cuenta es la forma de la curva. Iteración 3: sobre A, dar la señal del realce explícita, o normalizar cada corte |
| 3 | Con la red A, (A) darle EARLY−PRE y LATE−EARLY ya calculados, o (B) normalizar cada corte para quitar las diferencias de intensidad entre hospitales, hará subir el AUC de validación por encima de ~0,57 | `it3_pequena_realce_f0`, `it3_pequena_norm_f0` | Ninguno mejora a 2A. **Realce**: AUC val medio 0,522 (2A: 0,564), últimas 5 épocas 0,503. **Normalizar**: 0,530; sube despacio hasta ~0,57 (épocas 13-19) y luego memoriza (train 0,82, val 0,50) | Ni la señal del realce explícita ni quitar la intensidad de cada hospital ayudan. Tras 7 modelos, todo el fold 0 queda en un AUC medio de 0,52-0,56 y las diferencias son del tamaño del ruido. Comprobado que no es un error de etiquetas: pCR igual en cortes y pacientes, ninguna paciente en dos folds. Antes de seguir cambiando la red, medir 2A en los otros 4 folds |
| 4 | Un solo fold de ~220 pacientes no basta para comparar modelos que difieren en 0,02. Entrenar 2A en los folds 1-4 da su AUC con media ± desviación (lo pide la rúbrica) y dice si el fold 0 es especialmente difícil | `it4_pequena_reg_f1` … `_f4` | **2A en los 5 folds**: AUC de la época elegida 0,575 ± 0,032; suavizado 0,571 ± 0,021; **AUC medio de todas las épocas 0,544 ± 0,020** (folds 0,515-0,564). Diferencia train−val en la época elegida 0,07 ± 0,10 (fold 3: 0,23, memoriza). Duke ≈ 0,45-0,50 en todos los folds | El fold 0 no era el difícil: todos dan ~0,55. Con la configuración actual la red no pasa de ~0,57. Hace falta un cambio de fondo. Restricción: en la defensa la app recibe **una sola muestra** (un corte) por paciente, así que nada que necesite varios cortes de la misma paciente (2,5D, predicción por paciente) |
| 5 | Dos cambios de fondo compatibles con la app, cada uno en los 5 folds: (A) **reducir la imagen a 128 px** dentro de la red (ve zonas más grandes de la mama y menos detalle fino que memorizar); (B) **pooling de atención** (puede dar más peso a la zona del tumor en lugar de promediar toda la mama). Criterio fijado antes: solo cuenta una mejora de la media de 5 folds ≥ 0,03 sobre 2A | `it5_red128_f0…f4`, `it5_atencion_f0…f4` | *pendiente* | *pendiente* |
