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
| 3 | Con la red A, (A) darle EARLY−PRE y LATE−EARLY ya calculados, o (B) normalizar cada corte para quitar las diferencias de intensidad entre hospitales, hará subir el AUC de validación por encima de ~0,57 | `it3_pequena_realce_f0`, `it3_pequena_norm_f0` | *pendiente* | *pendiente* |
