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
| 2 | Si el sobreajuste es por exceso de capacidad o por pasos grandes, (A) una red 4 veces más pequeña con dropout 0,7 y weight decay 1e-3, o (B) la red base con lr 1e-4, mantendrán el AUC de validación subiendo o estable más épocas | `it2_pequena_reg_f0`, `it2_lr1e-4_f0` | *pendiente* | *pendiente* |
