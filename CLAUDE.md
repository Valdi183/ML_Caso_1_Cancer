# Contexto para asistentes (TEMPORAL)

Fichero temporal de contexto para Claude (Claude Code y el Proyecto de claude.ai). Se
borrara al terminar la parte tecnica. **La fuente de verdad son README.md y docs/**:
aqui solo hay un resumen que remite alli. Regla: CLAUDE.md nunca es el unico sitio
donde esta algo importante; lo que deba sobrevivir se escribe en README.md o docs/.
Si este fichero contradice al repo, manda el repo.

## El caso

Trabajo academico individual. Dataset **BreastDCEDL** (1.273 pacientes, 12.703 cortes).
Objetivo: predecir **pCR** (sin tumor invasivo tras la quimioterapia) desde una RM DCE
**anterior** al tratamiento. Pronostico, no deteccion. Un corte = 3 PNG 256x256 en gris
(`_PRE`, `_EARLY`, `_LATE`): tres instantes del contraste, **no RGB**; `z012` es una
altura, no un tiempo. pCR=0 es el 70,6 %. Tres cohortes (`spy1`, `spy2`, `duke`)
procesadas distinto. `utils_caso.py` vive en la carpeta del dataset; `config.utils()`
lo importa. Que pide el profesor y la rubrica: `docs/decisiones.md`.

**Reglas que invalidan el trabajo** (README, "Reglas del caso"): CNN propia desde cero
(nada preentrenado); separacion por paciente con la columna `fold`; `test` una sola vez,
al final; no publicar la validacion privada.

## Estado actual (21-10-2026)

**Modelo congelado** (detalle y porques en `docs/decisiones.md`; reproduccion en el
README, "Modelo final"):

- Opciones (de `experimentos/it6.txt`):
  `--aumentado --dropout 0.7 --wd 1e-3 --reducir 2 --filtros 8,16,32`, perdida normal,
  semilla 42. 3 bloques, resumen por media, 128 px, 18.329 parametros.
- Ejecuciones: `it6_3bloques_f0`, `it6_3bloques_f1`, `it8_3bloques_f2`, `_f3`, `_f4`.
  Pesos en `models/<nombre>.pt` (ignorados por git: se suben con `git add -f`).
- 5 folds: **AUC medio de todas las epocas 0,551 ± 0,011**; epoca elegida
  0,589 ± 0,020 (inflado). Por cohorte (epoca elegida): spy2 0,586, spy1 0,576
  (IC enorme), Duke 0,531.
- **Perdida normal**, no ponderada: empatan en AUC (0,553 ± 0,007 la ponderada); la
  ponderacion suma ln 2,4 al logit, desplaza las probabilidades sin cambiar el orden,
  y solo mueve la sensibilidad a 0,5. La normal da probabilidades calibradas.
- **Umbral:** criterio fijado, especificidad >= 0,90 por paciente, agregando por la
  media de sus cortes, sobre las predicciones fuera de fold. **Valor pendiente.**
- El modelo final es el **conjunto de los 5 modelos** (media de probabilidades) y solo
  se puede evaluar en test.

## Lo descartado (detalle en `docs/iteraciones.md`)

- Iteraciones 1-8: red base de 4 bloques, perdida ponderada, lr, realce EARLY-PRE,
  normalizar por corte, 256/128/64 px, 3 a 5 bloques, ancha, mini, media+maximo,
  atencion libre y supervisada con la caja del tumor. Todo queda en AUC medio ~0,545,
  con diferencias del tamaño del ruido (±0,04 por fold).
- La red aprende la cohorte (0,998) y a mirar el tumor (93 % de la atencion en val),
  pero apenas pCR (subtipo triple: no concluyente): el limite son los datos (un corte
  2D, ~1.000 pacientes, pCR depende tambien del tratamiento).
- Busqueda de estructuras cerrada; no abrir lineas nuevas. Las clinicas (AUC 0,69) son
  solo comparacion.

## Pendiente, en orden (entrega: 30-10-2026)

1. **Umbral** con las predicciones fuera de fold (`results/<ejecucion>/predicciones_val.csv`
   de las 5 ejecuciones): script nuevo, sin tocar test.
2. **App desplegada con URL** (`app/`): un corte (PRE, EARLY, LATE) -> probabilidad de
   pCR con el conjunto y el umbral. Sin preprocesado propio: `src.models.cnn.cargar`.
3. **Test UNA vez** con todas las metricas de la rubrica (`docs/decisiones.md`): matriz,
   AUC con IC, precision/recall/F1 por clase, F1 macro, frente a "siempre no pCR", por
   cohorte, calibracion, casos fallados con imagenes; comparacion con clinicas (0,69).
4. **Informe** y **5 diapositivas**.

Avisos para el informe (tambien en `docs/decisiones.md`):
- El umbral se elige por paciente (media de ~10 cortes) pero la app recibe un solo corte.
- El conjunto de los 5 modelos solo se puede evaluar en test.

## Maquinas

- **Portatil** (Windows 11, i3, 8 GB, sin GPU): **la principal desde ahora**. Codigo,
  app, analisis y evaluacion del test, todo en CPU (la red final es pequeña).
- **PC de sobremesa** (RTX 3060 Ti): solo si algo necesitara GPU; no hay nada previsto.
- **Laboratorio**: ya no hace falta.

Preparar el portatil: `git pull`; si no hay `.venv`, `.\configurar.ps1` (README,
"Puesta en marcha"); dataset donde diga `.bdcedl_ruta` o la variable `BDCEDL`, que
resuelve `src/config.py` (README, "Donde viven los datos"). Comprobar con
`python -m src.utils.verificar_entorno`. Traer los 5 `.pt` del modelo final (se suben
desde el PC con `git add -f`).

## Como trabajar

- Todo como modulo desde la raiz: `python -m src.<...>`; imports `from src import config`.
  Rutas solo desde `src/config.py`. Lo que pida calculo, en script `.py`.
- `ruff check .` y `pytest -q` deben pasar antes de dar algo por terminado.
- **Git lo hace el usuario a mano.** No ejecutar git que cambie nada (add, commit, push,
  restore...); consultar si se puede. Al terminar, listar los ficheros creados o
  modificados y proponer el mensaje (Conventional Commits, en imperativo).
- Explicar el porque (el concepto) antes de implementar.
- **Nunca pedir ni aceptar tokens o credenciales en el chat.**
- `import torch` antes que matplotlib en scripts y cuadernos que dibujen.
- Si cambia este fichero o `docs/`, recordar al usuario que lo suba al Proyecto de
  claude.ai.

## Cierre (antes de borrar este fichero)

- [ ] Buscar en todo el repo que ningun fichero mencione CLAUDE.md (README, docs, codigo,
      ruff.toml, pytest.ini, CI, .gitignore) y quitar esas menciones. A 21-10-2026:
      `README.md` (arbol de la estructura) y `docs/iteraciones.md` ("detalle en
      `CLAUDE.md`", que debe pasar a remitir a `docs/decisiones.md`).
- [ ] Decidir que hacer con la carpeta `.claude/` si existe (hoy no existe).
- [ ] Confirmar que README.md basta para reproducir y usar todo (entrenar, umbral, test,
      app) sin CLAUDE.md.
- [ ] El borrado lo hace el usuario; ningun asistente lo borra por su cuenta.
