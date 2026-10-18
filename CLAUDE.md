# Contexto para Claude

Trabajo academico individual. Este fichero es el contexto compartido del proyecto:
Claude Code lo lee solo en el portatil, y en el laboratorio se usa Claude en el
navegador (claude.ai), dentro de un Proyecto que tiene este fichero como
conocimiento. Lo que deba recordarse en los dos sitios se escribe aqui (y se vuelve
a subir al Proyecto de claude.ai cuando cambie), no en la memoria local.

## El caso

Dataset **BreastDCEDL**. El objetivo NO es detectar cancer: es predecir **pCR** (que
tras la quimioterapia no quede tumor invasivo) a partir de una RM DCE tomada **antes**
del tratamiento. Es pronostico: el AUC esperable ronda 0,70-0,80.

- 1.273 pacientes, 12.703 cortes, 38.109 PNG. Un corte = 3 PNG 256x256 en gris
  (`_PRE`, `_EARLY`, `_LATE`): tres instantes del contraste, **no RGB**.
- `z012` es una altura axial, no un tiempo. La señal esta en `EARLY - PRE`.
- pCR=0 es el 70,6 % (pos_weight 2,400).
- Tres cohortes procesadas distinto (`spy1`, `spy2`, `duke`): sesgo real que el
  informe debe discutir.
- `utils_caso.py` vive en la carpeta del dataset (fuera del repo); `config.utils()`
  lo importa. Resuelve particion por paciente, `Dataset` y agregacion por paciente.

**Reglas que invalidan el trabajo** (ver tambien el README):
1. CNN propia desde cero: nada de ResNet/VGG/EfficientNet ni pesos preentrenados.
2. Separacion por paciente con la columna `fold` de `samples.csv`; nunca cortes al azar.
3. `test` se usa una sola vez, al final.
4. No publicar la validacion privada.

Entregables: repo + pesos + informe + app web **desplegada con URL** + 5 diapositivas.
En la defensa el profesor mete 5 muestras privadas en la app sin tocar el codigo:
el preprocesado de la app debe ser identico al del entrenamiento (por eso vive
dentro de la red, en `src/models/cnn.py`). **Entrega el 30-10-2026.**

## Flujo portatil / laboratorio

- **Portatil** (Windows 11, i3, 8 GB, sin GPU): se escribe el codigo y se prueba con
  `python -m src.training.entrenar --rapido`. Entrenar de verdad aqui es inviable
  (~17 min/epoca).
- **Laboratorio** (Linux, GPU AMD RX 6700 XT, ROCm 7.14; no hay NVIDIA ni
  `nvidia-smi`): entrenamiento real. PyTorch para ROCm usa la misma API
  `torch.cuda` (`torch.cuda.is_available()` es True y el dispositivo se llama
  "cuda"), asi que el codigo no cambia. ~53 s/epoca con la red base. Los equipos
  **ya vienen preparados** (lo dijo el profesor; por eso se borro `configurar.sh`):
  basta clonar y configurar git. Se deja entrenando dia y noche con
  `bash experimentos/lanzar.sh itN` (bloquear pantalla, no cerrar sesion): entrena en
  segundo plano cada linea de `experimentos/itN.txt` (opciones de entrenar), con log en
  `logs/itN.log`; `bash experimentos/ver.sh itN` para ver el progreso. Se niega a
  lanzar si PyTorch no ve la GPU. Se trabaja por rondas de 6-8 modelos
  (`docs/iteraciones.md`). **No se puede instalar nada** fuera de un
  entorno virtual: todo paquete va en `.venv`, y no hay sudo. Claude se usa desde
  el navegador, asi que alli Claude no ejecuta nada: el usuario pega codigo y salidas.
- **PC de sobremesa** (Windows 11, i5-11400F, 16 GB, NVIDIA RTX 3060 Ti 8 GB, desde
  17-10-2026): tercera maquina de entrenamiento, con Claude Code. Entorno con
  `.\configurar.ps1` (`.venv` con PyTorch CUDA cu128, dataset en `C:\bdcedl`). Se
  lanza igual, `bash experimentos/lanzar.sh itN` desde **Git Bash**: el script
  encuentra `.venv/Scripts/python.exe` y, sin `pgrep`, usa `logs/en_marcha.pid`.
  Que Windows no se suspenda mientras entrena, y no jugar a la vez (comparte la VRAM).

Todo script de entrenamiento: modo rapido y completo sin tocar codigo, detecta
cuda/cpu solo, sin rutas fijas (`src/config.py`), guarda en `results/` y `models/`
para commitearlo antes de irse del laboratorio.

## Organizacion del repo

Sigue la plantilla de la asignatura (UAX): codigo en `src/` (`config.py`, `data/`,
`models/`, `training/`, `utils/`), `notebooks/`, `docs/figuras/`, `models/` (pesos),
`results/<experimento>/`, `tests/`, `app/`, `presentations/`, CI en
`.github/workflows/ci.yml` (ruff + pytest en cada push). Todo se ejecuta **desde la
raiz como modulo**: `python -m src.training.entrenar ...`; los imports son
`from src import config`, `from src.models.cnn import ...`.
- Semilla e hiperparametros por defecto: `config.DEFECTO` (los usa argparse).
- Pesos: `*.pt` ignorado; los que se entregan, `git add -f models/<nombre>.pt`.
- Antes de proponer un commit: `ruff check .` y `pytest -q` deben pasar.
- Cuadernos: los avisos E402/E501/I001 estan desactivados a proposito (ruff.toml).

**Seguridad en el laboratorio:** cada PC tiene una cuenta Linux "alumno" comun a
quien se siente en ese equipo, asi que el siguiente ve los ficheros. Lo que importa
es que no pueda hacer push al repo:
- git: `credential.helper` vacio y `user.*` solo con `--local` (a mano, ver README).
  El push se autentica con un token fine-grained que el usuario pega cada vez.
- VS Code: `.vscode/settings.json` desactiva "Sign in with GitHub". No aceptarlo.
- Navegador: GitHub y claude.ai en ventana privada; cerrar sesion al irse.
- **Nunca pedir ni aceptar que se pegue el token en el chat**: queda guardado en
  el historial de la conversacion.

**Commits:** los hace siempre el usuario. No hacer `git commit` ni `git push` sin su
OK explicito para ese commit; preparar los cambios y proponer el mensaje en
formato Conventional Commits, en imperativo (`feat(models): añadir ...`,
`fix(data): ...`, `docs: ...`, `chore: ...`), un proposito por commit.

## Estado del modelo

`src/models/cnn.py` + `src/training/entrenar.py`: CNN de 4 bloques (16-32-64-128),
preprocesado sin pesos dentro de la red (`--realce` EARLY-PRE y LATE-EARLY,
`--sin-late`, `--normalizar` = z-score por corte con una media/desviacion para las
tres fases), resumen del mapa final con `--pooling media|max|mediamax|atencion`
(atencion empieza igual que la media), dropout, un logit; agregacion por paciente
fuera de la red. 294.129 parametros la red base.

Primer entrenamiento completo (`results/ponderada_f0`, fold 0): AUC por paciente
0,58 en la epoca 1, parada en la 9. La perdida de validacion nunca baja de la de
predecir siempre 0,5 (0,979) y desde la epoca 4 memoriza (train baja, val sube);
AUC inestable entre epocas (0,47-0,58). AUC similar por cohorte (0,55-0,59).
Cola del fold 0 (08-10-2026, sobre `--ponderada --aumentado`: lr 3e-4, realce,
normalizar, pooling max, atencion): mejor AUC 0,51-0,59, AUC medio por epoca
0,48-0,54, perdida de validacion nunca claramente por debajo de 0,979. Simulado:
un modelo SIN señal da 0,56-0,58 quedandose con la mejor de 9-25 epocas (±0,04 por
epoca con 219 pacientes). Conclusion: ninguna variante aprende pCR; las diferencias
son ruido. Con realce/normalizar la mejor epoca llega tarde (17-18): el aumentado
frena la memorizacion. AUC por cohorte no interpretable (spy1: 21 pacientes, 4 pCR).

Comprobacion del circuito (`results/objduke_normal_f0`, 13-10-2026): la misma red
aprende la cohorte Duke con AUC 0,97 en la epoca 1 y 0,998 en la 13 (41/42 Duke
acertadas, 0 falsos positivos). El codigo funciona: el ~0,5 de pCR es falta de
señal en este planteamiento, no un error. Dos avisos: (1) la red capta la cohorte
trivialmente, asi que en pCR puede usarla como atajo; (2) epocas 2 y 8 con picos
de perdida de validacion (1,98) y la sensibilidad a 0,5 hundida aunque el AUC se
mantiene: con lr 1e-3 las probabilidades se desplazan en bloque entre epocas.
Diagnostico de subtipo (`results/obj{triple,her2,hrpos}_normal_f0`, 13-10-2026): la
red tampoco aprende el subtipo tumoral desde la imagen. Mejor AUC 0,57 (lo que da
el azar quedandose con la mejor epoca), media por epoca 0,52-0,53, dentro de spy2
0,54-0,57; la perdida de validacion nunca baja de la de un modelo constante
(triple 0,668 vs 0,656; her2 0,540 vs 0,534; hrpos 0,714 vs 0,677) mientras la de
train cae (memoriza). Conclusion: con la mama entera a 256 px la red ve el
hospital pero no el tumor. MIL sobre las mismas imagenes no lo arreglaria: el
siguiente paso es CENTRARSE EN EL TUMOR.

Caja del tumor (`notebooks/localizacion_tumor.ipynb`, 13-10-2026): sraw/eraw/scol/ecol
SI marcan el tumor, pero estan giradas 90 grados y en la escala del volumen original:
filas = 256-ecol*k .. 256-scol*k, columnas = sraw*k .. eraw*k, k = 256/n_xy
(`src/data/tumor.py::caja_en_imagen`). Fiable para n_xy <= 320 (~89 % de pacientes,
comprobado a ojo en ~40); para n_xy 384-512 (casi todas spy2) no cae y devuelve None.
**La caja NO llega a la app** (en la defensa solo hay imagenes): se puede usar para
entrenar o diagnosticar, pero el modelo final tiene que localizar el tumor solo.

Expectativas realistas (literatura, ver conversacion 13-10-2026): los autores de
BreastDCEDL obtienen AUC 0,72 en test con un ViT (Duke 0,54, I-SPY1 0,68, I-SPY2
0,78); solo imagen en otros trabajos ~0,70-0,75. Con CNN desde cero y estos datos,
0,60-0,65 seria un buen resultado y ~0,70 excelente. Con clinicas: 0,70-0,75.
El plan inicial (imagen + modelo imagen+clinicas) quedo sustituido por el correo del
14-10 (abajo): el modelo es solo imagen y las clinicas son solo comparacion.

## Indicaciones del profesor (correo 14-10-2026) — MANDAN sobre lo anterior

- El caso es una **CNN propia desde cero que predice pCR con las 3 fases**. Evaluacion
  por paciente (media de sus cortes), umbral elegido en validacion, test una vez.
- Las variables de patients.csv son **comparacion, no modelo principal** (en clinica
  no se tienen todas; la app solo recibe imagenes). El modelo imagen+clinica pasa a
  ser opcional y de baja prioridad. Los diagnosticos (cohorte, subtipo, caja) ya
  cumplieron; no abrir mas lineas laterales.
- Referencia: imagen pretratamiento + test independiente = 0,70-0,80 publicado; con
  cortes 2D y 3 cohortes, **0,60-0,70 en test es buen trabajo**. IC del AUC en test
  ±0,08. Su figura "BIEN": 5 folds, AUC de train y val suben juntos hasta ~0,66-0,74
  en 15-25 epocas, test 0,64. La nuestra se parece a la "MAL" (val plano ~0,58).
- Rubrica de 10: EDA; particion por paciente con **5 folds media ± desviacion**;
  arquitectura justificada (fases no como RGB); reproducible (semilla, parada, epoca
  elegida); **curvas de AUC de train Y validacion** y diferencia en la epoca elegida;
  **perdida normal vs ponderada** y su efecto en la sensibilidad; umbral y agregacion
  elegidos en validacion con **criterio clinico explicito** (que error es peor);
  test una vez con matriz, AUC con IC, precision/recall/F1 por clase, F1 macro y
  comparacion con "siempre no pCR"; por cohorte, calibracion, casos fallados con
  imagenes; bibliografia, limitaciones y app con el mismo preprocesado.

## Iteraciones (detalle en `docs/iteraciones.md`)

Mejor configuracion hasta ahora (2A): `--aumentado --filtros 8,16,32,64 --dropout 0.7
--wd 1e-3`, perdida normal. 5 folds: AUC elegido 0,575 ± 0,032, medio 0,544 ± 0,020.
Realce, normalizar y lr 1e-4 no mejoran. **La app recibe UNA muestra (un corte) por
paciente** (GUIA.md C7): nada de 2,5D ni prediccion por paciente.
Iteracion 5 (5 folds): `--reducir 2` (128 px) 0,587 ± 0,017 y atencion 0,582 ± 0,031,
iguales a 2A, pero 128 px va a 12 s/epoca (frente a 40): es la base para explorar.

**Por que sobreentrena (lo vio el profesor en la perdida de validacion):** ~880
pacientes con ~10 cortes casi iguales = pocos ejemplos distintos; la etiqueta es por
paciente, asi que el atajo es memorizar "esta mama = pCR"; lo poco que generaliza se
aprende en las epocas 1-2 y despues solo memoriza; la perdida de val se dispara porque
las predicciones se vuelven seguras y erroneas. **Es un problema de estructura** (que le
resulta facil aprender a la red), no de hiperparametros: lr, realce y normalizar no
movieron nada; la red mas pequeña fue lo unico que ayudo.

**Plan (17-10-2026), como los compañeros que mejor van (25-35 estructuras):** catalogo
de arquitecturas en `src/models/` con `--arquitectura`; rondas de 6-8 estructuras,
cribado en folds 0 y 1 frente a la referencia en esos folds, solo pasan a 5 folds las
que mejoren >= 0,03 de AUC medio; cuaderno `notebooks/comparativa_modelos.ipynb` con
las curvas de todas. Ideas por causa: menos pesos (separables, menos bloques); contra
memorizar (Dropout2d, GroupNorm, mixup, aumentado de intensidad); contra el tumor
diluido (atencion/max, capa cinetica 1x1 por pixel, stem 7x7, dilatadas, residuales).
Despues: umbral clinico, test una vez, app, informe, 5 diapositivas.

## Relevo: como continuar (para cualquier Claude que retome el trabajo)

Hay tres asistentes: Claude Code en el portatil, Claude Code en el PC de sobremesa y
Claude en claude.ai (tutor, no toca el repo). **La unica fuente de verdad es el repo.**

- **Antes de nada `git pull`; al acabar, commit y `git push`.** Solo una maquina edita
  CODIGO a la vez (las otras solo entrenan y suben `results/`). Si se cruzan cambios
  sin commitear, guardar copia, `git restore`, `git pull` y volver a aplicarlos.
- **Estado (18-10-2026):** iteracion 6 (forma de la red) **terminada**: ninguna forma
  mejora. Analisis 5b hecho (`python -m src.utils.analisis_cv`). **Siguiente paso: it7**,
  atencion supervisada con la caja (abajo). **Catalogo de estructuras SIN empezar** (capa cinetica 1x1,
  residuales, separables, GroupNorm, Dropout2d, stem 7x7, dilatadas, mixup, aumentado
  de intensidad): quien lo empiece lo anota aqui para que el otro no lo duplique.
- **Lista para lanzar (18-10-2026): atencion supervisada con la caja** (iteracion 7,
  `experimentos/it7.txt`). `--peso-caja λ` (exige `--pooling atencion`): perdida = BCE +
  λ·L_caja, L_caja = −log(masa de atencion en la caja del tumor), solo al entrenar; la red y
  su `config` no cambian (la app no necesita la caja). La mascara pasa por el mismo aumentado
  que las fases. historial.csv anade `auc_val_mismo_hospital` (todo modelo) y, con atencion,
  `masa_caja_val` / `fraccion_caja_val` (atencion que cae en la caja frente a su area).
- **Analizar una iteracion:** `git pull`; para cada `results/<nombre>/` leer
  `historial.csv` y `metricas.json`. Comparar sobre todo el **AUC de validacion medio de
  todas las epocas** (el pico de una epoca es ruido, ±0,04) y el suavizado de la epoca
  elegida, mas la forma de las curvas (train y val juntas = sano; train sube y val cae =
  memoriza; train no sube = no aprende). Comparar SIEMPRE con la referencia en los
  mismos folds (base 128 px, `it5_red128_f0..f4`: AUC medio por fold 0,560 / 0,520 / 0,549 / 0,529 / 0,569;
  5 folds 0,545 ± 0,021). Criterio fijado: cribado en folds 0 y 1, pasa a 5 folds solo
  lo que mejora la media >= 0,03. Mirar `curvas.png` de los mejores.
- **Registrar** en `docs/iteraciones.md` (fila: hipotesis, modelos, resultado,
  decision) y crear `experimentos/itN+1.txt` (una linea = opciones de entrenar, con
  `--nombre itN_<idea>_f<fold>`). Antes de proponer commit: `ruff check .` y `pytest -q`.
- **Commits para el usuario:** un proposito por commit, en un bloque para copiar y pegar
  en Git Bash, encadenado con `&&`, con `git commit -q -m "..." -- <ficheros>` (asi solo
  entran esos ficheros), terminando en `git --no-pager log --stat --oneline -N && git
  status -s && echo "=== TODO BIEN: ya puedes hacer git push"`. El push lo hace el.
- **Tras cada iteracion**, recordar al usuario que suba `docs/iteraciones.md` (y
  `CLAUDE.md` si cambio) al Proyecto de claude.ai.
- **Pendiente despues de elegir el modelo:** umbral con criterio clinico (con las
  predicciones de validacion de los 5 folds), test UNA vez con todas las metricas de la
  rubrica, comparacion con el modelo clinico (0,69), app desplegada, informe, 5
  diapositivas. Entrega el 30-10-2026: la app no puede dejarse para el final.


**Este diseño es PROVISIONAL, no esta fijado.** El usuario lo dijo explicitamente:
el proyecto puede ir por otro lado. No tratarlo como decision tomada.

Lo que dejo el EDA (`notebooks/eda.ipynb`, 30-09-2026) y condiciona el diseño:
- Variables clinicas (subtipo + edad + volumen): AUC 0,69. Solo cohorte: 0,55.
  Medidas globales de realce: ~0,5. Es la referencia que la CNN tiene que superar.
- Las imagenes son la mama entera con orientacion variable. En el EDA la caja
  sraw/eraw/scol/ecol "tal cual" no caia en el tumor; despues se vio que SI lo marca,
  girada 90 grados y escalada (ver "Caja del tumor" arriba, `src/data/tumor.py`).
- LATE-EARLY depende del protocolo de cada cohorte.
- Cortes de una misma paciente muy correlacionados (0,81).
- Por eso realce y fase LATE van como ablaciones, no como decision.
- Variables clinicas: respondido en el correo del 14-10 (solo comparacion, no modelo
  principal).

## Como trabajar con el usuario

- Todo en scripts `.py`, salvo el EDA (Jupyter). Lo que pida calculo, siempre script.
- Nada de planes por dias: su dedicacion varia. Proponer el siguiente paso.
- Explicar el porque (el concepto) antes de implementar.
- `import torch` antes que matplotlib en scripts y cuadernos que dibujen (en el
  portatil, al reves, el kernel de Jupyter muere).
