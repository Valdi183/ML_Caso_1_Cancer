# Caso BreastDCEDL — prediccion de pCR

Red neuronal convolucional propia que, a partir de una resonancia de mama con
contraste (DCE-MRI) tomada **antes** de empezar la quimioterapia, predice si la
paciente alcanzara **pCR**: que tras el tratamiento no quede tumor invasivo.

No es deteccion de tumores. Es **pronostico de respuesta a un tratamiento que
todavia no ha empezado**, y por eso las metricas realistas rondan AUC 0,70-0,80.

| | train | test | total |
|---|---|---|---|
| pacientes | 1.097 | 176 | **1.273** |
| cortes | 10.945 | 1.758 | **12.703** |
| imagenes PNG | 32.835 | 5.274 | **38.109** |

El 70,6 % de los casos son pCR=0. Un modelo que prediga siempre la clase
mayoritaria acierta ese 70,6 % sin haber aprendido nada.

---

## Puesta en marcha en una maquina nueva

**Laboratorio (Linux, GPU AMD con ROCm):** los equipos ya vienen preparados, no
hay que instalar nada. Basta con clonar y configurar git para un equipo compartido
(ver [Trabajar en un equipo compartido](#trabajar-en-un-equipo-compartido)):

```bash
git clone https://github.com/Valdi183/ML_Caso_1_Cancer.git
cd ML_Caso_1_Cancer
```

**Windows (el portatil, o el PC de sobremesa con NVIDIA):** tres ordenes:

```powershell
git clone https://github.com/Valdi183/ML_Caso_1_Cancer.git
cd ML_Caso_1_Cancer
.\configurar.ps1
```

`configurar.ps1` crea el entorno virtual, instala las dependencias, **detecta si
hay GPU NVIDIA** y pone la rueda de PyTorch que corresponda, descarga las 38.109
imagenes y verifica que todo funciona. Tarda unos 20 minutos, casi todo descarga.

Es **idempotente**: si se corta, vuelve a lanzarlo y retoma donde estaba.

Opciones:

```powershell
.\configurar.ps1 -Datos D:\bdcedl     # otra carpeta para el dataset
.\configurar.ps1 -SinDatos            # solo el entorno de Python
.\configurar.ps1 -Torch cu118         # forzar una rueda concreta de PyTorch
```

Si PowerShell se niega a ejecutar el script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Eso afecta solo a esa ventana de terminal y no toca la configuracion del equipo.

### Comprobar que todo sigue bien

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.utils.verificar_entorno
```

Revisa paquetes, GPU, las 38.109 imagenes, la particion sin fuga por paciente y
que las fases DCE se leen en el orden correcto. Ademas cronometra un entrenamiento
corto y te dice cuanto dura una epoca en esa maquina.

---

## Donde viven los datos

El dataset **no esta en el repositorio** y nunca debe estarlo: son 1,27 GB en
38.109 ficheros. Se descarga aparte y cada maquina lo guarda donde quiera.

```powershell
python -m src.data.descargar --destino C:\bdcedl --hilos 32
```

Ningun script lleva la ruta escrita dentro. [`src/config.py`](src/config.py) la
resuelve sola, en este orden:

1. la variable de entorno `BDCEDL`
2. el fichero `.bdcedl_ruta` en la raiz (lo escribe `configurar.ps1`, no se versiona)
3. rutas habituales: `data/bdcedl`, `../bdcedl` (junto al repo), `C:\bdcedl`,
   `D:\bdcedl`, `~/bdcedl`, `/content/breastdcedl` en Colab

Asi que desde cualquier script, test o cuaderno del repositorio:

```python
from src import config

uc = config.utils()               # utils_caso.py ya importable
samples = uc.cargar_samples()
entrenamiento, validacion = uc.particion(samples, fold_val=0)
```

> **Elige una ruta corta y en disco local.** No OneDrive, no Dropbox, no unidad
> de red: 38.000 ficheros sincronizandose hacen el entrenamiento inutilizable, y
> las rutas profundas chocan con el limite de 260 caracteres de Windows.

---

## Estructura

Sigue la plantilla de la asignatura (UAX). Todo el codigo vive en `src/` y se
ejecuta **desde la raiz del repositorio** como modulo: `python -m src.<...>`.

```
ML_Caso_1_Cancer/
├── .github/workflows/ci.yml   ← CI: ruff + pytest en cada push
├── app/                       ← app web (Streamlit): app.py, ejemplos/, requirements.txt
├── data/                      ← sitio opcional para el dataset (no se versiona)
├── docs/figuras/              ← figuras del informe
├── docs/data/referencias.json ← referencias externas de la web (asignatura, Fridman, licencia)
├── models/                    ← pesos .pt (solo se versionan los que se entregan)
├── notebooks/eda.ipynb        ← analisis exploratorio
├── presentations/             ← diapositivas
├── results/<experimento>/     ← config, historial, metricas, curvas de cada entreno
├── src/
│   ├── config.py              ← carpetas, semilla e hiperparametros, dataset
│   ├── data/descargar.py      ← baja las 38.109 imagenes (reanudable)
│   ├── data/carga.py          ← cortes en memoria, lotes, aumentado
│   ├── data/imagen.py         ← lectura de un PNG (la misma en entrenamiento y app)
│   ├── models/cnn.py          ← la CNN y su preprocesado (unica definicion)
│   ├── inferencia.py          ← logica de la app: modelo, validacion de la entrada, prediccion
│   ├── visualizacion.py       ← lo que la app enseña alrededor: mapa, curva, escala, cifras
│   ├── training/entrenar.py   ← entrena y valida por paciente (--rapido para probar)
│   └── utils/                 ← semilla, metricas, curvas, umbral, test, combinado, saturacion
├── tests/                     ← tests de la red (no necesitan el dataset)
│   └── test_visualizacion.py  ← tests de src/visualizacion.py
├── experimentos/              ← una lista de entrenamientos por iteracion + lanzar.sh / ver.sh
├── configurar.ps1             ← prepara el portatil Windows de cero
├── CLAUDE.md                  ← contexto del proyecto para el asistente
├── requirements.txt, ruff.toml, pytest.ini
```

`notebooks/comparativa_modelos.ipynb` reune las curvas de todas las iteraciones y las figuras del modelo final (calculos en `src/utils/figuras.py`; guarda `docs/figuras/comparativa_*.png` y `results/comparacion/inventario.csv`): se abre con Jupyter desde la raiz (`.venv` necesita `pip install jupyter`) y se ejecuta entero, ~2,5 min en CPU; no usa test.

El dataset aporta ademas `utils_caso.py`, que ya resuelve la particion por
paciente, el `Dataset` de PyTorch y la agregacion de cortes a paciente.

**Entorno:** `venv` + `pip` con `requirements.txt` (la opcion 2 de la plantilla).
PyTorch se instala aparte porque su rueda depende de la GPU (ver `requirements.txt`).

---

## Entrenar

```bash
python -m src.training.entrenar --rapido                       # prueba de minutos (CPU)
python -m src.training.entrenar --ponderada --aumentado        # entrenamiento real
python -m src.training.entrenar --help                         # todas las opciones
bash experimentos/lanzar.sh it2                               # con GPU: la lista experimentos/it2.txt
bash experimentos/ver.sh it2                                  # volver a ver su progreso
```

`lanzar.sh` funciona en el laboratorio (Linux) y en Windows desde **Git Bash**
(el PC con NVIDIA): usa el Python de `.venv` en los dos casos. En Windows, que el
equipo no se suspenda mientras entrena (Configuracion → Energia → "Nunca").

La semilla y los hiperparametros por defecto (`epocas`, `lote`, `lr`, `wd`,
`paciencia`, `min_epocas`, `suavizado`) estan en `src/config.py` (`DEFECTO`). Cada experimento guarda su
configuracion exacta en `results/<nombre>/config.json`. Las iteraciones del modelo
(hipotesis, resultado y decision de cada una) se registran en `docs/iteraciones.md`.

Los pesos van a `models/<nombre>.pt`, que git ignora. Solo los que se entregan
se suben, a proposito: `git add -f models/<nombre>.pt`.

---

## Modelo final: como reproducirlo

CNN de 3 bloques (8-16-32), resumen por media, imagen reducida a 128 px dentro de
la red, 18.329 parametros, perdida normal, semilla 42 (`config.DEFECTO`). Es un
conjunto de 5 modelos, uno por fold de validacion. Por que esta configuracion:
[`docs/decisiones.md`](docs/decisiones.md); todo lo probado:
[`docs/iteraciones.md`](docs/iteraciones.md).

```bash
O="--aumentado --dropout 0.7 --wd 1e-3 --reducir 2 --filtros 8,16,32"
python -m src.training.entrenar $O --fold 0 --nombre it6_3bloques_f0
python -m src.training.entrenar $O --fold 1 --nombre it6_3bloques_f1
python -m src.training.entrenar $O --fold 2 --nombre it8_3bloques_f2
python -m src.training.entrenar $O --fold 3 --nombre it8_3bloques_f3
python -m src.training.entrenar $O --fold 4 --nombre it8_3bloques_f4
```

(Son las lineas de `experimentos/it6.txt` y `experimentos/it8.txt`; con GPU,
`bash experimentos/lanzar.sh it8`. La comparacion con perdida ponderada son las
mismas ordenes con `--ponderada` y `--nombre it8_3bloques_pond_f<fold>`.)

Cada ejecucion deja `models/<nombre>.pt` y `results/<nombre>/` (`config.json`,
`historial.csv`, `metricas.json`, `predicciones_val.csv`, `curvas.png`). Los pesos
se cargan con `src.models.cnn.cargar`, que reconstruye la red con su preprocesado.

El resumen fuera de fold de una configuracion (sin entrenar, sin tocar test):

```bash
python -m src.utils.analisis_cv --config 3bloques it6_3bloques_f0 it6_3bloques_f1 \
    it8_3bloques_f2 it8_3bloques_f3 it8_3bloques_f4
```

**Umbral: 0,3635** (`python -m src.utils.umbral`, campo `umbral_final` de
`results/umbral/resumen.json`). Criterio: especificidad >= 0,90 por paciente (media
de sus cortes); regla: percentil 90 de la probabilidad de todas las pacientes pCR=0
fuera de fold juntas. Detalle y justificacion en `docs/decisiones.md`.

**Test (hecho una vez):** `python -m src.utils.evaluar_test` evalua el conjunto de los 5
modelos con el umbral fijado y se niega a repetirse si ya existe
`results/test/resumen.json` (el plan, con el SHA-256 de cada peso, esta en
`results/test/plan.json`). Por paciente: AUC 0,544 [0,457-0,629], sensibilidad 0,038,
especificidad 0,935, F1 macro 0,430. La meta realista es el modelo de referencia de la
asignatura, con AUC ≈ 0,64 en test; Fridman et al. 2025 (0,72 en el mismo test) usa un modelo preentrenado,
así que es una cota superior y no es comparable con una CNN desde cero.
Detalle en `docs/decisiones.md`; figuras en `docs/figuras/test_*.png`.

---

## App

**URL:** _(pendiente de desplegar)_

Recibe **un corte**: tres PNG en escala de grises de 256×256 (PRE, EARLY y LATE, máximo
2 MB cada uno), o uno de los ejemplos de `app/ejemplos/` (pacientes de train, nunca de
test). Muestra:

- **Visor:** las tres fases y un mapa de captación sobre EARLY, con la curva de captación
  de la zona que más realza (descriptivos: el modelo no los usa).
- **Resultado:** la probabilidad de pCR y la clase con el umbral, en una escala con el
  umbral, la tasa base y la probabilidad de cada uno de los 5 modelos; el SHA-256 de los
  pesos, el dispositivo y el tiempo de respuesta, y la descarga del resultado en JSON.
- **«¿Cuánto fiarse?»:** el AUC de test, cuántas de las marcadas como pCR lo son
  (validación y test) y cuántas pCR detecta en test, leído de `results/`.
- **Imagen frente a variables clínicas:** el AUC de validación de la clínica sola, la
  imagen sola y las dos combinadas, solo como comparación (no pide datos clínicos).
- Desplegables con cómo funciona, cómo se validó y los detalles técnicos.

Uso educativo: **sin validez clínica**.

La interfaz está en `app/app.py`. Lo que se enseña alrededor de la predicción está en
`src/visualizacion.py` (testeado en `tests/test_visualizacion.py`): el mapa y la curva de
captación, la escala del resultado, las cifras de «¿Cuánto fiarse?» (leídas de
`results/umbral/` y `results/test/`), los mensajes de error y el JSON descargable.

El mapa y la curva son descriptivos: el modelo no los usa. El tipo de curva sale del
cambio de EARLY a LATE: más de +10 %, persistente; ±10 %, meseta; menos de −10 %, lavado.
Donde EARLY o LATE saturan, solo se afirma el tipo que la cota permite. El momento de LATE
varía según el hospital.

Si existe `results/combinado/resumen.json` (`python -m src.utils.combinado`), la app
muestra el AUC de validación de la clínica sola, la imagen sola y las dos combinadas, solo
como comparación: no pide datos clínicos. Si existe `docs/figuras/arquitectura.png`
(`python -m src.utils.arquitectura`), muestra el esquema de la red.

El tema (gris de visor y Atkinson Hyperlegible desde Google Fonts) está en
`.streamlit/config.toml`. La app escribe la probabilidad con coma decimal;
`python -m src.inferencia` la imprime con punto, con los mismos dígitos.

- **Modelo:** el conjunto de los 5 pesos de `results/test/plan.json` (comprueba su
  SHA-256 al arrancar), en CPU, en modo evaluación y sin gradientes. Probabilidad = media
  de las 5 sigmoides, como en la evaluación de test.
- **Umbral:** 0,3635, leído de `results/umbral/resumen.json` (`umbral_final`). Si falta, la
  app no arranca. No usa el `umbral` guardado en los `.pt` (es el de Youden de cada fold).
- **Mismo preprocesado que el entrenamiento:** el PNG se lee con `src/data/imagen.py`, la
  función que usa `cargar_en_memoria`, y la red recibe `uint8 / 255`; el resto va dentro
  de la red. `tests/test_imagen.py` comprueba que la lectura da exactamente lo mismo que la
  de `utils_caso.py`, y `tests/test_app.py`, que la predicción de la app coincide (1e-6)
  con la del pipeline. La app no necesita la carpeta del dataset.
- **Entrada:** rechaza, con un mensaje claro, lo que no sea PNG, otro tamaño (no
  redimensiona), más de 2 MB, ficheros corruptos, color, 16 bits y valores no finitos.
  Avisa, sin bloquear, si EARLY no es más brillante que PRE (fases intercambiadas) o si el
  sufijo `_PRE`/`_EARLY`/`_LATE` del nombre no cuadra con la casilla. Solo trabaja con los
  bytes subidos: ni rutas ni ejecución de nada.

En local, desde la raíz y con el `.venv`:

```bash
pip install streamlit          # una vez; el resto ya está en el entorno
streamlit run app/app.py       # http://localhost:8501
```

**Despliegue (Streamlit Community Cloud):** desde el repositorio de GitHub, fichero
principal `app/app.py`, Python 3.12. Las dependencias del servidor están en
`app/requirements.txt` (PyTorch de CPU, versiones fijadas); `.streamlit/config.toml` limita
la subida a 2 MB por fichero. Los 5 pesos y los JSON de `results/` que usa ya están en el
repositorio.

**Comprobar la app con el pipeline** (en la defensa): la misma probabilidad, con los mismos
decimales que muestra la app:

```bash
python -m src.inferencia --pre X_PRE.png --early X_EARLY.png --late X_LATE.png
```

**Nota numérica:** las cifras oficiales (validación, umbral y test) se calcularon en GPU
con TF32, que PyTorch activa por defecto en cuDNN. La predicción (`src.inferencia`: app,
orden anterior y validación al entrenar) va en FP32, con lo que GPU y CPU dan lo mismo; las
probabilidades cambian hasta ~7e-4 frente a las oficiales. En validación
(`python -m src.utils.fp32_val`, `results/fp32_val/`) el AUC cambia en la cuarta decimal y
ninguna paciente cambia de clase con 0,3635. En test hay una paciente a 1,9e-4 del umbral
cuya clase podría cambiar; no se recalcula para no volver a usar el test. Detalle en
`docs/decisiones.md`.

---

## Web estática (GitHub Pages)

La app de Streamlit sigue de respaldo. La web estática (`web/`) es a la vez la presentación del caso y la
app: la predicción se hace **en el navegador** con ONNX Runtime Web (WebAssembly, 1 hilo), las imágenes no
salen del ordenador y no hace falta servidor. Todo va en el repositorio, sin CDN ni nada externo en
tiempo de ejecución (`web/js/vendor/LEEME.md`; tipografía Atkinson Hyperlegible, SIL OFL, en `web/fonts/`).

- `web/index.html`: **el caso explicado**, con métricas: portada con un corte real (PRE → EARLY), el
  problema, los datos, el modelo, cómo se eligió, resultados en test, comparación y límites, créditos.
  Los gráficos son SVG dibujados desde los datos (`web/js/graficos.js`).
- `web/predecir.html`: **la predicción**, con lo que hace la app de Streamlit: visor de las tres fases, mapa
  y curva de captación (descriptivos), resultado con su escala y los 5 modelos, «¿cuánto fiarse?»,
  comparación con las clínicas, detalles técnicos, descarga en JSON y «Comprobar con el pipeline».
- **Ninguna cifra está escrita en el HTML ni en el JS.** Salen de `web/datos.json` (todas las cifras de las
  dos páginas) y de `web/modelo.json` (modelo, SHA-256, referencias de PyTorch), que genera
  `python -m src.web.exportar` desde `results/`, la red y `docs/data/referencias.json` (la referencia de la
  asignatura, Fridman et al. y la licencia: lo que no sale de `results/`). `--solo-datos` regenera solo
  `datos.json`, sin volver a exportar el ONNX. Un test comprueba que está al día y que cada `data-v` del
  HTML apunta a un dato que existe.
- `web/js/png.js` lee el PNG sin `canvas` (gestión de color), con las mismas reglas y mensajes que
  `src/inferencia.py`; `web/js/visualizacion.js` es el mapa y la curva de `src/visualizacion.py`.
- **Exportar** (solo si cambia el modelo o `results/`; `pip install -r requirements-web.txt`): el ONNX
  (`web/modelo/conjunto.onnx`, los 5 pesos del plan, opset 17), `web/modelo.json`, `web/datos.json` y los
  ejemplos con su licencia. El `.onnx` está ignorado por git salvo en `web/modelo/`.
- **Probar en local:** `python -m http.server` dentro de `web/` y abrir <http://localhost:8000/>.
- **Comprobar:** `pytest -q` (ONNX frente a PyTorch FP32, < 1e-5 y misma clase, en los ejemplos y en 51
  cortes de train; cifras y ficheros de prueba al día) y, con navegadores reales,
  `python -m src.web.probar_navegador` (`pip install playwright && python -m playwright install firefox`;
  Chrome es el instalado): la lectura del PNG da lo mismo que Python en 76 casos, el mapa y la curva en 24
  cortes, y los 4 ejemplos coinciden con `python -m src.inferencia` a 4 decimales. Los casos los generan
  `python -m src.web.casos_lectura` y `python -m src.web.casos_visualizacion` (en `web/pruebas/`).
  `python -m src.web.capturas CARPETA` hace capturas a 1440, 1024 y 390 px y avisa de errores de consola y
  de desbordes.
- **Despliegue:** `.github/workflows/pages.yml` publica `web/` en cada push a `main` que lo toque. En
  GitHub: *Settings → Pages → Source: GitHub Actions*.

---

## Tests y CI

```bash
pip install ruff pytest
ruff check .        # estilo y errores comunes
pytest -q           # tests de la red: formas, normalizacion, atencion, guardar/cargar
```

GitHub ejecuta lo mismo en cada push a `main` (pestaña *Actions* del repositorio).

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/),
en imperativo: `feat(models): añadir pooling con atencion`, `fix(data): ...`,
`docs: ...`, `chore: ...`. Un commit, un proposito.

---

## Reglas del caso

Invalidan el trabajo:

1. **Mezclar cortes de una misma paciente** entre entrenamiento y validacion. La
   unidad estadistica es la paciente, no el corte. Usa la columna `fold`.
2. **Usar `test` para ajustar** el modelo o los hiperparametros. Se toca una sola
   vez, al final.
3. **Usar modelos preentrenados** o transfer learning. La CNN es propia, desde
   cero, con una unica salida.
4. **Publicar la validacion privada** o sus etiquetas.
5. El trabajo es **individual**.

Errores que salen caros:

- Tratar los 3 canales como RGB. Son PRE, EARLY y LATE: tres instantes del
  contraste. Nada de normalizacion de ImageNet ni `ColorJitter`.
- Aumentado canal a canal. Desalinea las fases y destruye el realce, que es
  justo la señal. Toda transformacion geometrica va al tensor completo.
- Confundir `z012` con un instante de tiempo. Es una **altura** axial; el tiempo
  lo da el sufijo del fichero.
- Evaluar con `shuffle=True`. Las probabilidades dejan de corresponder a las filas.
- Normalizar distinto en la app web que en el entrenamiento.

---

## Trabajar en un equipo compartido

El equipo del laboratorio es de uso publico. Dos consecuencias:

**Nada importante vive solo alli.** Todo lo que produzcas —pesos, metricas,
figuras, cuadernos— se commitea y se sube antes de levantarte de la silla. Asume
que la sesion puede borrarse.

**No dejes credenciales.** Nada mas clonar, desactiva el almacen de credenciales
solo para este repositorio y pon tu autor:

```bash
git config --local credential.helper ""
git config --local user.name  "Valdi183"
git config --local user.email "vvaldcal@myuax.com"
git config --show-origin --list | grep -E "credential|user\."   # comprobarlo
```

`--local` escribe en `.git/config`, dentro de la carpeta. Nunca uses `--global`
en una maquina compartida: eso queda en el perfil del equipo.

Para subir, autentica con un **token personal** (GitHub ya no acepta contraseña):
GitHub → Settings → Developer settings → Personal access tokens → Fine-grained,
con `Contents: Read and write` solo sobre este repositorio y caducidad corta. Al
hacer `git push` pide usuario (`Valdi183`) y contraseña: ahi se pega el token. Con
el helper vacio no se guarda en ningun sitio, y lo pedira en cada push.

**VS Code:** `.vscode/settings.json` desactiva el "Sign in with GitHub", que daria
acceso a toda la cuenta y quedaria guardado en el equipo. Si VS Code pregunta si
confias en la carpeta, di que si (si no, ignora ese ajuste). Si aun asi aparece un
aviso de iniciar sesion con GitHub, cancelalo.

**Entrenamientos largos:** lanzalos para que sobrevivan a cerrar la terminal, y al
irte bloquea la pantalla en vez de cerrar sesion:

```bash
nohup python -m src.training.entrenar --ponderada > entreno.log 2>&1 &
tail -f entreno.log          # Ctrl+C sale del tail, el entrenamiento sigue
```

**Navegador:** GitHub, claude.ai o cualquier otra cuenta, siempre en ventana
privada, y sin dejar que el navegador guarde contraseñas.

Al terminar la sesion: `git push` y cierra la ventana privada (eso cierra las
sesiones de GitHub y claude.ai). El repo y el dataset pueden quedarse en tu
carpeta: quien se siente despues los vera, pero sin credenciales guardadas no puede
subir nada a tu repositorio.

---

## Licencia y aviso

Imagenes desidentificadas derivadas de **BreastDCEDL**, licencia **CC BY-NC 4.0**.
Uso docente e investigador exclusivamente.

**Esto no es un dispositivo medico.** No se permite reidentificacion, diagnostico
ni recomendacion terapeutica. Un buen resultado retrospectivo no demuestra
seguridad, causalidad ni utilidad clinica prospectiva.
