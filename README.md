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

**Windows (el portatil):** tres ordenes:

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
├── app/                       ← demo web (pendiente)
├── data/                      ← sitio opcional para el dataset (no se versiona)
├── docs/figuras/              ← figuras del informe
├── models/                    ← pesos .pt (solo se versionan los que se entregan)
├── notebooks/eda.ipynb        ← analisis exploratorio
├── presentations/             ← diapositivas
├── results/<experimento>/     ← config, historial, metricas, curvas de cada entreno
├── src/
│   ├── config.py              ← carpetas, semilla e hiperparametros, dataset
│   ├── data/descargar.py      ← baja las 38.109 imagenes (reanudable)
│   ├── data/carga.py          ← cortes en memoria, lotes, aumentado
│   ├── models/cnn.py          ← la CNN y su preprocesado (unica definicion)
│   ├── training/entrenar.py   ← entrena y valida por paciente (--rapido para probar)
│   └── utils/                 ← semilla, metricas y curvas, verificar_entorno
├── tests/                     ← tests de la red (no necesitan el dataset)
├── cola.sh                    ← varios entrenamientos seguidos (laboratorio)
├── configurar.ps1             ← prepara el portatil Windows de cero
├── CLAUDE.md                  ← contexto del proyecto para el asistente
├── requirements.txt, ruff.toml, pytest.ini
```

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
nohup bash cola.sh > cola.log 2>&1 &                           # la cola del laboratorio
```

La semilla y los hiperparametros por defecto (`epocas`, `lote`, `lr`, `wd`,
`paciencia`) estan en `src/config.py` (`DEFECTO`). Cada experimento guarda su
configuracion exacta en `results/<nombre>/config.json`.

Los pesos van a `models/<nombre>.pt`, que git ignora. Solo los que se entregan
se suben, a proposito: `git add -f models/<nombre>.pt`.

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
