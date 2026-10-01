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

Tres ordenes. En Windows (el portatil):

```powershell
git clone https://github.com/Valdi183/ML_Caso_1_Cancer.git
cd ML_Caso_1_Cancer
.\configurar.ps1
```

En Linux (el equipo del laboratorio):

```bash
git clone https://github.com/Valdi183/ML_Caso_1_Cancer.git
cd ML_Caso_1_Cancer
bash configurar.sh                 # dataset en ../bdcedl; otra ruta con --datos
```

Los dos scripts hacen lo mismo. `configurar.sh` acepta `--datos`, `--sin-datos`
y `--torch`, y ademas deja git configurado para un equipo compartido (ver mas abajo).

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
python verificar_entorno.py
```

Revisa paquetes, GPU, las 38.109 imagenes, la particion sin fuga por paciente y
que las fases DCE se leen en el orden correcto. Ademas cronometra un entrenamiento
corto y te dice cuanto dura una epoca en esa maquina.

---

## Donde viven los datos

El dataset **no esta en el repositorio** y nunca debe estarlo: son 1,27 GB en
38.109 ficheros. Se descarga aparte y cada maquina lo guarda donde quiera.

```powershell
python descargar_datos.py --destino C:\bdcedl --hilos 32
```

Ningun script lleva la ruta escrita dentro. [`config.py`](config.py) la resuelve
sola, en este orden:

1. la variable de entorno `BDCEDL`
2. el fichero `.bdcedl_ruta` (lo escribe `configurar.ps1`, no se versiona)
3. rutas habituales: `./breastdcedl`, `C:\bdcedl`, `D:\bdcedl`, `~/bdcedl`,
   `/content/breastdcedl` en Colab

Asi que desde cualquier script o cuaderno del repositorio:

```python
import config

uc = config.utils()               # utils_caso.py ya importable
samples = uc.cargar_samples()
entrenamiento, validacion = uc.particion(samples, fold_val=0)
```

> **Elige una ruta corta y en disco local.** No OneDrive, no Dropbox, no unidad
> de red: 38.000 ficheros sincronizandose hacen el entrenamiento inutilizable, y
> las rutas profundas chocan con el limite de 260 caracteres de Windows.

---

## Estructura

```
ML_Caso_1_Cancer/
├── configurar.ps1          ← prepara una maquina Windows de cero
├── configurar.sh           ← lo mismo en Linux (laboratorio)
├── verificar_entorno.py    ← comprueba que todo esta listo
├── descargar_datos.py      ← baja las 38.109 imagenes (reanudable)
├── config.py               ← localiza el dataset sin rutas fijas
├── eda.ipynb               ← analisis exploratorio
├── modelo.py               ← la CNN y su preprocesado (unica definicion)
├── entrenar.py             ← entrena y valida por paciente (--rapido para probar)
├── CLAUDE.md               ← contexto del proyecto para el asistente
├── requirements.txt
├── modelos/                ← pesos entregables
├── resultados/             ← metricas, matrices de confusion, curvas
└── informe/figuras/        ← figuras del informe
```

El dataset aporta ademas `utils_caso.py`, que ya resuelve la particion por
paciente, el `Dataset` de PyTorch y la agregacion de cortes a paciente.

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

**No dejes credenciales.** El almacen de credenciales se desactiva solo para este
repositorio (`configurar.sh` lo hace solo; a mano son estas tres lineas):

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
nohup python entrenar.py --ponderada > entreno.log 2>&1 &
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
