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

Tres ordenes. Funciona igual en el portatil y en el equipo del laboratorio.

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
├── configurar.ps1          ← prepara una maquina nueva de cero
├── verificar_entorno.py    ← comprueba que todo esta listo
├── descargar_datos.py      ← baja las 38.109 imagenes (reanudable)
├── config.py               ← localiza el dataset sin rutas fijas
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

**No dejes credenciales.** Al clonar, desactiva el almacen de credenciales solo
para este repositorio:

```powershell
git clone https://github.com/Valdi183/ML_Caso_1_Cancer.git
cd ML_Caso_1_Cancer
git config --local credential.helper ""
git config --local user.name  "Valdi183"
git config --local user.email "vvaldcal@myuax.com"
```

`--local` escribe en `.git/config`, que desaparece con la carpeta. Nunca uses
`--global` en una maquina compartida: eso queda en el perfil del equipo.

Para subir, autentica con un **token personal** (GitHub ya no acepta contraseña):
GitHub → Settings → Developer settings → Personal access tokens → Fine-grained,
con permiso de escritura solo sobre este repositorio y caducidad corta. Lo pegas
como contraseña cuando `git push` lo pida. Con el helper desactivado no se guarda
en ningun sitio.

Al terminar la sesion: borra la carpeta del repositorio y la del dataset, y
cierra sesion en el navegador.

---

## Licencia y aviso

Imagenes desidentificadas derivadas de **BreastDCEDL**, licencia **CC BY-NC 4.0**.
Uso docente e investigador exclusivamente.

**Esto no es un dispositivo medico.** No se permite reidentificacion, diagnostico
ni recomendacion terapeutica. Un buen resultado retrospectivo no demuestra
seguridad, causalidad ni utilidad clinica prospectiva.
