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
dentro de la red, en `modelo.py`). Entrega hacia el 25-10-2026.

## Flujo portatil / laboratorio

- **Portatil** (Windows 11, i3, 8 GB, sin GPU): se escribe el codigo y se prueba con
  `python entrenar.py --rapido`. Entrenar de verdad aqui es inviable (~17 min/epoca).
- **Laboratorio** (Linux, GPU NVIDIA): entrenamiento real. Se prepara con
  `bash configurar.sh`; se puede dejar entrenando dia y noche (`nohup ... &` y
  bloquear pantalla, no cerrar sesion). **No se puede instalar nada** fuera de un
  entorno virtual: todo paquete va en `.venv`, y no hay sudo. Claude se usa desde
  el navegador, asi que alli Claude no ejecuta nada: el usuario pega codigo y salidas.

Todo script de entrenamiento: modo rapido y completo sin tocar codigo, detecta
cuda/cpu solo, sin rutas fijas (`config.py`), guarda en `resultados/` y `modelos/`
para commitearlo antes de irse del laboratorio.

**Seguridad en el laboratorio:** cada PC tiene una cuenta Linux "alumno" comun a
quien se siente en ese equipo, asi que el siguiente ve los ficheros. Lo que importa
es que no pueda hacer push al repo:
- git: `credential.helper` vacio y `user.*` solo con `--local` (lo hace `configurar.sh`).
  El push se autentica con un token fine-grained que el usuario pega cada vez.
- VS Code: `.vscode/settings.json` desactiva "Sign in with GitHub". No aceptarlo.
- Navegador: GitHub y claude.ai en ventana privada; cerrar sesion al irse.
- **Nunca pedir ni aceptar que se pegue el token en el chat**: queda guardado en
  el historial de la conversacion.

**Commits:** los hace siempre el usuario. No hacer `git commit` ni `git push` sin su
OK explicito para ese commit; preparar los cambios y proponer el mensaje.

## Estado del modelo

`modelo.py` + `entrenar.py`: CNN de 4 bloques (16-32-64-128), preprocesado sin pesos
dentro de la red (realce EARLY-PRE y LATE-EARLY opcional, `--sin-late` opcional),
GAP, dropout, un logit; agregacion por paciente fuera de la red.

**Este diseño es PROVISIONAL, no esta fijado.** El usuario lo dijo explicitamente:
el proyecto puede ir por otro lado. No tratarlo como decision tomada.

Lo que dejo el EDA (`eda.ipynb`, 30-09-2026) y condiciona el diseño:
- Variables clinicas (subtipo + edad + volumen): AUC 0,69. Solo cohorte: 0,55.
  Medidas globales de realce: ~0,5. Es la referencia que la CNN tiene que superar.
- Las imagenes son la mama entera con orientacion variable; las columnas
  sraw/eraw/scol/ecol NO sirven para recortar el tumor.
- LATE-EARLY depende del protocolo de cada cohorte.
- Cortes de una misma paciente muy correlacionados (0,81).
- Por eso realce y fase LATE van como ablaciones, no como decision.
- Pregunta abierta al profesor: ¿se pueden usar variables clinicas?

## Como trabajar con el usuario

- Todo en scripts `.py`, salvo el EDA (Jupyter). Lo que pida calculo, siempre script.
- Nada de planes por dias: su dedicacion varia. Proponer el siguiente paso.
- Explicar el porque (el concepto) antes de implementar.
- `import torch` antes que matplotlib en scripts y cuadernos que dibujen (en el
  portatil, al reves, el kernel de Jupyter muere).
