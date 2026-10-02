#!/usr/bin/env bash
# Prepara el entorno del caso BreastDCEDL en una maquina Linux (el laboratorio).
#
# Equivalente a configurar.ps1: credenciales de git solo para este repositorio,
# entorno virtual, dependencias, PyTorch con la rueda que corresponda a la GPU
# (ROCm si es AMD, CUDA si es NVIDIA, CPU si no hay ninguna), dataset descargado
# y verificacion final. No necesita sudo ni instala nada fuera de .venv: solo usa
# git, python3 y, si existen, /dev/kfd (AMD) o nvidia-smi (NVIDIA).
#
# Es idempotente: si algo ya esta hecho, lo detecta y sigue. Si se corta a
# medias, se relanza y retoma donde estaba.
#
#     bash configurar.sh                         # dataset en ../bdcedl, junto al repo
#     bash configurar.sh --datos ~/otra/bdcedl
#     bash configurar.sh --sin-datos             # solo el entorno de Python
#     bash configurar.sh --torch rocm7.2         # forzar rueda: rocmX.Y, cu128, cu126, cu118 o cpu

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATOS="$(dirname "$REPO")/bdcedl"
SIN_DATOS=0
TORCH=""

# Ruedas ROCm de PyTorch: la que usa el laboratorio y otra de reserva por si falla.
# Las que existen: https://download.pytorch.org/whl/torch/ (buscar "+rocm")
ROCM_DEFECTO="rocm7.14"
ROCM_RESERVA="rocm7.2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datos)     DATOS="$2"; shift 2 ;;
        --sin-datos) SIN_DATOS=1; shift ;;
        --torch)     TORCH="$2"; shift 2 ;;
        -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
        *)           echo "Opcion desconocida: $1 (mira --help)"; exit 1 ;;
    esac
done
if ! [[ "$TORCH" =~ ^(|cu128|cu126|cu118|cpu|rocm[0-9]+\.[0-9]+)$ ]]; then
    echo "--torch debe ser rocmX.Y (p. ej. $ROCM_DEFECTO), cu128, cu126, cu118 o cpu"; exit 1
fi
DATOS="$(realpath -m "$DATOS")"

paso()  { printf '\n\033[36m== %s\033[0m\n' "$1"; }
bien()  { printf '   \033[32mOK\033[0m  %s\n' "$1"; }
aviso() { printf '   \033[33m!\033[0m   %s\n' "$1"; }
falla() { printf '\n\033[31mFALLA\033[0m %s\n' "$1"; exit 1; }

# Familia de una rueda: rocm7.14 -> rocm, cu126 -> cuda, cpu -> cpu
familia() { case "$1" in rocm*) echo rocm ;; cu*) echo cuda ;; *) echo cpu ;; esac; }

# Chip AMD segun el kernel (gfx_target_version, p. ej. 100301 = gfx1031). Se lee de
# /sys porque ahi no influye HSA_OVERRIDE_GFX_VERSION, a diferencia de rocminfo.
gfx_amd() {
    local f v
    for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        v="$(awk '$1 == "gfx_target_version" {print $2}' "$f" 2>/dev/null || true)"
        if [[ -n "$v" && "$v" != "0" ]]; then echo "$v"; return 0; fi   # 0 = nodo de CPU
    done
    return 1
}
nombre_gfx() { printf 'gfx%d%d%x' $(($1 / 10000)) $(($1 / 100 % 100)) $(($1 % 100)); }

echo
echo "Configuracion del caso BreastDCEDL"
echo "Repositorio: $REPO"

# --------------------------------------------------------------------------- #
paso "1/7  Git: credenciales solo para este repositorio"
# --------------------------------------------------------------------------- #
# El equipo es compartido. Con el helper vacio en .git/config, git pide el token
# en cada push y no lo guarda en ningun sitio, aunque el sistema tenga un almacen
# de credenciales configurado. --local escribe dentro del repo, nunca en el perfil.
git -C "$REPO" config --local credential.helper ""
git -C "$REPO" config --local user.name  "Valdi183"
git -C "$REPO" config --local user.email "vvaldcal@myuax.com"
bien "credential.helper vacio, autor Valdi183 (solo en .git/config)"
if git config --global --get user.name >/dev/null 2>&1; then
    aviso "ojo: el perfil del equipo tiene un user.name global ($(git config --global --get user.name))."
    aviso "No afecta a este repo, pero no es tuyo: no lo toques."
fi

# --------------------------------------------------------------------------- #
paso "2/7  Buscando Python"
# --------------------------------------------------------------------------- #
PY=""
for cmd in python3 python; do
    command -v "$cmd" >/dev/null 2>&1 || continue
    if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$cmd"
        bien "$cmd -> $("$cmd" --version 2>&1)"
        break
    fi
    aviso "$cmd es $("$cmd" --version 2>&1): hace falta 3.10 o superior"
done
[[ -n "$PY" ]] || falla "No hay un Python 3.10+ en el PATH. Pide en el laboratorio cual usar (module load, conda...)."

# --------------------------------------------------------------------------- #
paso "3/7  Entorno virtual"
# --------------------------------------------------------------------------- #
VENV="$REPO/.venv"
PYV="$VENV/bin/python"

if [[ -x "$PYV" ]] && "$PYV" -m pip --version >/dev/null 2>&1; then
    bien ".venv ya existe, lo reutilizo (borralo si quieres empezar de cero)"
else
    rm -rf "$VENV"
    echo "   creando .venv ..."
    if "$PY" -m venv "$VENV" >/dev/null 2>&1; then
        bien ".venv creado"
    else
        # En Ubuntu/Debian el paquete python3-venv suele faltar y sin sudo no se
        # puede instalar. Se crea el venv sin pip y se le pone pip a mano.
        aviso "este Python no trae ensurepip (falta python3-venv); creo el venv sin pip"
        rm -rf "$VENV"
        "$PY" -m venv --without-pip "$VENV" || falla "no se pudo crear el entorno virtual"
        # Con el propio Python: en el laboratorio no se puede contar con curl ni wget
        "$PY" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
            https://bootstrap.pypa.io/get-pip.py "$VENV/get-pip.py" \
            || falla "no se pudo descargar get-pip.py"
        "$PYV" "$VENV/get-pip.py" --quiet || falla "no se pudo instalar pip en el venv"
        rm -f "$VENV/get-pip.py"
        bien ".venv creado (pip instalado con get-pip.py)"
    fi
fi

"$PYV" -m pip install --quiet --upgrade pip setuptools wheel || falla "no se pudo actualizar pip"
bien "pip actualizado"

# --------------------------------------------------------------------------- #
paso "4/7  Dependencias"
# --------------------------------------------------------------------------- #
"$PYV" -m pip install --disable-pip-version-check -r "$REPO/requirements.txt" \
    || falla "no se pudieron instalar las dependencias. El error deberia estar justo arriba."
"$PYV" -c "import numpy, pandas, PIL, matplotlib, sklearn, requests, tqdm" \
    || falla "las dependencias se instalaron pero no se pueden importar"
bien "numpy, pandas, pillow, matplotlib, scikit-learn, requests, tqdm"

# --------------------------------------------------------------------------- #
paso "5/7  PyTorch"
# --------------------------------------------------------------------------- #
# Que rueda pide esta maquina. AMD va primero: sin nvidia-smi no significa sin GPU.
GFX=""
if gfx="$(gfx_amd)"; then GFX="$gfx"; fi

if [[ -z "$TORCH" ]]; then
    if [[ -e /dev/kfd ]]; then
        TORCH="$ROCM_DEFECTO"
        bien "GPU AMD detectada (/dev/kfd)${GFX:+, chip $(nombre_gfx "$GFX")}: rueda $TORCH"
        if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
            aviso "no tienes permiso sobre /dev/kfd: tu usuario debe estar en el grupo render (orden groups)"
        fi
    elif ! command -v nvidia-smi >/dev/null 2>&1; then
        TORCH="cpu"
        aviso "ni /dev/kfd ni nvidia-smi: no hay GPU AMD ni NVIDIA (o no hay driver), instalo la rueda CPU"
    else
        cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1 || true)"
        if [[ -n "$cuda" ]]; then
            mayor="${cuda%%.*}"; menor="${cuda#*.}"
            bien "GPU NVIDIA detectada, driver con CUDA $cuda"
            if   (( mayor >= 13 ));                 then TORCH="cu128"
            elif (( mayor == 12 && menor >= 8 ));   then TORCH="cu128"
            elif (( mayor == 12 ));                 then TORCH="cu126"
            else                                         TORCH="cu118"
            fi
        else
            TORCH="cu126"
            aviso "no pude leer la version de CUDA en nvidia-smi, pruebo con cu126"
        fi
    fi
fi
DESEADA="$(familia "$TORCH")"

# Si ya hay torch, se comprueba que sea de la familia correcta (una rueda CPU o CUDA
# en una maquina AMD funciona, pero entrena sin GPU sin avisar).
INSTALAR=1
if instalada="$("$PYV" -c 'import torch; print("rocm" if torch.version.hip else "cuda" if torch.version.cuda else "cpu")' 2>/dev/null | tail -n 1)" \
        && [[ -n "$instalada" ]]; then
    version="$("$PYV" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -n 1)"
    if [[ "$instalada" == "$DESEADA" ]]; then
        bien "torch $version ya instalado (borra .venv si quieres cambiar de rueda)"
        INSTALAR=0
    else
        aviso "torch $version es la rueda $instalada, pero esta maquina pide $DESEADA: la cambio"
        "$PYV" -m pip uninstall --quiet -y torch || falla "no se pudo desinstalar la rueda anterior"
    fi
fi

if (( INSTALAR )); then
    candidatas=("$TORCH")
    if [[ "$DESEADA" == "rocm" && "$TORCH" != "$ROCM_RESERVA" ]]; then
        candidatas+=("$ROCM_RESERVA")
    fi
    instalado=0
    for rueda in "${candidatas[@]}"; do
        echo "   instalando torch ($rueda), esto tarda unos minutos ..."
        if "$PYV" -m pip install --disable-pip-version-check torch \
                --index-url "https://download.pytorch.org/whl/$rueda"; then
            instalado=1
            TORCH="$rueda"
            break
        fi
        aviso "fallo la rueda $rueda"
    done
    if (( ! instalado )); then
        # La rueda de PyPI por defecto es la de NVIDIA: en una maquina AMD no usaria la GPU
        if [[ "$DESEADA" == "rocm" ]]; then
            falla "no se pudo instalar PyTorch para ROCm. Mira que ruedas rocmX.Y existen en
      https://download.pytorch.org/whl/torch/ y relanza con --torch rocmX.Y"
        fi
        aviso "reintento con la de PyPI por defecto"
        "$PYV" -m pip install --disable-pip-version-check torch || falla "no se pudo instalar PyTorch"
    fi
    version="$("$PYV" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -n 1)" \
        || falla "PyTorch se instalo pero no se puede importar"
    bien "torch $version instalado"
fi

# Las Radeon RX 6000 que no son gfx1030 (6700 XT = gfx1031, etc.) no estan soportadas
# oficialmente por ROCm, pero funcionan con los kernels de gfx1030. Si el sistema ya
# fija HSA_OVERRIDE_GFX_VERSION (en el laboratorio lo hace /etc/environment) se respeta;
# si no, se fija en el activate del .venv para que llegue tambien a nohup.
if [[ "$DESEADA" == "rocm" ]]; then
    if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
        bien "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION ya viene del sistema"
    elif [[ -n "$GFX" ]] && (( GFX > 100300 && GFX < 100400 )); then
        if ! grep -q 'HSA_OVERRIDE_GFX_VERSION' "$VENV/bin/activate"; then
            {
                echo
                echo "# Anadido por configurar.sh: ROCm no soporta oficialmente $(nombre_gfx "$GFX");"
                echo "# se usan los kernels de gfx1030, de la misma familia (RDNA2)."
                echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0"
            } >> "$VENV/bin/activate"
        fi
        export HSA_OVERRIDE_GFX_VERSION=10.3.0
        bien "chip $(nombre_gfx "$GFX"): HSA_OVERRIDE_GFX_VERSION=10.3.0 fijado en .venv/bin/activate"
    fi
fi

gpu="$("$PYV" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "SIN GPU")' \
        2>/dev/null | tail -n 1 || true)"
if [[ -z "$gpu" || "$gpu" == "SIN GPU" ]]; then
    case "$DESEADA" in
        rocm)
            aviso "PyTorch (ROCm) no ve la GPU AMD. Comprueba que estas en el grupo render (orden groups)"
            aviso "y, si el chip no es gfx1030, el valor de HSA_OVERRIDE_GFX_VERSION." ;;
        cuda)
            aviso "PyTorch no ve ninguna GPU. Si esta maquina tiene una NVIDIA, borra .venv"
            aviso "y relanza con --torch cu128 / cu126 / cu118 segun el CUDA que diga nvidia-smi." ;;
        *)
            aviso "PyTorch funcionara solo con CPU: sirve para --rapido, no para entrenar de verdad." ;;
    esac
else
    bien "PyTorch usa la GPU: $gpu (rueda $(familia "$TORCH"))"
fi

# --------------------------------------------------------------------------- #
paso "6/7  Dataset"
# --------------------------------------------------------------------------- #
if (( SIN_DATOS )); then
    aviso "omitido por --sin-datos"
else
    if [[ -f "$DATOS/metadata/samples.csv" ]]; then
        bien "ya hay dataset en $DATOS, compruebo que este completo"
    else
        echo "   descargando 38.109 imagenes (1,27 GB) en $DATOS ..."
    fi
    "$PYV" "$REPO/descargar_datos.py" --destino "$DATOS" --hilos 32 \
        || falla "la descarga no termino. Relanza este script: es reanudable y solo bajara lo que falte."
    "$PYV" -c "import sys; sys.path.insert(0, sys.argv[1]); import config; print('   ', config.fijar_ruta(sys.argv[2]))" \
        "$REPO" "$DATOS" || falla "no se pudo guardar la ruta del dataset"
    bien "ruta guardada en .bdcedl_ruta"
fi

# --------------------------------------------------------------------------- #
paso "7/7  Verificacion"
# --------------------------------------------------------------------------- #
if ! (cd "$REPO" && "$PYV" verificar_entorno.py); then
    printf '\n\033[31mLa verificacion ha encontrado problemas. Mira la lista de arriba.\033[0m\n'
    exit 1
fi

echo
echo "Listo. Activa el entorno con:"
echo "    source .venv/bin/activate"
echo