#!/usr/bin/env bash
# Lanza en segundo plano los entrenamientos de una iteracion, uno detras de otro,
# y muestra su progreso.
#
#     bash experimentos/lanzar.sh it2      entrena cada linea de experimentos/it2.txt
#     bash experimentos/ver.sh it2         vuelve a mostrar el progreso
#
# Sigue entrenando aunque se cierre la terminal o VS Code (nohup); Ctrl+C solo deja
# de mostrar el progreso. El log queda en logs/<iteracion>.log (git lo ignora).
#
# Funciona en Linux (laboratorio) y en Windows desde Git Bash (PC con NVIDIA).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

it="${1:?Uso: bash experimentos/lanzar.sh <iteracion>   (por ejemplo: it2)}"
lista="experimentos/$it.txt"
[[ -f "$lista" ]] || { echo "No existe $lista"; exit 1; }

# Un entrenamiento ya en marcha: en Linux lo ve pgrep; Git Bash no trae pgrep,
# asi que ahi se mira el PID que deja este script en logs/en_marcha.pid.
if command -v pgrep >/dev/null; then
    if pgrep -f "src.training.entrenar" >/dev/null; then
        echo "Ya hay un entrenamiento en marcha; no lanzo otro encima:"
        pgrep -af "src.training.entrenar"
        exit 1
    fi
elif [[ -f logs/en_marcha.pid ]] && kill -0 "$(cat logs/en_marcha.pid)" 2>/dev/null; then
    echo "Ya hay un entrenamiento en marcha (PID $(cat logs/en_marcha.pid)); no lanzo otro encima."
    exit 1
fi

# El Python del .venv aunque no se haya activado el entorno (Linux / Windows)
PY=python
if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
    PY=.venv/Scripts/python.exe
fi

# En Windows, Python escribe a fichero en cp1252 y con bufer: el log se corromperia
# con caracteres fuera de esa tabla y ver.sh tardaria en mostrar nada. Inocuo en Linux.
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

# Sin GPU entrenaria en CPU durante horas sin avisar. PERMITIR_CPU=1 lo salta (pruebas).
if [[ "${PERMITIR_CPU:-0}" != 1 ]] && ! "$PY" -c "import sys, torch; sys.exit(not torch.cuda.is_available())"; then
    echo "PyTorch no ve la GPU: no lanzo nada (¿esta bien conectada? ¿es el .venv correcto?)"
    exit 1
fi

mkdir -p logs
log="logs/$it.log"
nohup bash -c '
    while IFS= read -r opciones || [[ -n "$opciones" ]]; do
        [[ -z "${opciones// /}" || "$opciones" == \#* ]] && continue
        echo
        echo "=== $(date +%H:%M:%S)  $opciones"
        "$0" -m src.training.entrenar $opciones || echo "=== FALLO: $opciones"
    done < "$1"
    echo
    echo "=== $(date +%H:%M:%S)  Iteracion terminada"
' "$PY" "$lista" > "$log" 2>&1 &
echo $! > logs/en_marcha.pid

echo "Lanzado en segundo plano. Log completo: $log"
exec bash experimentos/ver.sh "$it"
