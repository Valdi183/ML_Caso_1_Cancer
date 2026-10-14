#!/usr/bin/env bash
# Muestra el progreso de una iteracion:  bash experimentos/ver.sh it2
# Ctrl+C deja de mostrarlo; el entrenamiento sigue.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
log="logs/${1:?Uso: bash experimentos/ver.sh <iteracion>   (por ejemplo: it2)}.log"
[[ -f "$log" ]] || { echo "No existe $log (¿se ha lanzado esa iteracion?)"; exit 1; }
echo "(Ctrl+C deja de mostrar; el entrenamiento sigue)"
tail -n +1 -f "$log" | grep --line-buffered -E "===|Experimento|Red:|AUC train|Epoca elegida|Error|Traceback"
