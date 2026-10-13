#!/usr/bin/env bash
# Cola de entrenamientos: se ejecutan uno detras de otro en la misma GPU.
# Si uno falla, sigue con el siguiente. Se edita en el portatil, se sube con git
# y en el laboratorio se lanza (con el entorno activado) asi:
#
#     nohup bash cola.sh > cola.log 2>&1 &
#     tail -f cola.log            # Ctrl+C sale del tail, la cola sigue
#
# Cada experimento guarda en results/<nombre>/ y models/<nombre>.pt

cd "$(dirname "${BASH_SOURCE[0]}")"

# El Python del .venv aunque no se haya activado el entorno
PY=python
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

# Diagnostico: ¿ve la red el tumor? Aprende el SUBTIPO tumoral en lugar de pCR
# (la red solo recibe la imagen; el subtipo es la respuesta a adivinar).
# Referencia: la cohorte sola da AUC 0,52-0,55 para el subtipo; la cohorte Duke
# se aprendio con AUC 0,998 (results/objduke_normal_f0).
experimentos=(
    "--objetivo triple --epocas 20"
    "--objetivo her2 --epocas 20"
    "--objetivo hrpos --epocas 20"
)

for opciones in "${experimentos[@]}"; do
    echo
    echo "=== $(date '+%H:%M:%S')  python -m src.training.entrenar $opciones"
    "$PY" -m src.training.entrenar $opciones || echo "=== FALLO: $opciones"
done

echo
echo "=== $(date '+%H:%M:%S')  Cola terminada"
