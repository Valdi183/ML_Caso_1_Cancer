#!/usr/bin/env bash
# Cola de entrenamientos: se ejecutan uno detras de otro en la misma GPU.
# Si uno falla, sigue con el siguiente. Se edita en el portatil, se sube con git
# y en el laboratorio se lanza (con el entorno activado) asi:
#
#     nohup bash cola.sh > cola.log 2>&1 &
#     tail -f cola.log            # Ctrl+C sale del tail, la cola sigue
#
# Cada experimento guarda en resultados/<nombre>/ y modelos/<nombre>.pt

cd "$(dirname "${BASH_SOURCE[0]}")"

# El Python del .venv aunque no se haya activado el entorno
PY=python
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

experimentos=(
    "--ponderada --aumentado"
    "--ponderada --aumentado --lr 3e-4 --nombre ponderada_aum_lr3e-4_f0"
    "--ponderada --realce --aumentado"
)

for opciones in "${experimentos[@]}"; do
    echo
    echo "=== $(date '+%H:%M:%S')  python entrenar.py $opciones"
    "$PY" entrenar.py $opciones || echo "=== FALLO: $opciones"
done

echo
echo "=== $(date '+%H:%M:%S')  Cola terminada"
