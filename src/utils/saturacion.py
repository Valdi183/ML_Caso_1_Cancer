#!/usr/bin/env python3
"""¿Cuantos cortes tienen saturada la zona que mas realza? Solo train, nada de test.

    python -m src.utils.saturacion

Un PNG de 8 bits no guarda nada por encima de 255: si la zona que mas realza esta a 255 en
EARLY, ni la magnitud del realce ni el cambio de EARLY a LATE estan en la imagen, para
ningun modelo. Para cada corte de train se toma la zona de `src.visualizacion.curva_captacion`
(el 2 % de los pixeles de la mama con mayor EARLY - PRE) y se mira que fraccion de ella esta
a 255 en EARLY y en LATE. Un corte cuenta como saturado si lo esta MAS DE LA MITAD de la zona;
tambien con >= 250 (SATURADO, el criterio de la app). Por cohorte, por corte y por paciente.
Escribe results/saturacion/resumen.json.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src import config
from src import visualizacion as vis
from src.data.carga import rutas_corte
from src.data.imagen import leer_corte
from src.utils.analisis_cv import COHORTES

MAXIMO = 255


def zona_que_mas_realza(x: np.ndarray, fraccion: float = vis.FRACCION_ZONA) -> np.ndarray:
    """(3, k) uint8: PRE, EARLY y LATE de los pixeles de la zona de `vis.curva_captacion`
    (misma mascara, mismo suavizado y mismo orden; tests/test_saturacion.py lo comprueba)."""
    x = np.asarray(x)
    mama = vis.mascara_mama(x)
    realce = (vis.suavizar(x[1]) - vis.suavizar(x[0]))[mama]
    k = max(1, int(round(fraccion * realce.size)))
    zona = np.argsort(realce, kind="stable")[-k:]
    return np.stack([f[mama][zona] for f in x])


def saturacion_corte(x: np.ndarray) -> dict:
    """Fraccion de la zona a 255 y a >= SATURADO en EARLY y en LATE."""
    zona = zona_que_mas_realza(x)
    return {"early_255": float((zona[1] == MAXIMO).mean()), "late_255": float((zona[2] == MAXIMO).mean()),
            "early_250": float((zona[1] >= vis.SATURADO).mean()),
            "late_250": float((zona[2] >= vis.SATURADO).mean())}


def resumir(tabla: pd.DataFrame) -> dict:
    """Por corte: fraccion de cortes con mas de la mitad de la zona saturada (y la mediana de
    la fraccion saturada). Por paciente: fraccion de pacientes con mas de la mitad de sus
    cortes saturados."""
    r = {"cortes": int(len(tabla)), "pacientes": int(tabla.patient_id.nunique())}
    for col in ("early_255", "late_255", "early_250", "late_250"):
        saturado = tabla[col] > 0.5
        por_paciente = saturado.groupby(tabla.patient_id).mean()
        r[col] = {"fraccion_cortes": float(saturado.mean()),
                  "mediana_fraccion_de_la_zona": float(tabla[col].median()),
                  "fraccion_pacientes_mayoria_de_cortes": float((por_paciente > 0.5).mean())}
    ambas = (tabla.early_255 > 0.5) & (tabla.late_255 > 0.5)
    r["early_y_late_255_fraccion_cortes"] = float(ambas.mean())
    return r


def main() -> None:
    uc = config.utils()
    samples = uc.cargar_samples()
    pacientes = uc.cargar_patients().set_index("pid")
    filas = samples[samples.split == "train"].reset_index(drop=True)                 # sin test
    raiz = config.raiz_datos()
    with ThreadPoolExecutor(max_workers=8) as hilos:
        medidas = list(hilos.map(lambda f: saturacion_corte(leer_corte(rutas_corte(f, raiz))),
                                 filas.itertuples()))
    tabla = pd.concat([filas[["patient_id"]], pd.DataFrame(medidas)], axis=1)
    tabla["cohorte"] = tabla.patient_id.map(pacientes.dataset)

    resumen = {
        "aviso": "Solo train (sin test). Descriptivo: no cambia nada del modelo.",
        "criterio": f"corte saturado si mas de la mitad de la zona que mas realza (el "
                    f"{100 * vis.FRACCION_ZONA:.0f} % de la mama con mayor EARLY - PRE, la de "
                    "src.visualizacion.curva_captacion) esta a 255 (o >= 250, el criterio de la app)",
        "todas": resumir(tabla),
        "por_cohorte": {c: resumir(tabla[tabla.cohorte == c]) for c in COHORTES},
    }
    destino = config.RESULTADOS / "saturacion"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'':<6} {'cortes':>6}  EARLY=255  LATE=255  ambas   EARLY>=250  LATE>=250")
    for nombre, r in [("todas", resumen["todas"]), *resumen["por_cohorte"].items()]:
        print(f"{nombre:<6} {r['cortes']:>6}  {r['early_255']['fraccion_cortes']:>9.3f} "
              f"{r['late_255']['fraccion_cortes']:>9.3f} {r['early_y_late_255_fraccion_cortes']:>6.3f} "
              f"{r['early_250']['fraccion_cortes']:>11.3f} {r['late_250']['fraccion_cortes']:>10.3f}")
    print(f"\nGuardado en {(destino / 'resumen.json').relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
