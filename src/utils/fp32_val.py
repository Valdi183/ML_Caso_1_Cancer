#!/usr/bin/env python3
"""Predicciones fuera de fold recalculadas en FP32 (sin TF32), solo validacion.

    python -m src.utils.fp32_val

Las predicciones oficiales (results/<ejecucion>/predicciones_val.csv) se calcularon en
GPU con lotes de 128, donde cuDNN usa TF32 por defecto. Aqui se vuelven a calcular con
`src.inferencia.probabilidades` (FP32, como la app) con los mismos 5 pesos (SHA-256 del
plan) y se comparan: diferencia maxima de probabilidad, AUC por paciente, percentil 90 de
las pCR=0 y matriz con el umbral oficial. Es descriptivo: el umbral sigue siendo el
oficial y no se sobrescribe nada. Test NO se usa.

Escribe results/fp32_val/<ejecucion>.csv (por corte: prob FP32 y oficial) y resumen.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src import config, inferencia
from src.data.carga import cargar_en_memoria
from src.utils.analisis_cv import auc
from src.utils.umbral import CONFIGURACIONES, metricas, umbral_especificidad

DESTINO = config.RESULTADOS / "fp32_val"


def por_paciente(cortes: pd.DataFrame, columna: str) -> pd.DataFrame:
    tabla = cortes.groupby("patient_id").agg(prob=(columna, "mean"), pcr=("pCR", "first"),
                                             fold=("fold", "first"))
    tabla["percentil"] = tabla.groupby("fold").prob.rank(pct=True)
    return tabla


def resumen_tabla(tabla: pd.DataFrame, umbral: float) -> dict:
    m = metricas(tabla.prob, tabla.pcr, umbral)
    aucs = [auc(d.prob, d.pcr) for _, d in tabla.groupby("fold")]
    return {"auc_por_fold": [float(a) for a in aucs], "auc_media_folds": float(np.mean(aucs)),
            "auc_global_percentiles_por_fold": auc(tabla.percentil, tabla.pcr),
            "auc_global_crudo": auc(tabla.prob, tabla.pcr),
            "percentil_90_pcr0": umbral_especificidad(tabla.prob[tabla.pcr == 0]),
            "con_umbral_oficial": {k: m[k] for k in ("VP", "FP", "FN", "VN", "sensibilidad", "especificidad",
                                                     "VPP", "f1_macro")}}


def main() -> None:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conjunto = inferencia.cargar_conjunto(dispositivo=str(dispositivo))      # pesos comprobados
    uc = config.utils()
    samples = uc.cargar_samples()
    DESTINO.mkdir(parents=True, exist_ok=True)

    cortes, por_ejecucion = [], {}
    for nombre in CONFIGURACIONES["normal"]:
        red = conjunto.redes[conjunto.nombres.index(nombre)]
        fold = json.loads((config.RESULTADOS / nombre / "config.json").read_text(encoding="utf-8"))["fold"]
        oficial = pd.read_csv(config.RESULTADOS / nombre / "predicciones_val.csv")
        _, valida = uc.particion(samples, fold_val=fold)
        if not np.array_equal(valida.sample_id.values, oficial.sample_id.values):
            raise SystemExit(f"{nombre}: los cortes de validacion no estan en el orden de "
                             "predicciones_val.csv")
        prob = inferencia.probabilidades(red, cargar_en_memoria(valida, f"fold {fold}"), dispositivo)
        c = oficial.rename(columns={"prob": "prob_oficial"}).assign(prob_fp32=prob, fold=fold)
        c.to_csv(DESTINO / f"{nombre}.csv", index=False)
        diferencia = (c.prob_fp32 - c.prob_oficial).abs()
        por_ejecucion[nombre] = {"fold": fold, "cortes": int(len(c)),
                                 "diferencia_maxima": float(diferencia.max()),
                                 "diferencia_media": float(diferencia.mean())}
        cortes.append(c)
    cortes = pd.concat(cortes)

    umbral = inferencia.leer_umbral()
    oficial, fp32 = por_paciente(cortes, "prob_oficial"), por_paciente(cortes, "prob_fp32")
    cambian = int(((oficial.prob > umbral) != (fp32.prob > umbral)).sum())
    resumen = {
        "aviso": "Descriptivo, solo validacion fuera de fold. El umbral oficial no cambia (0,3635) y las "
                 "predicciones oficiales no se sobrescriben. Test no se usa.",
        "dispositivo": str(dispositivo), "umbral_oficial": umbral,
        "por_ejecucion": por_ejecucion,
        "diferencia_maxima_corte": float((cortes.prob_fp32 - cortes.prob_oficial).abs().max()),
        "diferencia_maxima_paciente": float((fp32.prob - oficial.prob).abs().max()),
        "pacientes": int(len(oficial)), "cortes": int(len(cortes)),
        "oficial_tf32": resumen_tabla(oficial, umbral),
        "fp32": resumen_tabla(fp32, umbral),
        "pacientes_que_cambian_de_clase": cambian,
        "pacientes_a_menos_de_5e-4_del_umbral_oficial": int(((oficial.prob - umbral).abs() < 5e-4).sum()),
    }
    (DESTINO / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float), encoding="utf-8")

    o, f = resumen["oficial_tf32"], resumen["fp32"]
    print(f"\nFP32 frente a oficial (TF32), {resumen['pacientes']} pacientes, {resumen['cortes']} cortes, "
          f"en {dispositivo}")
    print(f"  diferencia maxima de probabilidad: corte {resumen['diferencia_maxima_corte']:.2e}, "
          f"paciente {resumen['diferencia_maxima_paciente']:.2e}")
    for clave, titulo in (("auc_media_folds", "AUC media de folds"),
                          ("auc_global_percentiles_por_fold", "AUC global (percentiles por fold)"),
                          ("percentil_90_pcr0", "percentil 90 de las pCR=0")):
        print(f"  {titulo:<36} oficial {o[clave]:.6f}   FP32 {f[clave]:.6f}")
    for nombre, x in (("oficial", o), ("FP32", f)):
        m = x["con_umbral_oficial"]
        print(f"  matriz {nombre:<8} con {umbral:.4f}: VP {m['VP']} FP {m['FP']} FN {m['FN']} VN {m['VN']}  "
              f"sens {m['sensibilidad']:.3f} esp {m['especificidad']:.3f}")
    print(f"  pacientes que cambian de clase: {cambian}")
    print(f"Guardado en {DESTINO.relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
