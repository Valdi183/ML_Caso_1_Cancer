"""Metricas y graficas de un entrenamiento que no dependen de la red."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch  # antes que matplotlib: al reves, en Windows chocan sus DLL
from sklearn.metrics import roc_curve

from src import config


def perdida_media(probs: np.ndarray, etiquetas: torch.Tensor, criterio) -> float:
    logits = torch.logit(torch.from_numpy(probs).clamp(1e-6, 1 - 1e-6))
    return criterio(logits, etiquetas).item()


def umbral_youden(filas: pd.DataFrame, probs: np.ndarray) -> float:
    """Umbral que maximiza sensibilidad + especificidad, sobre la media por paciente."""
    uc = config.utils()
    por_paciente = uc.agregar_por_paciente(filas.patient_id.values, probs).join(
        filas.groupby("patient_id").pCR.first())
    fpr, tpr, umbrales = roc_curve(por_paciente.pCR, por_paciente.prob)
    return float(umbrales[np.argmax(tpr - fpr)])


def dibujar_curvas(historial: pd.DataFrame, ruta: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].plot(historial.epoca, historial.perdida_train, label="train")
    ejes[0].plot(historial.epoca, historial.perdida_val, label="validacion")
    ejes[0].set(title="Perdida", xlabel="epoca")
    ejes[0].legend()
    ejes[1].plot(historial.epoca, historial.auc_val, color="#55A868")
    ejes[1].axhline(0.69, color="gray", ls="--", lw=1, label="referencia clinica (0,69)")
    ejes[1].axhline(0.5, color="#C44E52", ls=":", lw=1, label="azar")
    ejes[1].set(title="AUC por paciente (validacion)", xlabel="epoca", ylim=(0.3, 1))
    ejes[1].legend(fontsize=8)
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(ruta, dpi=120, facecolor="white")
    plt.close(figura)
