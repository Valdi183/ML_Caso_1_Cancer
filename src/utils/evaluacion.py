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


def dibujar_curvas(historial: pd.DataFrame, ruta: Path, titulo: str,
                   epoca_elegida: int | None = None) -> None:
    """Perdida y AUC por paciente, train y validacion juntas, con la epoca elegida marcada.

    Es la figura que demuestra (o no) el sobreajuste: un modelo sano tiene las dos
    curvas de AUC subiendo juntas; uno que memoriza lleva train a 1 y deja validacion plana.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    azul, naranja = "#2B6CB0", "#DD6B20"
    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].plot(historial.epoca, historial.perdida_train, color=azul, label="entrenamiento")
    ejes[0].plot(historial.epoca, historial.perdida_val, color=naranja, label="validacion")
    ejes[0].set(title="Perdida", xlabel="epoca")
    if "auc_train" in historial:
        ejes[1].plot(historial.epoca, historial.auc_train, color=azul, label="entrenamiento")
    ejes[1].plot(historial.epoca, historial.auc_val, color=naranja, alpha=0.45, lw=1, label="validacion")
    if "auc_val_suave" in historial:
        ejes[1].plot(historial.epoca, historial.auc_val_suave, color=naranja, lw=2,
                     label="validacion (suavizada)")
    ejes[1].axhline(0.5, color="#C44E52", ls=":", lw=1, label="azar")
    ejes[1].set(title="AUC por paciente", xlabel="epoca", ylim=(0.3, 1))
    if epoca_elegida is not None:
        for ax in ejes:
            ax.axvline(epoca_elegida, color="gray", ls="--", lw=1)
        fila = historial[historial.epoca == epoca_elegida].iloc[0]
        if "auc_train" in historial:
            ejes[1].set_title(f"AUC por paciente  (epoca {epoca_elegida}: train {fila.auc_train:.2f}, "
                              f"val {fila.auc_val:.2f})")
    for ax in ejes:
        ax.legend(fontsize=8)
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(ruta, dpi=120, facecolor="white")
    plt.close(figura)
