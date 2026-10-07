"""Carga de cortes en memoria, submuestreo para pruebas, lotes y aumentado de datos.

La particion por paciente, la lectura de cada imagen y el `Dataset` de PyTorch los
da `utils_caso.py` (en la carpeta del dataset); aqui solo lo que el entrenamiento
necesita encima de eso.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src import config


def submuestra(filas: pd.DataFrame, pacientes: int, semilla: int) -> pd.DataFrame:
    """Para --rapido: unas pocas pacientes, manteniendo la proporcion de pCR."""
    por_paciente = filas.groupby("patient_id").pCR.first()
    elegidas = (por_paciente.groupby(por_paciente, group_keys=False)
                            .apply(lambda g: g.sample(max(1, round(pacientes * len(g) / len(por_paciente))),
                                                      random_state=semilla)))
    return filas[filas.patient_id.isin(elegidas.index)]


def cargar_en_memoria(filas: pd.DataFrame, etiqueta: str) -> torch.Tensor:
    """(N, 3, 256, 256) uint8. Leer los PNG en cada epoca haria esperar a la GPU.

    Se usa `uc.cargar_imagen` (la misma funcion que la app) y se vuelve a 0-255:
    es exacto, porque las imagenes son PNG de 8 bits divididos entre 255.
    """
    uc = config.utils()
    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as hilos:
        imagenes = list(hilos.map(lambda f: np.rint(uc.cargar_imagen(f) * 255).astype(np.uint8),
                                  filas.itertuples()))
    tensor = torch.from_numpy(np.stack(imagenes))
    print(f"  {etiqueta}: {len(filas)} cortes, {filas.patient_id.nunique()} pacientes, "
          f"{tensor.numel() / 1024**3:.2f} GB, {time.perf_counter() - inicio:.0f} s")
    return tensor


def lotes(n: int, tamano: int, desordenar: bool):
    """Indices de cada lote. Sin desordenar para evaluar: las probabilidades tienen
    que salir en el mismo orden que las filas."""
    orden = torch.randperm(n) if desordenar else torch.arange(n)
    for i in range(0, n, tamano):
        yield orden[i:i + tamano]


def aumentar(x: torch.Tensor) -> torch.Tensor:
    """Volteos y rotacion de hasta 15 grados, iguales para los canales de un corte.

    Se transforma el tensor completo de cada corte, nunca canal a canal: desalinear
    PRE, EARLY y LATE destruiria el realce.
    """
    n = x.shape[0]
    volteo_h = torch.rand(n, device=x.device) < 0.5
    volteo_v = torch.rand(n, device=x.device) < 0.5
    x = torch.where(volteo_h[:, None, None, None], x.flip(3), x)
    x = torch.where(volteo_v[:, None, None, None], x.flip(2), x)

    angulo = (torch.rand(n, device=x.device) * 2 - 1) * np.deg2rad(15)
    cos, sen = torch.cos(angulo), torch.sin(angulo)
    ceros = torch.zeros_like(angulo)
    theta = torch.stack([torch.stack([cos, -sen, ceros], 1), torch.stack([sen, cos, ceros], 1)], 1)
    rejilla = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, rejilla, align_corners=False, padding_mode="zeros")
