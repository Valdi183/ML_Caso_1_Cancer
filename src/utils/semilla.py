"""Semilla unica para que un experimento se pueda repetir."""

from __future__ import annotations

import random

import numpy as np
import torch


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
