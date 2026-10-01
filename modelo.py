"""CNN propia para predecir pCR a partir de un corte DCE-MRI.

Este fichero es la UNICA definicion de la red y de su preprocesado. Lo importan
`entrenar.py` y la app web, asi que la entrada que ve la red al entrenar y al
predecir en la defensa es exactamente la misma: el tensor (3, 256, 256) en [0, 1]
que devuelve `utils_caso.cargar_imagen`, sin ninguna normalizacion adicional
fuera de aqui.

    from modelo import CNNpCR, cargar
    red = CNNpCR(realce=True)                 # para entrenar
    red, info = cargar("modelos/base.pt")     # para predecir (ya en eval)

Variantes (se eligen al crear la red y se guardan con los pesos):

    realce    anade EARLY - PRE y LATE - EARLY como canales extra
    sin_late  descarta la fase LATE (su momento cambia segun la cohorte)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Preprocesado(nn.Module):
    """Construye los canales de entrada a partir de PRE, EARLY y LATE.

    No tiene pesos: son restas y selecciones fijas. Va dentro de la red para que
    la app no pueda hacerlo distinto.
    """

    def __init__(self, realce: bool = False, sin_late: bool = False):
        super().__init__()
        self.realce = realce
        self.sin_late = sin_late

    @property
    def canales(self) -> int:
        fases = 2 if self.sin_late else 3
        restas = (1 if self.sin_late else 2) if self.realce else 0
        return fases + restas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre, early, late = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        partes = [pre, early] if self.sin_late else [pre, early, late]
        if self.realce:
            partes.append(early - pre)                 # captacion del contraste
            if not self.sin_late:
                partes.append(late - early)            # lavado (washout) si es negativo
        return torch.cat(partes, dim=1)


def bloque(entrada: int, salida: int) -> nn.Sequential:
    """2 x (conv 3x3 + BatchNorm + ReLU) y max pool: mitad de tamano, mas filtros.

    Las convoluciones no llevan bias porque BatchNorm ya anade su propio
    desplazamiento justo despues.
    """
    return nn.Sequential(
        nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.Conv2d(salida, salida, 3, padding=1, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNpCR(nn.Module):
    """Corte (B, 3, 256, 256) en [0, 1]  ->  logit de pCR (B,).

    Cuatro bloques convolucionales (256 -> 16 px de lado), global average pooling
    para no depender de donde este el tumor, dropout y una capa lineal.
    """

    def __init__(self, realce: bool = False, sin_late: bool = False,
                 filtros: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.5):
        super().__init__()
        # Todo lo necesario para reconstruir la red: se guarda junto a los pesos
        self.config = {"realce": realce, "sin_late": sin_late,
                       "filtros": list(filtros), "dropout": dropout}

        self.preprocesado = Preprocesado(realce, sin_late)
        entradas = [self.preprocesado.canales, *filtros[:-1]]
        self.bloques = nn.Sequential(*[bloque(a, b) for a, b in zip(entradas, filtros)])
        self.cabeza = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filtros[-1], 1),
        )
        self._inicializar()

    def _inicializar(self):
        # He (Kaiming): mantiene la escala de las activaciones con ReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.bloques(self.preprocesado(x))).squeeze(1)


def parametros(red: nn.Module) -> int:
    return sum(p.numel() for p in red.parameters() if p.requires_grad)


def guardar(red: CNNpCR, ruta: Path | str, **info) -> None:
    """Guarda pesos + configuracion de la arquitectura + lo que se pase en `info`
    (umbral, metricas, parametros de entrenamiento...), que la app muestra."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": red.config, "pesos": red.state_dict(), "info": info}, ruta)


def cargar(ruta: Path | str, dispositivo: str | torch.device = "cpu") -> tuple[CNNpCR, dict]:
    """Reconstruye la red desde un fichero de `guardar`, ya en modo evaluacion."""
    paquete = torch.load(ruta, map_location=dispositivo, weights_only=True)
    red = CNNpCR(**paquete["config"]).to(dispositivo)
    red.load_state_dict(paquete["pesos"])
    red.eval()
    return red, paquete["info"]


if __name__ == "__main__":
    # Resumen rapido de cada variante: canales de entrada y parametros
    for opciones in ({}, {"realce": True}, {"sin_late": True}, {"realce": True, "sin_late": True}):
        red = CNNpCR(**opciones)
        salida = red(torch.rand(2, 3, 256, 256))
        n = f"{parametros(red):,}".replace(",", ".")
        print(f"{str(opciones):40s} canales {red.preprocesado.canales}  "
              f"parametros {n}  salida {tuple(salida.shape)}")
