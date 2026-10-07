"""CNN propia para predecir pCR a partir de un corte DCE-MRI.

Este fichero es la UNICA definicion de la red y de su preprocesado. Lo importan
`entrenar.py` y la app web, asi que la entrada que ve la red al entrenar y al
predecir en la defensa es exactamente la misma: el tensor (3, 256, 256) en [0, 1]
que devuelve `utils_caso.cargar_imagen`, sin ninguna normalizacion adicional
fuera de aqui.

    from src.models.cnn import CNNpCR, cargar
    red = CNNpCR(realce=True)                 # para entrenar
    red, info = cargar("models/base.pt")      # para predecir (ya en eval)

Variantes (se eligen al crear la red y se guardan con los pesos):

    realce      anade EARLY - PRE y LATE - EARLY como canales extra
    sin_late    descarta la fase LATE (su momento cambia segun la cohorte)
    normalizar  estandariza cada corte (media 0, desviacion 1) antes de todo
    pooling     como se resume el mapa final: media, max, mediamax o atencion
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

    def __init__(self, realce: bool = False, sin_late: bool = False, normalizar: bool = False):
        super().__init__()
        self.realce = realce
        self.sin_late = sin_late
        self.normalizar = normalizar

    @property
    def canales(self) -> int:
        fases = 2 if self.sin_late else 3
        restas = (1 if self.sin_late else 2) if self.realce else 0
        return fases + restas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalizar:
            # Cada cohorte procesa las imagenes distinto y sus intensidades no son
            # comparables. Se estandariza cada corte con UNA media y UNA desviacion
            # para las tres fases juntas: por separado se borraria cuanto mas brilla
            # EARLY que PRE, que es justo la captacion de contraste.
            media = x.mean(dim=(1, 2, 3), keepdim=True)
            desviacion = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - media) / (desviacion + 1e-6)
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


POOLINGS = ("media", "max", "mediamax", "atencion")


class Resumen(nn.Module):
    """Mapa final (B, C, H, W)  ->  vector (B, C) o (B, 2C) que ve la capa lineal.

    media     global average pooling: todas las posiciones pesan igual. Si el tumor
              ocupa poca superficie, su señal se diluye entre el tejido normal.
    max       la respuesta mas fuerte de cada detector, este donde este.
    mediamax  las dos concatenadas: cuanto hay de cada patron y su pico.
    atencion  media ponderada con un peso por posicion que aprende la red (una
              conv 1x1 y softmax sobre H x W): aprende donde mirar. Empieza con
              todos los pesos iguales, es decir, igual que `media`.
    """

    def __init__(self, canales: int, modo: str = "media"):
        super().__init__()
        if modo not in POOLINGS:
            raise ValueError(f"pooling '{modo}' no existe; opciones: {POOLINGS}")
        self.modo = modo
        self.salidas = 2 * canales if modo == "mediamax" else canales
        if modo == "atencion":
            self.atencion = nn.Conv2d(canales, 1, 1)

    def pesos_atencion(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W): cuanto pesa cada posicion. Suma 1 en cada imagen."""
        b, _, h, w = x.shape
        return torch.softmax(self.atencion(x).flatten(1), dim=1).view(b, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modo == "media":
            return x.mean(dim=(2, 3))
        if self.modo == "max":
            return x.amax(dim=(2, 3))
        if self.modo == "mediamax":
            return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        pesos = self.pesos_atencion(x)
        return (x * pesos.unsqueeze(1)).sum(dim=(2, 3))


class CNNpCR(nn.Module):
    """Corte (B, 3, 256, 256) en [0, 1]  ->  logit de pCR (B,).

    Cuatro bloques convolucionales (256 -> 16 px de lado), un resumen del mapa
    final (`pooling`, por defecto la media), dropout y una capa lineal.
    """

    def __init__(self, realce: bool = False, sin_late: bool = False,
                 filtros: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.5,
                 normalizar: bool = False, pooling: str = "media"):
        super().__init__()
        # Todo lo necesario para reconstruir la red: se guarda junto a los pesos.
        # Los pesos anteriores a normalizar/pooling no los llevan y cargan con los
        # valores por defecto, que reproducen la red de entonces.
        self.config = {"realce": realce, "sin_late": sin_late,
                       "filtros": list(filtros), "dropout": dropout,
                       "normalizar": normalizar, "pooling": pooling}

        self.preprocesado = Preprocesado(realce, sin_late, normalizar)
        entradas = [self.preprocesado.canales, *filtros[:-1]]
        self.bloques = nn.Sequential(*[bloque(a, b) for a, b in zip(entradas, filtros)])
        resumen = Resumen(filtros[-1], pooling)
        # Mismas posiciones que antes (la lineal sigue siendo cabeza.3) para que los
        # pesos ya entrenados sigan cargando.
        self.cabeza = nn.Sequential(
            resumen,
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(resumen.salidas, 1),
        )
        self._inicializar()

    def _inicializar(self):
        # He (Kaiming): mantiene la escala de las activaciones con ReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        # La atencion empieza uniforme (todas las posiciones igual = media)
        resumen = self.cabeza[0]
        if resumen.modo == "atencion":
            nn.init.zeros_(resumen.atencion.weight)
            nn.init.zeros_(resumen.atencion.bias)

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
    for opciones in ({}, {"realce": True}, {"sin_late": True}, {"realce": True, "sin_late": True},
                     {"normalizar": True}, *({"pooling": p} for p in POOLINGS[1:])):
        red = CNNpCR(**opciones)
        salida = red(torch.rand(2, 3, 256, 256))
        n = f"{parametros(red):,}".replace(",", ".")
        print(f"{str(opciones):40s} canales {red.preprocesado.canales}  "
              f"parametros {n}  salida {tuple(salida.shape)}")
