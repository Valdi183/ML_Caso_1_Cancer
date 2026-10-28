#!/usr/bin/env python3
"""Esquema de la arquitectura del modelo entregado: docs/figuras/arquitectura.png.

    python -m src.utils.arquitectura

Nada escrito a mano: la red se construye con la configuracion guardada en los pesos del
congelado (models/it6_3bloques_f0.pt; los 5 comparten configuracion), los tamaños salen de
pasar un corte vacio con hooks y los parametros se cuentan modulo a modulo. El umbral se
lee con src.inferencia.leer_umbral (results/umbral/resumen.json). La app muestra la figura si existe.
"""

from __future__ import annotations

import torch  # antes que matplotlib (figuras)

from src import config
from src.models.cnn import CNNpCR, parametros
from src.utils import figuras
from src.utils.umbral import CONFIGURACIONES


def etapas(red: CNNpCR) -> list[dict]:
    """Una fila por etapa: nombre, detalle, forma de la salida (sin el lote) y parametros."""
    formas = {}

    def guardar(nombre):
        return lambda _m, _e, salida: formas.__setitem__(nombre, tuple(salida.shape[1:]))

    modulos = {"pre": red.preprocesado, "resumen": red.cabeza[0],
               **{f"b{i}": b for i, b in enumerate(red.bloques)}}
    ganchos = [m.register_forward_hook(guardar(n)) for n, m in modulos.items()]
    red.eval()
    with torch.no_grad():
        red(torch.zeros(1, 3, 256, 256))
    for g in ganchos:
        g.remove()

    def forma(t):
        return " × ".join(str(d) for d in t)

    c = red.config
    filas = [{"nombre": "Entrada", "detalle": "PRE, EARLY, LATE\n(3 fases, no RGB)\nuint8 / 255",
              "forma": "3 × 256 × 256", "parametros": 0, "tipo": "entrada"},
             {"nombre": "Reducir", "detalle": f"media de bloques\n{c['reducir']} × {c['reducir']}",
              "forma": forma(formas["pre"]), "parametros": 0, "tipo": "fijo"}]
    detalle_bloque = "2 × (conv 3×3\n+ BatchNorm + ReLU)\nmax pool 2"
    for i, b in enumerate(red.bloques):
        filas.append({"nombre": f"Bloque {i + 1}", "detalle": detalle_bloque,
                      "forma": forma(formas[f"b{i}"]), "parametros": parametros(b), "tipo": "conv"})
    filas.append({"nombre": "Resumen", "detalle": f"{c['pooling']} global\n(cada mapa → 1 valor)",
                  "forma": forma(formas["resumen"]), "parametros": parametros(red.cabeza[0]),
                  "tipo": "cabeza"})
    filas.append({"nombre": "Clasificador", "detalle": f"dropout {decimal(c['dropout'], 1)}\nlineal "
                                                       f"{formas['resumen'][0]} → 1\nsigmoide",
                  "forma": "p(pCR)", "parametros": parametros(red.cabeza), "tipo": "salida"})
    return filas


def miles(n: int) -> str:
    """18329 -> '18.329' (separador de miles español)."""
    return f"{n:,}".replace(",", ".")


def decimal(v: float, decimales: int) -> str:
    """0.3635 -> '0,3635' (coma decimal)."""
    return f"{v:.{decimales}f}".replace(".", ",")


def dibujar(filas: list[dict], total: int, umbral: float):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt = figuras.plt
    colores = {"entrada": "#E2E8F0", "fijo": "#EDF2F7", "conv": "#BEE3F8", "cabeza": "#FEEBC8",
               "salida": "#FED7D7"}
    bordes = {"entrada": figuras.GRIS, "fijo": figuras.GRIS, "conv": figuras.AZUL, "cabeza": figuras.NARANJA,
              "salida": figuras.ROJO}
    ancho, alto, hueco = 1.95, 1.9, 0.4
    figura, ax = plt.subplots(figsize=(1.05 * len(filas) * (ancho + hueco), 4.4))
    for i, f in enumerate(filas):
        x = i * (ancho + hueco)
        ax.add_patch(FancyBboxPatch((x, 0), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=colores[f["tipo"]], edgecolor=bordes[f["tipo"]], lw=1.5))
        ax.text(x + ancho / 2, alto - 0.22, f["nombre"], ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + ancho / 2, alto / 2 + 0.05, f["detalle"], ha="center", va="center", fontsize=8.5,
                linespacing=1.3)
        ax.text(x + ancho / 2, 0.22, f["forma"], ha="center", va="center", fontsize=9.5, family="monospace")
        if f["parametros"]:
            ax.text(x + ancho / 2, -0.25, f"{miles(f['parametros'])} parámetros", ha="center",
                    va="center", fontsize=8, color=figuras.GRIS)
        if i:
            ax.add_patch(FancyArrowPatch((x - hueco + 0.04, alto / 2), (x - 0.04, alto / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#4A5568", lw=1.3))
    fin = len(filas) * (ancho + hueco) - hueco
    ax.text(fin / 2, -0.85, f"Conjunto entregado: las 5 redes (una por fold, misma arquitectura, "
                            f"{miles(total)} parámetros cada una) → media de las 5 probabilidades → "
                            f"pCR si p > {decimal(umbral, 4)}",
            ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": figuras.ROJO})
    ax.text(fin / 2, alto + 0.35, "CNN propia desde cero (sin preentrenar): un corte de RM DCE → "
                                  "probabilidad de pCR", ha="center", va="center", fontsize=12)
    ax.set(xlim=(-0.15, fin + 0.15), ylim=(-1.25, alto + 0.6))
    ax.set_aspect("equal")
    ax.axis("off")
    return figura


def main() -> None:
    from src.inferencia import leer_umbral
    from src.models.cnn import cargar
    red, _ = cargar(config.MODELOS / f"{CONFIGURACIONES['normal'][0]}.pt")
    umbral = leer_umbral()
    filas = etapas(red)
    ruta = figuras.guardar(dibujar(filas, parametros(red), umbral), "arquitectura")
    figuras.plt.close("all")
    for f in filas:
        print(f"{f['nombre']:<13} {f['forma']:<15} {f['parametros']:>7}")
    print(f"Total {parametros(red)} parametros; guardado en {ruta.relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
