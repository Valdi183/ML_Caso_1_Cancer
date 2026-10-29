"""Exporta el conjunto de los 5 modelos a ONNX para la web estatica (GitHub Pages).

    python -m src.web.exportar            # el ONNX, web/modelo.json, web/ejemplos/ y web/datos.json
    python -m src.web.exportar --solo-datos   # solo web/datos.json (las cifras de las dos paginas)

Un unico ONNX con los 5 pesos del plan (results/test/plan.json, comprobados por SHA-256 al
cargarlos) en modo evaluacion. El preprocesado va dentro de la red (src/models/cnn.py), asi que
la entrada es el corte tal cual: (1, 3, 256, 256) float32 en [0, 1] (el PNG leido / 255).
Salidas: `prob` (1,) = media de las 5 sigmoides y `por_modelo` (1, 5).

Opset 17 con el exportador clasico de TorchScript (`dynamo=False`): solo Conv, AveragePool,
MaxPool, Relu, ReduceMean, Gemm, Sigmoid y operaciones de forma, todas en ONNX Runtime Web
(WASM). No se toca el modelo ni el umbral: se lee `umbral_final` de results/umbral/resumen.json.

`web/modelo.json` lleva lo que la pagina muestra y comprueba: umbral, SHA-256 del ONNX y de los
5 .pt, parametros de entrenamiento, metricas internas (leidas de results/) y la probabilidad de
referencia de PyTorch (CPU, FP32) de los ejemplos de app/ejemplos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from src import inferencia
from src.data.imagen import FASES
from src.web import datos as datos_web

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
EJEMPLOS = REPO / "app" / "ejemplos"
OPSET = 17
NOMBRE_ONNX = "conjunto.onnx"


class Conjunto(torch.nn.Module):
    """Los 5 modelos en paralelo: (B, 3, 256, 256) -> (media de las 5 sigmoides (B,), cada una (B, 5))."""

    def __init__(self, redes):
        super().__init__()
        self.redes = torch.nn.ModuleList(redes)

    def forward(self, x: torch.Tensor):
        p = torch.stack([torch.sigmoid(red(x)) for red in self.redes], dim=1)
        return p.mean(dim=1), p


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def exportar_onnx(conjunto: inferencia.Conjunto, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    modelo = Conjunto(conjunto.redes).eval()
    ejemplo = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        torch.onnx.export(modelo, (ejemplo,), str(ruta), input_names=["imagen"],
                          output_names=["prob", "por_modelo"], opset_version=OPSET, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(ruta)))


def copiar_ejemplos(destino: Path) -> list[dict]:
    """Copia los 4 ejemplos (y su licencia) y devuelve su fila de ejemplos.csv."""
    destino.mkdir(parents=True, exist_ok=True)
    with open(EJEMPLOS / "ejemplos.csv", encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        for fase in FASES:
            shutil.copyfile(EJEMPLOS / f"{fila['corte']}_{fase}.png", destino / f"{fila['corte']}_{fase}.png")
    shutil.copyfile(EJEMPLOS / "ejemplos.csv", destino / "ejemplos.csv")
    shutil.copyfile(EJEMPLOS / "LEEME.md", destino / "LEEME.md")
    return filas


def referencias(conjunto: inferencia.Conjunto, filas: list[dict]) -> list[dict]:
    """Probabilidad de PyTorch (CPU, FP32) de cada ejemplo: lo que tiene que dar el navegador."""
    salida = []
    for fila in filas:
        rutas = [EJEMPLOS / f"{fila['corte']}_{fase}.png" for fase in FASES]
        r = inferencia.pipeline(conjunto, rutas, "cpu")
        salida.append({
            "corte": fila["corte"], "paciente": fila["paciente"], "cohorte": fila["cohorte"],
            "pcr": int(fila["pCR"]), "fold": int(fila["fold"]),
            "ficheros": {fase: f"ejemplos/{fila['corte']}_{fase}.png" for fase in FASES},
            "referencia": {"prob": r["prob"], "clase": r["clase"],
                           "por_modelo": [r["por_modelo"][n] for n in conjunto.nombres]},
        })
    return salida


def descripcion(conjunto: inferencia.Conjunto, onnx: Path) -> dict:
    import onnx as onnx_lib
    import onnxruntime
    plan = json.loads(inferencia.PLAN.read_text(encoding="utf-8"))["pesos"]
    datos = onnx.read_bytes()
    return {
        "formato": 1,
        "aviso": "Generado por src/web/exportar.py; no se edita a mano.",
        "modelo": {
            "onnx": f"modelo/{onnx.name}", "bytes": len(datos), "sha256": sha256_bytes(datos),
            "opset": OPSET,
            "entrada": {"nombre": "imagen", "forma": [1, 3, 256, 256], "tipo": "float32",
                        "rango": [0, 1], "nota": "PNG en gris de 8 bits dividido entre 255; el resto "
                                                 "del preprocesado va dentro de la red"},
            "salidas": {"prob": "media de las 5 sigmoides, forma (1,)", "por_modelo": "forma (1, 5)"},
            "pesos": [{"nombre": n, "fichero": p["fichero"], "sha256": conjunto.sha256[n]}
                      for n, p in zip(conjunto.nombres, plan)],
            "plan": "results/test/plan.json",
            "arquitectura": conjunto.config_red, "parametros_por_modelo": conjunto.parametros,
        },
        "umbral": conjunto.umbral,
        "regla": "pCR si probabilidad > umbral",
        "entrenamiento": {n: {"epoca": i["epoca"], "auc_val": i["auc_val"], **i["entrenamiento"]}
                          for n, i in conjunto.info.items()},
        "metricas": inferencia.metricas_internas(),
        "generado_con": {"torch": torch.__version__, "onnx": onnx_lib.__version__,
                         "onnxruntime": onnxruntime.__version__, "numpy": np.__version__},
    }


def main(argv=None) -> None:
    a = argparse.ArgumentParser(description="Exporta el modelo y las cifras de la web (web/).")
    a.add_argument("--solo-datos", action="store_true",
                   help="solo web/datos.json (las cifras de las dos paginas); no vuelve a exportar el ONNX")
    args = a.parse_args(argv)
    conjunto = inferencia.cargar_conjunto()          # comprueba el SHA-256 de los 5 .pt con el plan
    filas = copiar_ejemplos(WEB / "ejemplos")
    if not args.solo_datos:
        onnx = WEB / "modelo" / NOMBRE_ONNX
        exportar_onnx(conjunto, onnx)
        datos = descripcion(conjunto, onnx)
        datos["ejemplos"] = referencias(conjunto, filas)
        texto = json.dumps(datos, ensure_ascii=False, indent=1) + "\n"
        (WEB / "modelo.json").write_text(texto, encoding="utf-8")
        modelo = datos["modelo"]
        print(f"{onnx.relative_to(REPO)}: {modelo['bytes'] / 1024:.0f} KB, sha256 {modelo['sha256']}")
        print(f"umbral {conjunto.umbral:.4f}; {len(datos['ejemplos'])} ejemplos con referencia de PyTorch")
    cifras = datos_web.escribir(conjunto, [{"corte": f["corte"], "cohorte": f["cohorte"]} for f in filas])
    print(f"web/datos.json: {len(cifras['seleccion']['configuraciones'])} configuraciones; "
          f"portada {cifras['portada']['corte']}")


if __name__ == "__main__":
    main()
