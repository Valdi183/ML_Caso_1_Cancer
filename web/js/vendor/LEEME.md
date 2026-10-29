# Librerias de terceros (copiadas tal cual, sin CDN)

La web no carga nada de fuera: todo esto va en el repositorio. Version y origen exactos:

| Fichero | Paquete (npm) | Version | Licencia |
|---|---|---|---|
| `ort/ort.wasm.min.js`, `ort/ort-wasm-simd-threaded.mjs`, `ort/ort-wasm-simd-threaded.wasm` | `onnxruntime-web`, carpeta `dist/` (compilacion solo WASM, sin WebGL ni WebGPU) | 1.30.0 | MIT (`LICENSE-onnxruntime.txt`) |
| `UPNG.js` | `upng-js` | 2.1.0 | MIT (`LICENSE-UPNG.txt`) |
| `pako_inflate.min.js` | `pako`, carpeta `dist/` (solo descompresion zlib) | 1.0.11 | MIT y Zlib (`LICENSE-pako.txt`) |

UPNG.js solo se usa para quitar los filtros PNG y desentrelazar; la cabecera, los CRC, los
modos de color y las reglas de la app estan en `../png.js`. Nada de `canvas`: el navegador
puede aplicar gestion de color a un PNG pintado y cambiar los valores.
