/* El conjunto de los 5 modelos en el navegador: ONNX Runtime Web (WebAssembly, 1 hilo).
 *
 * Carga web/modelo.json (umbral, SHA-256, referencias), descarga el ONNX y comprueba que su SHA-256
 * es el de modelo.json antes de crear la sesion. Entrada: (1, 3, 256, 256) float32 en [0, 1] (el PNG
 * leido por png.js dividido entre 255); salidas: `prob` (media de las 5 sigmoides) y `por_modelo`.
 * Un hilo: GitHub Pages no puede enviar las cabeceras que piden los hilos de WebAssembly.
 */
(function (raiz) {
  "use strict";

  let sesion = null, meta = null;

  async function cargar(base) {
    base = base || "";
    meta = await (await fetch(base + "modelo.json")).json();
    const respuesta = await fetch(base + meta.modelo.onnx);
    if (!respuesta.ok) throw new Error("No se pudo descargar " + meta.modelo.onnx);
    const bytes = new Uint8Array(await respuesta.arrayBuffer());
    const huella = await raiz.Png.sha256Hex(bytes);
    if (huella !== meta.modelo.sha256) {
      throw new Error("El modelo descargado no coincide con el SHA-256 de modelo.json: no se usa.");
    }
    const ort = raiz.ort;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    // URL absoluta: ONNX Runtime importa el .mjs como modulo y una ruta relativa no es un especificador valido
    ort.env.wasm.wasmPaths = new URL(base + "js/vendor/ort/", document.baseURI).href;
    sesion = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return meta;
  }

  /* fases: [Uint8Array(65536) x 3] en el orden PRE, EARLY, LATE. */
  async function predecir(fases) {
    if (!sesion) throw new Error("El modelo no esta cargado.");
    const entrada = raiz.Png.aEntrada(fases);
    const inicio = performance.now();
    const salida = await sesion.run({ imagen: new raiz.ort.Tensor("float32", entrada, [1, 3, 256, 256]) });
    const ms = performance.now() - inicio;
    const prob = salida.prob.data[0];
    const porModelo = Array.from(salida.por_modelo.data);
    if (!Number.isFinite(prob) || porModelo.some((p) => !Number.isFinite(p))) {
      throw new Error("La red ha devuelto un valor no finito.");
    }
    const pcr = prob > meta.umbral;
    return { prob, porModelo, pcr, clase: pcr ? "pCR" : "no pCR", umbral: meta.umbral, ms,
             texto: texto(prob) };
  }

  /* Como la app y `python -m src.inferencia`: 4 decimales con coma decimal. */
  function texto(p) {
    return p.toFixed(4).replace(".", ",");
  }

  raiz.Modelo = { cargar, predecir, texto, get meta() { return meta; } };
})(window);
