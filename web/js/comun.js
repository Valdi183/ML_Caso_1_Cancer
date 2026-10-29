/* Lo que comparten las dos paginas: formato de las cifras (coma decimal, miles con punto, signo
 * menos de verdad) y la carga de datos.json. Ninguna cifra se escribe a mano en el HTML: cada
 * `data-v="ruta.al.dato"` se rellena aqui con `data-f="formato"` desde datos.json, que genera
 * `python -m src.web.exportar` a partir de results/ y docs/. */
(function (raiz) {
  "use strict";

  const MENOS = "−";

  function conComa(texto) {
    return texto.replace(".", ",").replace("-", MENOS);
  }

  /* 1273 -> "1.273" (el separador de miles tambien en los de cuatro cifras, como en el informe) */
  function entero(v) {
    const n = Math.round(Math.abs(v)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
    return (v < 0 ? MENOS : "") + n;
  }

  const decimales = (n) => (v) => conComa(Number(v).toFixed(n));
  const porCiento = (n) => (v) => conComa((100 * Number(v)).toFixed(n)) + " %";

  const signo3 = (v) => (v >= 0 ? "+" : "") + decimales(3)(v);

  const FORMATOS = {
    entero,
    d1: decimales(1), d2: decimales(2), d3: decimales(3), d4: decimales(4),
    pct0: porCiento(0), pct1: porCiento(1),
    /* [a, b] -> "0,457–0,629" */
    rango3: (v) => `${decimales(3)(v[0])}–${decimales(3)(v[1])}`,
    /* [a, b] -> "0,430 a 0,866" (con negativos no se lee bien un guion) */
    rango3a: (v) => `${decimales(3)(v[0])} a ${decimales(3)(v[1])}`,
    /* [a, b] -> "−0,001 a +0,028" (diferencias, con signo) */
    rangoSigno3: (v) => `${signo3(v[0])} a ${signo3(v[1])}`,
    signo3: (v) => signo3(v),
    /* proporcion -> "4" ("4 de cada 10") */
    de10: (v) => String(Math.floor(v * 10 + 0.5)),
    /* [8, 16, 32] -> "8, 16 y 32" */
    listaY: (v) => (v.length > 1 ? `${v.slice(0, -1).join(", ")} y ${v[v.length - 1]}` : String(v[0])),
    texto: (v) => String(v),
  };

  /* Prueba de n de cada 10 como en la app: la proporcion redondeada al entero mas cercano. */
  const deCada10 = (p) => Math.floor(p * 10 + 0.5);

  function camino(obj, ruta) {
    return ruta.split(".").reduce((o, k) => (o === undefined || o === null ? undefined : o[k]), obj);
  }

  /* Rellena todos los [data-v] de `raiz` (o del documento). Una ruta que no existe es un error
   * visible, no un hueco en silencio. */
  function rellenar(datos, nodo) {
    const faltan = [];
    (nodo || document).querySelectorAll("[data-v]").forEach((el) => {
      const ruta = el.getAttribute("data-v");
      const v = camino(datos, ruta);
      const pedido = el.getAttribute("data-f");
      // sin formato explicito: los enteros con miles (1.097) y lo demas con 3 decimales
      const f = FORMATOS[pedido || (Number.isInteger(v) ? "entero" : "d3")];
      if (v === undefined || v === null || !f) {
        faltan.push(ruta);
        el.textContent = "?";
        el.classList.add("dato-falta");
      } else {
        el.textContent = f(v);
      }
    });
    if (faltan.length) console.error("Datos que faltan en datos.json: " + faltan.join(", "));
    return faltan;
  }

  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

  /* Los parametros de entrenamiento (iguales en los 5 modelos), como filas de una lista de definicion. */
  function entrenamientoHtml(d) {
    const e = d.modelo.entrenamiento, cfg = d.modelo.config;
    const filas = [
      ["Filtros por bloque", cfg.filtros.join("-")],
      ["Resolución dentro de la red", `${256 / e.reducir} px`],
      ["Resumen del mapa final", e.pooling],
      ["Pérdida", e.ponderada ? "ponderada" : "normal"],
      ["Tasa de aprendizaje", conComa(String(e.lr))],
      ["Weight decay", conComa(String(e.wd))],
      ["Dropout", conComa(String(e.dropout))],
      ["Tamaño de lote", String(e.lote)],
      ["Épocas (máximo)", String(e.epocas)],
      ["Aumento de datos", e.aumentado ? "sí" : "no"],
      ["Semilla", String(e.semilla)],
    ];
    return filas.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  }

  async function cargarDatos() {
    const r = await fetch("datos.json");
    if (!r.ok) throw new Error("No se pudo cargar datos.json");
    return r.json();
  }

  raiz.Comun = { FORMATOS, entero, d1: FORMATOS.d1, d2: FORMATOS.d2, d3: FORMATOS.d3, d4: FORMATOS.d4,
                 pct0: FORMATOS.pct0, pct1: FORMATOS.pct1, conComa, deCada10, camino, rellenar,
                 cargarDatos, entrenamientoHtml, esc, MENOS };
})(window);
