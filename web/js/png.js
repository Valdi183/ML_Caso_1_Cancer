/* Lectura de un PNG en el navegador, igual que la app (src/inferencia.py: validar_fase,
 * avisos_corte) y que el entrenamiento (src/data/imagen.py: leer_fase = PIL, convert("L")).
 *
 * Sin <canvas>: el navegador puede aplicar gestion de color al pintar un PNG y cambiar los
 * valores. Aqui se leen los bytes del fichero: cabecera y CRC (a mano), descompresion zlib
 * (pako) y quitar los filtros PNG (UPNG.js). Despues se pasa a escala de grises como PIL.
 *
 * Los mensajes son los de la app. Entre parentesis, en los errores de "corrupto", va el tipo de
 * excepcion de Python (OSError, SyntaxError...), que aqui se reproduce segun donde falla el fichero.
 * La prueba web/pruebas/lectura.html compara este fichero con Python en ~60 casos.
 */
(function (raiz) {
  "use strict";

  const LADO = 256;
  const TAMANO_MAXIMO = 2 * 1024 * 1024;               // bytes por fase
  const FIRMA = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  const PIXELES_BOMBA = 2 * 89478485;                  // PIL: DecompressionBombError por encima
  const FASES = ["PRE", "EARLY", "LATE"];

  class ErrorEntrada extends Error {}                  // el mensaje se le muestra tal cual al usuario
  class Fallo extends Error {                          // fallo interno del PNG: se traduce a "corrupto"
    constructor(tipo) { super(tipo); this.tipo = tipo; }
  }

  // --- utilidades -------------------------------------------------------- //
  const TABLA_CRC = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(d, ini, fin) {
    let c = 0xffffffff;
    for (let i = ini; i < fin; i++) c = TABLA_CRC[(c ^ d[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  const u32 = (d, p) => d[p] * 16777216 + ((d[p + 1] << 16) | (d[p + 2] << 8) | d[p + 3]);
  const ascii = (d, p, n) => String.fromCharCode(...d.subarray(p, p + n));
  // Un decimal como `f"{x:.1f}"` de Python: en un empate exacto (2,25) redondea al par; toFixed, hacia arriba.
  function fmt1(v) {
    const e = v * 10, base = Math.floor(e);
    if (e - base === 0.5) return ((base % 2 === 0 ? base : base + 1) / 10).toFixed(1);
    return v.toFixed(1);
  }
  const esNombre = (s) => /^\w{4}$/.test(s);           // como PIL: \w\w\w\w

  // (profundidad, tipo de color) -> modo de PIL y canales
  const MODOS = {
    "1,0": ["1", 1], "2,0": ["L", 1], "4,0": ["L", 1], "8,0": ["L", 1], "16,0": ["I;16", 1],
    "8,2": ["RGB", 3], "16,2": ["RGB", 3],
    "1,3": ["P", 1], "2,3": ["P", 1], "4,3": ["P", 1], "8,3": ["P", 1],
    "8,4": ["LA", 2], "16,4": ["RGBA", 2],              // PIL abre LA de 16 bits como RGBA
    "8,6": ["RGBA", 4], "16,6": ["RGBA", 4],
  };
  const CANALES_FICHERO = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

  // --- fase 1: hasta el primer IDAT (Image.open) ------------------------- //
  function abrir(d) {
    let p = 8, ihdr = null, paleta = null;
    for (;;) {
      if (d.length - p < 4) throw new Fallo("UnidentifiedImageError");
      if (d.length - p < 8) throw new Fallo("UnidentifiedImageError");
      const largo = u32(d, p), tipo = ascii(d, p + 4, 4);
      if (!esNombre(tipo)) throw new Fallo("UnidentifiedImageError");
      if (tipo === "IDAT") break;
      if (tipo === "IEND") throw new Fallo("UnidentifiedImageError");
      if (tipo === "IHDR" && largo < 13) throw new Fallo("ValueError");
      const fin = p + 8 + largo;
      if (fin > d.length) throw new Fallo("OSError");
      if (fin + 4 > d.length) throw new Fallo("UnidentifiedImageError");
      if (crc32(d, p + 4, fin) !== u32(d, fin)) throw new Fallo("UnidentifiedImageError");
      if (tipo === "IHDR") {
        const q = p + 8;
        ihdr = { w: u32(d, q), h: u32(d, q + 4), prof: d[q + 8], tipoColor: d[q + 9],
                 filtro: d[q + 11], entrelazado: d[q + 12] !== 0 };
        if (ihdr.filtro) throw new Fallo("UnidentifiedImageError");
      } else if (tipo === "PLTE") {
        paleta = d.slice(p + 8, fin);
      }
      p = fin + 4;
    }
    if (!ihdr || !MODOS[ihdr.prof + "," + ihdr.tipoColor] || ihdr.w <= 0 || ihdr.h <= 0) {
      throw new Fallo("UnidentifiedImageError");
    }
    return { ihdr, paleta, inicioIdat: p };
  }

  // --- fase 2: png.verify() (estructura y CRC de cada bloque hasta IEND) --- //
  function verificar(d, desde) {
    let p = desde;
    for (;;) {
      if (d.length - p < 4) throw new Fallo("OSError");               // struct.error -> "truncated PNG file"
      if (d.length - p < 8) throw new Fallo("SyntaxError");
      const largo = u32(d, p), tipo = ascii(d, p + 4, 4);
      if (!esNombre(tipo)) throw new Fallo("SyntaxError");
      if (tipo === "IEND") return;
      const fin = p + 8 + largo;
      if (fin > d.length) throw new Fallo("OSError");
      if (d.length - fin < 4) throw new Fallo("SyntaxError");
      if (crc32(d, p + 4, fin) !== u32(d, fin)) throw new Fallo("SyntaxError");
      p = fin + 4;
    }
  }

  // --- fase 3: png.load() ------------------------------------------------ //
  function datosIdat(d, desde) {
    const trozos = [];
    let p = desde, total = 0;
    while (p + 8 <= d.length && ascii(d, p + 4, 4) === "IDAT") {
      const largo = u32(d, p);
      trozos.push(d.subarray(p + 8, p + 8 + largo));
      total += largo;
      p += 12 + largo;
    }
    const todo = new Uint8Array(total);
    let o = 0;
    for (const t of trozos) { todo.set(t, o); o += t.length; }
    return todo;
  }

  const ADAM7 = [[0, 0, 8, 8], [0, 4, 8, 8], [4, 0, 8, 4], [0, 2, 4, 4], [2, 0, 4, 2], [0, 1, 2, 2], [1, 0, 2, 1]];

  /* (alto, ancho, canales) con un valor por muestra: el byte alto en 16 bits, el valor tal cual
   * en 1, 2 y 4 bits (sin escalar). Quita los filtros con UPNG y desentrelaza aqui. */
  function decodificar(idat, ihdr) {
    const { w, h, prof, tipoColor } = ihdr;
    const nch = CANALES_FICHERO[tipoColor];
    const bits = nch * prof;
    const falso = { ctype: tipoColor, depth: prof };
    let crudo;
    try {
      crudo = raiz.pako.inflate(idat);
    } catch (e) {
      throw new Fallo("OSError");
    }
    if (!crudo) throw new Fallo("OSError");
    const pasadas = ihdr.entrelazado
      ? ADAM7.map(([f0, c0, df, dc]) => [f0, c0, df, dc, Math.max(0, Math.ceil((h - f0) / df)),
                                         Math.max(0, Math.ceil((w - c0) / dc))])
      : [[0, 0, 1, 1, h, w]];
    let esperado = 0;
    for (const [, , , , sh, sw] of pasadas) if (sw * sh) esperado += sh * (1 + Math.ceil(sw * bits / 8));
    if (crudo.length < esperado) throw new Fallo("OSError");          // zlib o datos truncados
    const salida = new Uint8Array(w * h * nch);
    let di = 0;
    for (const [f0, c0, df, dc, sh, sw] of pasadas) {
      if (!(sw * sh)) continue;
      const bpl = Math.ceil(sw * bits / 8);
      for (let y = 0; y < sh; y++) if (crudo[di + y * (bpl + 1)] > 4) throw new Fallo("OSError");
      raiz.UPNG.decode._filterZero(crudo, falso, di, sw, sh);
      for (let y = 0; y < sh; y++) {
        const fila = di + y * bpl;
        for (let x = 0; x < sw; x++) {
          const destino = ((f0 + y * df) * w + (c0 + x * dc)) * nch;
          for (let c = 0; c < nch; c++) {
            const m = x * nch + c;
            let v;
            if (prof === 8) v = crudo[fila + m];
            else if (prof === 16) v = crudo[fila + 2 * m];
            else v = (crudo[fila + ((m * prof) >> 3)] >> (8 - prof - ((m * prof) & 7))) & ((1 << prof) - 1);
            salida[destino + c] = v;
          }
        }
      }
      di += sh * (1 + bpl);
    }
    return salida;
  }

  /* Los pixeles a escala de grises como `png.convert("L")` de PIL, con las reglas de la app:
   * L pasa tal cual; LA, RGB, RGBA y P solo si son gris (R = G = B); lo demas se rechaza. */
  function aGris(m, ihdr, paleta, fase) {
    const modo = MODOS[ihdr.prof + "," + ihdr.tipoColor][0];
    const n = LADO * LADO;
    const x = new Uint8Array(n);
    const avisos = [];
    if (modo === "L") {
      const escala = ihdr.prof === 2 ? 85 : ihdr.prof === 4 ? 17 : 1;
      for (let i = 0; i < n; i++) x[i] = m[i] * escala;
      return { x, avisos };
    }
    if (modo !== "LA" && modo !== "RGB" && modo !== "RGBA" && modo !== "P") {
      throw new ErrorEntrada(`${fase}: modo de imagen ${modo}; se espera un PNG en escala de grises de 8 bits, como los del dataset.`);
    }
    const color = new ErrorEntrada(`${fase}: la imagen tiene color; se espera escala de grises.`);
    if (modo === "LA") {
      for (let i = 0; i < n; i++) x[i] = m[2 * i];
    } else if (modo === "RGB" || modo === "RGBA") {
      const k = ihdr.tipoColor === 4 ? 2 : (modo === "RGB" ? 3 : 4);    // LA de 16 bits: gris + alfa
      for (let i = 0; i < n; i++) {
        if (ihdr.tipoColor === 4) { x[i] = m[k * i]; continue; }
        const r = m[k * i];
        if (r !== m[k * i + 1] || r !== m[k * i + 2]) throw color;
        x[i] = r;
      }
    } else {
      const pal = new Uint8Array(768);                                    // lo que falta de la paleta es negro (PIL)
      if (paleta) pal.set(paleta.subarray(0, Math.min(paleta.length, 768)));
      for (let i = 0; i < n; i++) {
        const j = 3 * m[i];
        if (pal[j] !== pal[j + 1] || pal[j] !== pal[j + 2]) throw color;
        x[i] = pal[j];
      }
    }
    avisos.push(`${fase}: PNG en modo ${modo} (gris guardado como ${modo}); se convierte a escala de grises igual que al entrenar.`);
    return { x, avisos };
  }

  /* Un PNG subido (Uint8Array) -> { x: Uint8Array(256*256), avisos }. Lanza ErrorEntrada. */
  function leerFase(datos, fase) {
    if (!(datos instanceof Uint8Array) || datos.length === 0) {
      throw new ErrorEntrada(`${fase}: el fichero esta vacio.`);
    }
    if (datos.length > TAMANO_MAXIMO) {
      throw new ErrorEntrada(`${fase}: ocupa ${fmt1(datos.length / 1024 ** 2)} MB; el maximo es ${(TAMANO_MAXIMO / 1024 ** 2).toFixed(0)} MB.`);
    }
    if (datos.length < 8 || FIRMA.some((b, i) => datos[i] !== b)) {
      throw new ErrorEntrada(`${fase}: no es un PNG (la cabecera no es la de PNG).`);
    }
    try {
      const { ihdr, paleta, inicioIdat } = abrir(datos);
      if (ihdr.w * ihdr.h > PIXELES_BOMBA) throw new Fallo("DecompressionBombError");
      if (ihdr.w !== LADO || ihdr.h !== LADO) {
        throw new ErrorEntrada(`${fase}: mide ${ihdr.w}x${ihdr.h} px; tiene que ser ${LADO}x${LADO} (no se redimensiona).`);
      }
      verificar(datos, inicioIdat);
      const muestras = decodificar(datosIdat(datos, inicioIdat), ihdr);
      return aGris(muestras, ihdr, paleta, fase);
    } catch (e) {
      if (e instanceof ErrorEntrada) throw e;
      if (e instanceof Fallo) {
        throw new ErrorEntrada(`${fase}: el PNG esta corrupto o no se puede leer (${e.tipo}).`);
      }
      throw new ErrorEntrada(`${fase}: el PNG esta corrupto o no se puede leer (${e && e.name || "Error"}).`);
    }
  }

  /* Ultima comprobacion antes de la red: tres fases de 256x256, finito, entero en 0-255. */
  function comprobarCorte(fases) {
    if (!Array.isArray(fases) || fases.length !== 3 || fases.some((f) => !(f instanceof Uint8Array) || f.length !== LADO * LADO)) {
      throw new ErrorEntrada(`El corte tiene forma distinta de (3, ${LADO}, ${LADO}).`);
    }
    return fases;
  }

  /* Tensor de la red: float32 (1, 3, 256, 256), cada valor entre 255. */
  function aEntrada(fases) {
    comprobarCorte(fases);
    const n = LADO * LADO;
    const t = new Float32Array(3 * n);
    for (let f = 0; f < 3; f++) for (let i = 0; i < n; i++) t[f * n + i] = fases[f][i] / 255;
    for (let i = 0; i < t.length; i++) {
      if (!Number.isFinite(t[i])) throw new ErrorEntrada("El corte tiene valores no finitos (NaN o infinito).");
    }
    return t;
  }

  /* Avisos que NO bloquean: fases posiblemente intercambiadas o nombres que no cuadran.
   * `nombres`: { PRE: "x_PRE.png", ... } (puede faltar). */
  function avisosCorte(fases, nombres) {
    const avisos = [];
    const media = (x) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s / x.length; };
    const pre = media(fases[0]), early = media(fases[1]);
    if (early <= pre) {
      avisos.push(`La media de EARLY (${fmt1(early)}) no supera a la de PRE (${fmt1(pre)}): al inyectar el contraste el tejido se ilumina, asi que puede que las fases esten intercambiadas.`);
    }
    for (const fase of FASES) {
      const nombre = nombres && nombres[fase];
      const m = /_(PRE|EARLY|LATE)\.png$/i.exec(nombre || "");
      if (m && m[1].toUpperCase() !== fase) {
        avisos.push(`El fichero subido como ${fase} se llama «${nombre}» (sufijo _${m[1].toUpperCase()}).`);
      }
    }
    return avisos;
  }

  async function sha256Hex(bytes) {
    const h = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    return Array.from(h, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  raiz.Png = { LADO, TAMANO_MAXIMO, FASES, ErrorEntrada, leerFase, comprobarCorte, aEntrada, avisosCorte, sha256Hex };
})(typeof window !== "undefined" ? window : globalThis);
