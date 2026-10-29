/* Mapa y curva de captacion en el navegador: el mismo calculo que src/visualizacion.py
 * (mapa_captacion, curva_captacion, tipo_curva, mascara_mama...). Son DESCRIPTIVOS: la red no los usa.
 *
 * NumPy hace parte de las cuentas en float32 (el cambio relativo, la comparacion con 0,1, el
 * logaritmo de Otsu); aqui se emula con Math.fround en los mismos puntos. La prueba
 * web/pruebas/visualizacion.html compara esto con Python en 24 cortes (los 4 ejemplos con las
 * fases en las 6 ordenes posibles). `x` es siempre [PRE, EARLY, LATE], cada una un
 * Uint8Array de 256 x 256.
 */
(function (raiz) {
  "use strict";

  const L = 256, N = L * L;
  const f32 = Math.fround;
  const CAMBIO_MESETA = 0.10;
  const FRACCION_ZONA = 0.02;
  const PERCENTIL_REALCE = 90;
  const REALCE_MINIMO = 20;
  const SATURADO = 250;
  const OPACIDAD = 0.6;
  const TIPOS = ["persistente", "meseta", "lavado"];
  const COLORES = { persistente: "#4E86CF", meseta: "#D4AE3C", lavado: "#CC4F50" };
  const DESCRIPCION = {
    persistente: "sigue subiendo de EARLY a LATE (más de +10 %)",
    meseta: "se mantiene de EARLY a LATE (entre −10 % y +10 %)",
    lavado: "baja de EARLY a LATE (más de −10 %)",
  };

  /* Filtro binomial 5x5, solo para ver. Con pesos multiplos de 1/16 y pixeles enteros, las
   * sumas son exactas en float32, asi que da lo mismo que NumPy. Bordes replicados. */
  function suavizar(img) {
    const k = [1, 4, 6, 4, 1];
    const tmp = new Float32Array(N), out = new Float32Array(N);
    const clamp = (i) => (i < 0 ? 0 : i > L - 1 ? L - 1 : i);
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        let s = 0;
        for (let i = 0; i < 5; i++) s += k[i] * img[clamp(y + i - 2) * L + x];
        tmp[y * L + x] = s / 16;
      }
    }
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        let s = 0;
        for (let i = 0; i < 5; i++) s += k[i] * tmp[y * L + clamp(x + i - 2)];
        out[y * L + x] = s / 16;
      }
    }
    return out;
  }

  /* Umbral de Otsu en 0-255 sobre una imagen ya en 0-255 (se trunca a entero, como astype(int64)). */
  function umbralOtsu(img) {
    const hist = new Float64Array(256);
    for (let i = 0; i < img.length; i++) {
      const v = Math.trunc(Math.min(Math.max(img[i], 0), 255));
      hist[v] += 1;
    }
    const total = img.length;
    let acumP = 0, acumMu = 0, muTotal = 0;
    for (let i = 0; i < 256; i++) muTotal += (hist[i] / total) * i;
    let mejor = -Infinity, indice = 0, hay = false;
    for (let i = 0; i < 256; i++) {
      const p = hist[i] / total;
      acumP += p;
      acumMu += p * i;
      const v = (muTotal * acumP - acumMu) ** 2 / (acumP * (1 - acumP));
      if (Number.isFinite(v) && v > mejor) { mejor = v; indice = i; hay = true; }
    }
    return hay ? indice : 0;
  }

  /* Tejido frente a fondo: EARLY suavizado por encima del Otsu de log(1 + intensidad). */
  function mascaraMama(x) {
    const early = suavizar(x[1]);
    const escala = 255 / Math.log(256);
    const logs = new Float32Array(N);
    for (let i = 0; i < N; i++) logs[i] = f32(f32(Math.log1p(Math.min(Math.max(early[i], 0), 255))) * f32(escala));
    const umbral = Math.expm1(umbralOtsu(logs) / escala);
    const mascara = new Uint8Array(N);
    let hay = false;
    for (let i = 0; i < N; i++) if (early[i] > umbral) { mascara[i] = 1; hay = true; }
    if (!hay) mascara.fill(1);
    return mascara;
  }

  /* (LATE - EARLY) / max(EARLY, 1) en float32. */
  function cambioRelativo(early, late) {
    const e = f32(early), l = f32(late);
    return f32(f32(l - e) / Math.max(e, 1));
  }

  /* Tipo de curva por el cambio EARLY -> LATE; con una fase saturada el cambio es una cota. */
  function tipoCurva(cambio, earlySat, lateSat) {
    const u = f32(CAMBIO_MESETA);
    if (earlySat && lateSat) return null;
    if (cambio > u) return earlySat ? null : "persistente";
    if (cambio < -u) return lateSat ? null : "lavado";
    return earlySat || lateSat ? null : "meseta";
  }

  /* np.percentile (interpolacion lineal) de un Float32Array. */
  function percentil(valores, q) {
    const v = Float32Array.from(valores).sort();
    const n = v.length;
    const virtual = (q / 100) * (n - 1);
    const lo = Math.floor(virtual), hi = Math.min(lo + 1, n - 1);
    const t = virtual - lo, a = v[lo], b = v[hi];
    const dif = b - a;
    return f32(t >= 0.5 ? b - dif * (1 - t) : a + dif * t);
  }

  /* round() de Python: los empates (x,5) van al par. */
  function redondearPar(v) {
    const r = Math.round(v);
    return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  function colorRgb(hex) {
    return [1, 3, 5].map((j) => parseInt(hex.slice(j, j + 2), 16));
  }

  /* EARLY en grises con color por tipo de curva donde mas realza. Devuelve rgb (RGBA para canvas),
   * tipo (Int8Array, -1 sin color), fracciones y recuentos. `paleta`: color por tipo (por defecto
   * el del diseno de la web; la prueba con Python usa el de la app de Streamlit). */
  function mapaCaptacion(x, paleta) {
    paleta = paleta || COLORES;
    const pre = suavizar(x[0]), early = suavizar(x[1]), late = suavizar(x[2]);
    const mama = mascaraMama(x);
    const realce = new Float32Array(N);
    const enMama = [];
    for (let i = 0; i < N; i++) {
      realce[i] = f32(early[i] - pre[i]);
      if (mama[i]) enMama.push(realce[i]);
    }
    const corte = f32(Math.max(REALCE_MINIMO, percentil(enMama, PERCENTIL_REALCE)));
    const u = f32(CAMBIO_MESETA);
    const tipo = new Int8Array(N).fill(-1);
    let nColoreados = 0, nSaturados = 0, nMama = 0;
    const cuenta = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      if (mama[i]) nMama++;
      if (!(mama[i] && realce[i] >= corte)) continue;
      const satE = x[1][i] >= SATURADO, satL = x[2][i] >= SATURADO;
      const c = cambioRelativo(early[i], late[i]);
      const sube = c > u, baja = c < -u;
      let t = -1;
      if (sube && !satE) t = 0;
      else if (!sube && !baja && !satE && !satL) t = 1;
      else if (baja && !satL) t = 2;
      if (t >= 0) { tipo[i] = t; cuenta[t]++; nColoreados++; } else nSaturados++;
    }
    const rgba = new Uint8ClampedArray(N * 4);
    const colores = TIPOS.map((t) => colorRgb(paleta[t]));
    const pasa = f32(1 - OPACIDAD), op = f32(OPACIDAD);
    for (let i = 0; i < N; i++) {
      const g = x[1][i];
      const t = tipo[i];
      for (let c = 0; c < 3; c++) {
        let v = g;
        if (t >= 0) v = f32(f32(pasa * g) + f32(op * colores[t][c]));
        rgba[i * 4 + c] = Math.min(255, Math.max(0, redondearPar(v)));
      }
      rgba[i * 4 + 3] = 255;
    }
    const fracciones = {};
    TIPOS.forEach((t, i) => { fracciones[t] = nColoreados ? cuenta[i] / nColoreados : 0; });
    return { rgba, tipo, fracciones, nColoreados, nSaturados, nMama };
  }

  /* Intensidad media en PRE, EARLY y LATE del 2 % de la mama que mas realza. */
  function curvaCaptacion(x, fraccion) {
    fraccion = fraccion === undefined ? FRACCION_ZONA : fraccion;
    const mama = mascaraMama(x);
    const pre = suavizar(x[0]), early = suavizar(x[1]);
    const indices = [];
    for (let i = 0; i < N; i++) if (mama[i]) indices.push(i);
    const realce = indices.map((i) => f32(early[i] - pre[i]));
    const n = indices.length;
    const k = Math.max(1, redondearPar(fraccion * n));
    const orden = Array.from({ length: n }, (_, j) => j)
      .sort((a, b) => realce[a] - realce[b] || a - b);           // estable: empates por posicion
    const zona = orden.slice(n - k).map((j) => indices[j]);
    const medias = {}, saturadas = {};
    ["PRE", "EARLY", "LATE"].forEach((nombre, f) => {
      let s = 0, sat = 0;
      for (const i of zona) { s += x[f][i]; if (x[f][i] >= SATURADO) sat++; }
      medias[nombre] = s / k;
      saturadas[nombre] = sat / k;
    });
    const cambio = cambioRelativo(medias.EARLY, medias.LATE);
    return { medias, cambio, nPixeles: k, tipo: tipoCurva(cambio, saturadas.EARLY > 0.5, saturadas.LATE > 0.5),
             saturadaEarly: saturadas.EARLY, saturadaLate: saturadas.LATE };
  }

  async function sha256Bytes(bytes) {
    return raiz.Png.sha256Hex(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  }

  raiz.Visualizacion = { TIPOS, COLORES, DESCRIPCION, FRACCION_ZONA, PERCENTIL_REALCE, SATURADO, suavizar,
                         mascaraMama, mapaCaptacion, curvaCaptacion, tipoCurva, cambioRelativo, sha256Bytes };
})(window);
