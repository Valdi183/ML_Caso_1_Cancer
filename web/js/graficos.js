/* Graficos de la web, en SVG y a partir de datos.json (nada escrito a mano). Cada funcion recibe los
 * datos y la anchura real del contenedor y devuelve el SVG: asi el texto mide lo mismo en un
 * movil que en un proyector. El color no decora: se distingue por trazo, forma y rotulo. Los
 * estilos (colores, tamaños de texto) estan en css/estilo.css (.g-*). */
(function (raiz) {
  "use strict";

  const C = raiz.Comun;
  const d1 = C.d1, d2 = C.d2, d3 = C.d3, d4 = C.d4, entero = C.entero, pct0 = C.pct0;

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const lin = (d0, d1_, r0, r1) => (v) => r0 + ((r1 - r0) * (v - d0)) / (d1_ - d0);
  const redondea = (v) => Math.round(v * 10) / 10;

  function svg(w, h, cuerpo, descripcion) {
    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(descripcion)}" ` +
           `style="max-width:${w}px" xmlns="http://www.w3.org/2000/svg">${cuerpo}</svg>`;
  }
  const texto = (x, y, t, clase, ancla, extra) =>
    `<text x="${redondea(x)}" y="${redondea(y)}" class="${clase || "g-t"}"` +
    `${ancla ? ` text-anchor="${ancla}"` : ""}${extra || ""}>${esc(t)}</text>`;
  const linea = (x1, y1, x2, y2, clase, extra) =>
    `<line x1="${redondea(x1)}" y1="${redondea(y1)}" x2="${redondea(x2)}" y2="${redondea(y2)}" class="${clase}"${extra || ""}/>`;
  const circulo = (x, y, r, clase, extra) => `<circle cx="${redondea(x)}" cy="${redondea(y)}" r="${r}" class="${clase}"${extra || ""}/>`;
  const rect = (x, y, w, h, clase, extra) =>
    `<rect x="${redondea(x)}" y="${redondea(y)}" width="${redondea(w)}" height="${redondea(h)}" class="${clase}"${extra || ""}/>`;
  const camino = (puntos, clase, extra) =>
    `<polyline points="${puntos.map(([x, y]) => `${redondea(x)},${redondea(y)}`).join(" ")}" class="${clase}"${extra || ""}/>`;

  /* Parte un texto en lineas de como mucho `max` caracteres, por " · " si lo hay y si no por espacios. */
  function partir(t, max, sep) {
    const piezas = t.split(sep || " · ");
    const lineas = [];
    let actual = "";
    for (const pieza of piezas) {
      const prueba = actual ? actual + (sep || " · ") + pieza : pieza;
      if (prueba.length > max && actual) { lineas.push(actual); actual = pieza; } else actual = prueba;
    }
    lineas.push(actual);
    return lineas;
  }

  /* Eje horizontal inferior con marcas y rejilla vertical suave. */
  function ejeX(x, y0, y1, marcas, formato, yTexto) {
    let s = linea(x(marcas[0]), y1, x(marcas[marcas.length - 1]), y1, "g-eje");
    for (const m of marcas) {
      s += linea(x(m), y0, x(m), y1, "g-rej") + linea(x(m), y1, x(m), y1 + 5, "g-eje") +
           texto(x(m), yTexto === undefined ? y1 + 24 : yTexto, formato(m), "g-t2", "middle");
    }
    return s;
  }

  function marcas(lo, hi, paso) {
    const r = [];
    for (let v = Math.ceil(lo / paso - 1e-9) * paso; v <= hi + 1e-9; v += paso) r.push(Math.round(v * 1e6) / 1e6);
    return r;
  }

  // ---------------------------------------------------------------------- //
  // 4. Como se eligio
  // ---------------------------------------------------------------------- //

  /* Las configuraciones probadas: AUC medio de validacion (punto grande) y el de cada fold (puntos
   * pequeños). Relleno: probadas en 4 o 5 folds; aro: en 1 o 2 (se mueven por ruido). */
  function configuraciones(d, W) {
    const filas = [...d.seleccion.configuraciones].sort((a, b) => b.auc_medio - a.auc_medio);
    const ancho = W >= 900;
    const etiquetaW = ancho ? Math.round(W * 0.46) : 0;
    const arriba = 54, abajo = 44, dcha = 14, izq = ancho ? etiquetaW + 12 : 14;
    const todos = filas.flatMap((c) => c.auc_por_fold);
    const lo = Math.floor((Math.min(...todos) - 0.004) * 50) / 50, hi = Math.ceil((Math.max(...todos) + 0.004) * 50) / 50;
    const x = lin(lo, hi, izq, W - dcha);
    const maxCar = Math.max(12, Math.floor(((ancho ? etiquetaW - 80 : W - 16)) / 9.2));
    // posicion de cada fila: en ancho, etiqueta a la izquierda (en 1-2 lineas); en estrecho, encima de los puntos
    let y = arriba;
    const pos = filas.map((c) => {
      const nFolds = `${c.folds.length} ${c.folds.length === 1 ? "fold" : "folds"}`;
      const lineas = ancho ? partir(c.etiqueta, maxCar) : partir(`${c.etiqueta} · ${nFolds}`, maxCar);
      const alto = ancho ? Math.max(36, 22 * lineas.length + 14) : 22 * lineas.length + 34;
      const fila = { c, nFolds, lineas, y0: y, alto, yc: ancho ? y + alto / 2 : y + alto - 14 };
      y += alto;
      return fila;
    });
    const fondo = y;
    const rango = d.seleccion.resumen.rango_4_o_5_folds;
    let s = rect(x(rango.min), arriba - 10, x(rango.max) - x(rango.min), fondo - arriba + 10, "g-banda");
    s += ejeX(x, arriba - 10, fondo, marcas(lo, hi, 0.02), d2, fondo + 26);
    s += linea(x(0.5), arriba - 10, x(0.5), fondo, "g-eje", ' stroke-dasharray="5 4"');
    s += texto(x(0.5), arriba - 18, "0,5 (azar)", "g-t2 g-halo", "middle");
    s += texto(Math.min(Math.max((x(rango.min) + x(rango.max)) / 2, 150), W - 150), 16,
               `franja: las de 4 o 5 folds, ${d3(rango.min)}\u2013${d3(rango.max)}`, "g-t2", "middle");
    for (const f of pos) {
      const c = f.c;
      s += linea(izq, f.yc, W - dcha, f.yc, "g-rej", ' stroke-opacity=".5"');
      if (ancho) {
        const y1 = f.yc - 11 * (f.lineas.length - 1) + 6;
        f.lineas.forEach((l, i) => { s += texto(0, y1 + 22 * i, l, c.congelado ? "g-t2 g-b" : "g-t2"); });
        s += texto(etiquetaW, y1, f.nFolds, "g-t2", "end");
      } else {
        f.lineas.forEach((l, i) => { s += texto(2, f.y0 + 20 + 22 * i, l, (c.congelado ? "g-t2 g-b" : "g-t2") + " g-halo"); });
      }
      for (const v of c.auc_por_fold) s += circulo(x(v), f.yc, 3.5, "g-relleno2", ' fill-opacity=".55"');
      if (c.folds.length >= 4) s += circulo(x(c.auc_medio), f.yc, 7.5, "g-relleno");
      else s += circulo(x(c.auc_medio), f.yc, 7, "g-fondo", ' stroke="var(--g-tx)" stroke-width="2.5"');
      if (c.congelado) {
        s += circulo(x(c.auc_medio), f.yc, 12.5, "g-caja");
        s += texto(Math.min(x(c.auc_medio) + 20, W - 130), f.yc - 14, `elegida ${d3(c.auc_medio)}`, "g-t g-b g-halo");
      }
    }
    const nota = `AUC medio de validación de ${filas.length} configuraciones, entre ${d3(Math.min(...filas.map((c) => c.auc_medio)))} y ` +
                 `${d3(Math.max(...filas.map((c) => c.auc_medio)))}`;
    return svg(W, fondo + abajo, s, nota);
  }

  /* Curvas de AUC de train y de validacion por epoca, un panel por fold. */
  function curvas(d, W) {
    const folds = d.seleccion.curvas;
    const cols = W >= 1000 ? 5 : W >= 620 ? 3 : 2;
    const hueco = 14, pw = (W - hueco * (cols - 1)) / cols, ph = 200, titulo = 30;
    const filas = Math.ceil(folds.length / cols);
    const todos = folds.flatMap((f) => [...f.auc_train, ...f.auc_val]);
    const lo = Math.floor((Math.min(...todos) - 0.005) * 20) / 20, hi = Math.ceil((Math.max(...todos) + 0.005) * 20) / 20;
    const maxEpoca = Math.max(...folds.map((f) => f.epoca[f.epoca.length - 1]));
    const paso = ph + titulo + 52, arriba = 44, alto = arriba + filas * paso;
    const margenIzq = 36;
    let s = "";
    // leyenda
    s += linea(0, 12, 34, 12, "g-trazo") + texto(42, 18, "validación", "g-t2");
    s += linea(158, 12, 192, 12, "g-trazo2", ' stroke-dasharray="6 4"') + texto(200, 18, "train", "g-t2");
    s += circulo(270, 12, 5.5, "g-relleno") + texto(282, 18, "época elegida", "g-t2");
    folds.forEach((f, i) => {
      const ox = (i % cols) * (pw + hueco), oy = arriba + Math.floor(i / cols) * paso;
      const x = lin(1, maxEpoca, ox + margenIzq, ox + pw - 6), y = lin(lo, hi, oy + titulo + ph, oy + titulo);
      s += texto(ox, oy + 16, `fold ${f.fold}`, "g-t g-b");
      s += texto(ox + pw, oy + 16, `época ${f.epoca_elegida}`, "g-t2", "end");
      for (const m of marcas(lo, hi, 0.1)) {
        s += linea(ox + margenIzq, y(m), ox + pw - 6, y(m), "g-rej") + texto(ox + margenIzq - 6, y(m) + 5, d1(m), "g-t2", "end");
      }
      s += linea(ox + margenIzq, y(0.5), ox + pw - 6, y(0.5), "g-eje", ' stroke-dasharray="4 4" stroke-opacity=".7"');
      s += linea(ox + margenIzq, oy + titulo + ph, ox + pw - 6, oy + titulo + ph, "g-eje");
      for (const m of [1, ...marcas(1, maxEpoca, 10).filter((v) => v > 5)]) {
        s += texto(x(m), oy + titulo + ph + 22, m, "g-t2", "middle");
      }
      s += camino(f.epoca.map((e, j) => [x(e), y(f.auc_train[j])]), "g-trazo2", ' stroke-dasharray="6 4"');
      s += camino(f.epoca.map((e, j) => [x(e), y(f.auc_val[j])]), "g-trazo");
      const j = f.epoca.indexOf(f.epoca_elegida);
      s += circulo(x(f.epoca_elegida), y(f.auc_val[j]), 6, "g-relleno");
    });
    return svg(W, alto, s, "AUC de train y de validación por época en cada uno de los 5 folds");
  }

  /* Perdida normal frente a ponderada: AUC medio (igual) y sensibilidad con corte 0,5 (muy distinta). */
  function perdida(d, W) {
    const p = d.seleccion.perdida;
    const etiquetaW = W >= 560 ? 190 : 0, izq = etiquetaW + (etiquetaW ? 6 : 12), dcha = 150;
    const filasDef = [["Pérdida normal", "normal"], ["Pérdida ponderada", "ponderada"]];
    const bloque = (titulo, y0, dominio, marcasX, formato, valor, ficha) => {
      const x = lin(dominio[0], dominio[1], izq, W - (W >= 560 ? dcha : 16));
      let s = texto(0, y0 + 20, titulo, "g-t g-b");
      const sup = y0 + 36;
      filasDef.forEach(([nombre, clave], i) => {
        const yc = sup + i * (etiquetaW ? 40 : 66) + (etiquetaW ? 20 : 44);
        if (etiquetaW) s += texto(0, yc + 5, nombre, "g-t");
        else s += texto(0, yc - 22, nombre, "g-t");
        const v = valor(p[clave]);
        s += linea(izq, yc, W - (W >= 560 ? dcha : 16), yc, "g-rej");
        for (const f of v.porFold) s += circulo(x(f), yc, 3.5, "g-relleno2", ' fill-opacity=".55"');
        s += linea(x(v.media - v.desviacion), yc, x(v.media + v.desviacion), yc, "g-trazo", ' stroke-width="3"');
        s += circulo(x(v.media), yc, 7.5, "g-relleno");
        const t = `${formato(v.media)} ± ${formato(v.desviacion)}`;
        if (W >= 560) s += texto(W - dcha + 10, yc + 5, t, "g-t");
        else s += texto(W - 16, yc - 22, t, "g-t", "end");
      });
      const yEje = sup + 2 * (etiquetaW ? 40 : 66) + 2;
      s += ejeX(x, sup, yEje, marcasX, d2, yEje + 24);
      return [s, yEje + 36];
    };
    const [a, hA] = bloque("AUC medio de validación (5 folds)", 0, [0.5, 0.6], [0.5, 0.55, 0.6], d3,
                           (q) => ({ media: q.auc_medio.media, desviacion: q.auc_medio.desviacion,
                                     porFold: q.auc_por_fold }));
    const [b, hB] = bloque("Sensibilidad con el corte 0,5", hA + 18, [0, 1], [0, 0.25, 0.5, 0.75, 1], d2,
                           (q) => ({ media: q.sensibilidad_05.media, desviacion: q.sensibilidad_05.desviacion,
                                     porFold: q.sensibilidad_05.por_fold }));
    return svg(W, hB, a + b, "AUC medio y sensibilidad con corte 0,5, con pérdida normal y ponderada");
  }

  /* Sensibilidad y especificidad frente al umbral (validacion fuera de fold, por paciente). */
  function umbral(d, W) {
    const u = d.seleccion.umbral, pts = u.curva;
    const izq = 52, dcha = 16, arriba = 44, alto = Math.min(380, Math.max(300, W * 0.5)), abajo = 54;
    const x = lin(0.15, 0.6, izq, W - dcha), y = lin(0, 1, arriba + alto, arriba);
    let s = "";
    for (const m of [0, 0.25, 0.5, 0.75, 1]) s += linea(izq, y(m), W - dcha, y(m), "g-rej") + texto(izq - 8, y(m) + 5, d2(m), "g-t2", "end");
    s += ejeX(x, arriba, arriba + alto, marcas(0.15, 0.6, 0.05).filter((_, i) => W >= 560 || i % 2 === 0), d2, arriba + alto + 24);
    s += texto(izq + (W - izq - dcha) / 2, arriba + alto + 48, "umbral (probabilidad por paciente)", "g-t2", "middle");
    const dentro = pts.filter((p) => p.umbral >= 0.15 && p.umbral <= 0.6);
    s += camino(dentro.map((p) => [x(p.umbral), y(p.especificidad)]), "g-trazo");
    s += camino(dentro.map((p) => [x(p.umbral), y(p.sensibilidad)]), "g-trazo2", ' stroke-dasharray="7 5"');
    const xu = x(u.valor);
    s += linea(xu, arriba - 8, xu, arriba + alto, "g-eje", ' stroke-width="2"');
    s += texto(Math.min(Math.max(xu, izq + 70), W - 90), arriba - 18, `umbral ${d4(u.valor)}`, "g-t g-b", "middle");
    s += circulo(xu, y(u.en_el_umbral.especificidad), 7, "g-relleno");
    s += circulo(xu, y(u.en_el_umbral.sensibilidad), 7, "g-fondo", ' stroke="var(--g-tx)" stroke-width="2.5"');
    const dcha_ = xu < W * 0.55;
    const tx = dcha_ ? xu + 14 : xu - 14, anc = dcha_ ? "start" : "end";
    s += texto(tx, y(u.en_el_umbral.especificidad) - 12, `especificidad ${d3(u.en_el_umbral.especificidad)}`, "g-t g-b", anc);
    s += texto(tx, y(u.en_el_umbral.sensibilidad) + 26, `sensibilidad ${d3(u.en_el_umbral.sensibilidad)}`, "g-t", anc);
    const fin = dentro[dentro.length - 1], ini = dentro[0];
    const en = (t) => dentro.reduce((a, b) => (Math.abs(b.umbral - t) < Math.abs(a.umbral - t) ? b : a));
    s += texto(x(0.5), y(en(0.5).especificidad) + 26, "especificidad", "g-t2", "middle");
    s += texto(x(0.2), y(en(0.2).sensibilidad) - 14, "sensibilidad", "g-t2", "middle");
    return svg(W, arriba + alto + abajo, s, "Sensibilidad y especificidad en validación en función del umbral; umbral elegido " + d4(u.valor));
  }

  // ---------------------------------------------------------------------- //
  // 5. Resultados en test
  // ---------------------------------------------------------------------- //

  function roc(d, W) {
    const r = d.test.roc, t = d.test.auc;
    const lado = Math.min(W, 520), izq = 60, abajo = 58, arriba = 12;
    const area = lado - izq - 12;
    const x = lin(0, 1, izq, izq + area), y = lin(0, 1, arriba + area, arriba);
    let s = "";
    for (const m of [0, 0.25, 0.5, 0.75, 1]) {
      s += linea(izq, y(m), izq + area, y(m), "g-rej") + texto(izq - 8, y(m) + 5, d2(m), "g-t2", "end");
    }
    s += ejeX(x, arriba, arriba + area, [0, 0.25, 0.5, 0.75, 1], d2, arriba + area + 24);
    s += texto(izq + area / 2, arriba + area + 48, "1 − especificidad (falsos positivos)", "g-t2", "middle");
    s += `<text transform="translate(14 ${arriba + area / 2}) rotate(-90)" class="g-t2" text-anchor="middle">sensibilidad</text>`;
    s += linea(x(0), y(0), x(1), y(1), "g-eje", ' stroke-dasharray="6 5"');
    s += camino(r.fpr.map((f, i) => [x(f), y(r.tpr[i])]), "g-trazo");
    const p = r.punto;
    s += circulo(x(p.fpr), y(p.tpr), 8, "g-relleno");
    s += linea(x(p.fpr), y(p.tpr) - 10, x(0.07), y(0.62), "g-eje");
    s += texto(x(0.03), y(0.69), `umbral ${d4(p.umbral)}`, "g-t g-b");
    s += texto(x(0.03), y(0.69) + 22, `sensibilidad ${d3(p.tpr)}`, "g-t2");
    s += texto(x(0.03), y(0.69) + 42, `especificidad ${d3(1 - p.fpr)}`, "g-t2");
    s += texto(izq + area - 4, y(0) - 36, `AUC ${d3(t.auc)}`, "g-t g-b", "end");
    s += texto(izq + area - 4, y(0) - 12, `[${d3(t.ic95[0])}–${d3(t.ic95[1])}]`, "g-t", "end");
    s += texto(x(0.5) + 8, y(0.5) - 8, "azar", "g-t2");
    return svg(lado, arriba + area + abajo, s, `Curva ROC en test: AUC ${d3(t.auc)}, intervalo ${d3(t.ic95[0])} a ${d3(t.ic95[1])}`);
  }

  /* AUC por cohorte y global, con su intervalo, sobre la linea del azar. */
  function cohortesTest(d, W) {
    const t = d.test;
    const filas = [{ nombre: "Todas", n: t.pacientes.n, pcr: t.pacientes.pcr, auc: t.auc.auc, ic: t.auc.ic95, fuerte: true },
                   ...d.datos.cohortes.map((c) => {
                     const q = t.por_cohorte[c.id];
                     return { nombre: c.nombre, n: q.pacientes, pcr: q.pcr, auc: q.auc, ic: q.ic95 };
                   })];
    return bosque(W, filas.map((f) => ({
      etiqueta: f.nombre, detalle: `${entero(f.n)} pacientes, ${entero(f.pcr)} con pCR`, valor: f.auc, ic: f.ic,
      forma: "circulo", relleno: true, fuerte: f.fuerte, texto: `${d3(f.auc)} [${d3(f.ic[0])}–${d3(f.ic[1])}]` })),
      { dominio: [0, 1], marcas: [0, 0.25, 0.5, 0.75, 1], ref: 0.5, refTexto: "azar", titulo: null },
      "AUC en test por cohorte, con intervalo del 95 %");
  }

  /* Comparacion: este modelo, la referencia de la asignatura, Fridman y las lineas base; abajo, la
   * clinica y la red base en validacion. La referencia de la asignatura va sin intervalo: no se conoce. */
  function comparacion(d, W) {
    const t = d.test, refs = d.comparacion.referencias, cmp = d.comparacion.clinica_e_imagen;
    const base = d.comparacion.baselines.red_base;
    const filas = [
      { grupo: `En test (${entero(t.pacientes.n)} pacientes, el mismo conjunto)` },
      { etiqueta: "Este modelo", detalle: "CNN propia, un corte 2D", valor: t.auc.auc, ic: t.auc.ic95, forma: "circulo", relleno: true,
        fuerte: true, texto: `${d3(t.auc.auc)} [${d3(t.auc.ic95[0])}–${d3(t.auc.ic95[1])}]` },
      { etiqueta: "Modelo de referencia de la asignatura", detalle: "intervalo desconocido", valor: refs.referencia_asignatura.auc_test,
        ic: null, forma: "cuadrado", relleno: true, texto: d2(refs.referencia_asignatura.auc_test) },
      { etiqueta: "Fridman et al. 2025", detalle: "ViT preentrenado: cota superior", valor: refs.fridman_2025.auc_test, ic: null,
        forma: "rombo", relleno: false, texto: d2(refs.fridman_2025.auc_test) },
      { etiqueta: "Siempre «no pCR»", detalle: "línea base: no ordena a nadie", valor: d.comparacion.baselines.siempre_no_pcr.auc,
        ic: null, forma: "circulo", relleno: false, texto: d2(d.comparacion.baselines.siempre_no_pcr.auc) },
      { grupo: "En validación, 5 folds (solo comparación; media ± desviación)" },
      { etiqueta: "Imagen sola (esta red)", detalle: "media de todas las épocas", valor: cmp.imagen_todas_las_epocas,
        ic: null, forma: "circulo", relleno: false, texto: d3(cmp.imagen_todas_las_epocas) },
      { etiqueta: "Red base de 4 bloques", detalle: `línea base; ${base.folds.length === 1 ? "solo el fold " + base.folds[0] : base.folds.length + " folds"}`,
        valor: base.auc_medio, ic: null, forma: "circulo", relleno: false, texto: d3(base.auc_medio) },
      { etiqueta: "Variables clínicas solas", detalle: "subtipo, edad y volumen", valor: cmp.clinica.media,
        ic: [cmp.clinica.media - cmp.clinica.desviacion, cmp.clinica.media + cmp.clinica.desviacion], forma: "circulo", relleno: false,
        texto: `${d3(cmp.clinica.media)} ± ${d3(cmp.clinica.desviacion)}` },
      { etiqueta: "Clínicas más imagen", detalle: "la app no pide datos clínicos", valor: cmp.combinada.media,
        ic: [cmp.combinada.media - cmp.combinada.desviacion, cmp.combinada.media + cmp.combinada.desviacion], forma: "circulo", relleno: false,
        texto: `${d3(cmp.combinada.media)} ± ${d3(cmp.combinada.desviacion)}` },
    ];
    return bosque(W, filas, { dominio: [0.35, 0.8], marcas: [0.4, 0.5, 0.6, 0.7, 0.8], ref: 0.5, refTexto: "azar" },
                  "AUC de este modelo en test comparado con la referencia de la asignatura, Fridman, las líneas base y las variables clínicas");
  }

  /* Los 5 modelos del conjunto: una fila por modelo y una columna por fold. Relleno: folds con los
   * que aprendio; caja vacia: el fold que no vio (su validacion). */
  function folds(d, W) {
    const pesos = d.modelo.pesos, n = pesos.length;
    const ancho = W >= 560;
    const izq = ancho ? 112 : 84, arriba = 34, alto = 34, hueco = 8;
    const celda = Math.min(96, Math.floor((W - izq - 8) / n) - hueco);
    let s = "";
    for (let f = 0; f < n; f++) {
      s += texto(izq + f * (celda + hueco) + celda / 2, 18, `fold ${f}`, "g-t2", "middle");
    }
    pesos.forEach((p, i) => {
      const y = arriba + i * (alto + hueco);
      s += texto(0, y + alto / 2 + 6, `Modelo ${i + 1}`, "g-t g-b");
      for (let f = 0; f < n; f++) {
        const x = izq + f * (celda + hueco);
        if (f === p.fold) {
          s += rect(x, y, celda, alto, "g-fondo", ' stroke="var(--g-tx)" stroke-width="2" stroke-dasharray="5 3"');
          s += texto(x + celda / 2, y + alto / 2 + 6, ancho ? "no lo ve" : "fuera", "g-t2", "middle");
        } else {
          s += rect(x, y, celda, alto, "g-relleno2", ' fill-opacity=".35"');
        }
      }
    });
    const y = arriba + n * (alto + hueco) + 10;
    s += rect(izq, y, 18, 14, "g-relleno2", ' fill-opacity=".35"') + texto(izq + 26, y + 12, "aprende con este fold", "g-t2");
    const x2 = izq + (ancho ? 240 : 0), y2 = ancho ? y : y + 24;
    s += rect(x2, y2, 18, 14, "g-fondo", ' stroke="var(--g-tx)" stroke-width="2" stroke-dasharray="5 3"') +
         texto(x2 + 26, y2 + 12, "no lo ve (lo usa para validar)", "g-t2");
    return svg(W, y2 + 26, s, `Los ${n} modelos: cada uno aprende con cuatro folds y no ve el quinto`);
  }

  /* Grafico de intervalos: una fila por estimacion, con etiqueta, valor e intervalo. */
  function bosque(W, filas, o, descripcion) {
    const ancho = W >= 900;
    const etiquetaW = ancho ? Math.round(W * 0.4) : 0;
    const dcha = ancho ? 190 : 12, izq = etiquetaW + (ancho ? 12 : 8);
    const x = lin(o.dominio[0], o.dominio[1], izq, W - dcha);
    const maxCar = Math.max(14, Math.floor((W - 12) / 9));
    const lineasGrupo = (f) => (ancho ? [f.grupo] : partir(f.grupo, maxCar, " "));
    const alturas = filas.map((f) => (f.grupo ? 22 + 24 * lineasGrupo(f).length : ancho ? 52 : 82));
    const arriba = 30;
    let y0 = arriba, s = "";
    const total = alturas.reduce((a, b) => a + b, 0);
    const baseEje = arriba + total;
    s += ejeX(x, arriba - 6, baseEje, o.marcas, d2, baseEje + 26);
    if (o.ref !== undefined) {
      s += linea(x(o.ref), arriba - 6, x(o.ref), baseEje, "g-eje", ' stroke-dasharray="5 4"');
      s += texto(x(o.ref), 14, o.refTexto, "g-t2", "middle");
    }
    filas.forEach((f, i) => {
      const h = alturas[i];
      if (f.grupo) {
        lineasGrupo(f).forEach((l, j) => { s += texto(0, y0 + 30 + 24 * j, l, "g-t g-b g-halo"); });
        y0 += h;
        return;
      }
      const yc = ancho ? y0 + h / 2 : y0 + h - 22;
      const peso = f.fuerte ? " g-b" : "";
      if (ancho) {
        s += texto(0, yc - 4, f.etiqueta, "g-t" + peso);
        if (f.detalle) s += texto(0, yc + 18, f.detalle, "g-t2");
      } else {
        s += texto(0, y0 + 20, f.etiqueta, "g-t g-halo" + peso);
        if (f.detalle) s += texto(0, y0 + 40, f.detalle, "g-t2 g-halo");
      }
      s += linea(izq, yc, W - dcha, yc, "g-rej", ' stroke-opacity=".5"');
      if (f.ic) {
        s += linea(x(f.ic[0]), yc, x(f.ic[1]), yc, "g-trazo", ' stroke-width="3"');
        s += linea(x(f.ic[0]), yc - 7, x(f.ic[0]), yc + 7, "g-trazo") + linea(x(f.ic[1]), yc - 7, x(f.ic[1]), yc + 7, "g-trazo");
      }
      const cx = x(f.valor), clase = f.relleno ? "g-relleno" : "g-fondo", marco = f.relleno ? "" : ' stroke="var(--g-tx)" stroke-width="2.5"';
      if (f.forma === "cuadrado") s += rect(cx - 8, yc - 8, 16, 16, clase, marco);
      else if (f.forma === "rombo") s += `<path d="M${redondea(cx)},${yc - 10} L${redondea(cx + 10)},${yc} L${redondea(cx)},${yc + 10} L${redondea(cx - 10)},${yc} Z" class="${clase}"${marco}/>`;
      else s += circulo(cx, yc, 8, clase, marco);
      if (ancho) s += texto(W - dcha + 14, yc + 5, f.texto, "g-t" + peso);
      else s += texto(W - 8, y0 + 20, f.texto, "g-t g-halo" + peso, "end");
      y0 += h;
    });
    return svg(W, baseEje + 40, s, descripcion);
  }

  function calibracion(d, W) {
    const f = d.test.fiabilidad, base = d.test.tasa_pcr_train, cal = d.test.calibracion;
    const lado = Math.min(W, 520), izq = 60, abajo = 58, arriba = 12, area = lado - izq - 12, tope = 0.7;
    const x = lin(0, tope, izq, izq + area), y = lin(0, tope, arriba + area, arriba);
    let s = "";
    const m = [0, 0.2, 0.4, 0.6];
    for (const v of m) s += linea(izq, y(v), izq + area, y(v), "g-rej") + texto(izq - 8, y(v) + 5, d1(v), "g-t2", "end");
    s += ejeX(x, arriba, arriba + area, m, d1, arriba + area + 24);
    s += texto(izq + area / 2, arriba + area + 48, "probabilidad media predicha", "g-t2", "middle");
    s += `<text transform="translate(14 ${arriba + area / 2}) rotate(-90)" class="g-t2" text-anchor="middle">pacientes con pCR (observado)</text>`;
    s += linea(x(0), y(0), x(tope), y(tope), "g-eje", ' stroke-dasharray="6 5"');
    s += texto(x(0.5) + 10, y(0.5) + 26, "calibración perfecta", "g-t2");
    s += linea(izq, y(base), izq + area, y(base), "g-eje", ' stroke-dasharray="2 4"');
    s += texto(izq + area - 4, y(base) - 8, `tasa base ${d3(base)}`, "g-t2", "end");
    for (const g of f) s += circulo(x(g.prob_media), y(g.observada), 7, "g-relleno");
    return svg(lado, arriba + area + abajo, s, `Calibración en test: pendiente ${d2(cal.pendiente)}, ${f.length} grupos`);
  }

  /* Fraccion de cortes con la zona que mas realza saturada a 255 en EARLY y en LATE. */
  function saturacion(d, W) {
    const s0 = d.saturacion;
    const filas = [{ nombre: "Todos los cortes", q: s0.todas },
                   ...d.datos.cohortes.map((c) => ({ nombre: c.nombre, q: s0.por_cohorte[c.id] }))];
    const ancho = W >= 620, izq = ancho ? 150 : 6, dcha = ancho ? 70 : 60;
    const x = lin(0, 1, izq, W - dcha), filaH = ancho ? 64 : 92, arriba = 34;
    let s = "";
    s += linea(izq, 22, izq, arriba + filas.length * filaH, "g-eje");
    for (const m of [0, 0.25, 0.5, 0.75, 1]) s += texto(x(m), arriba + filas.length * filaH + 24, pct0(m), "g-t2", "middle") + linea(x(m), 22, x(m), arriba + filas.length * filaH, "g-rej");
    s += rect(izq + 2, 4, 14, 14, "g-relleno") + texto(izq + 22, 17, "EARLY", "g-t2") +
         rect(izq + 100, 4, 14, 14, "g-relleno2") + texto(izq + 120, 17, "LATE", "g-t2");
    filas.forEach((f, i) => {
      const y0 = arriba + i * filaH + 8;
      if (ancho) s += texto(0, y0 + 26, f.nombre, "g-t");
      else s += texto(0, y0 - 2, f.nombre, "g-t");
      const off = ancho ? 0 : 6;
      const e = f.q.early_255.fraccion_cortes, l = f.q.late_255.fraccion_cortes;
      s += rect(izq, y0 + off, x(e) - izq, 20, "g-relleno") + texto(x(e) + 8, y0 + off + 16, pct0(e), "g-t g-b");
      s += rect(izq, y0 + off + 24, x(l) - izq, 20, "g-relleno2") + texto(x(l) + 8, y0 + off + 40, pct0(l), "g-t");
    });
    return svg(W, arriba + filas.length * filaH + 40, s, "Fracción de cortes de train con la zona que más realza saturada en EARLY y en LATE");
  }

  /* Monta un grafico en un contenedor y lo redibuja al cambiar su anchura. */
  function montar(contenedor, dibujar, datos) {
    let anterior = -1, esperando = false;
    const pintar = () => {
      esperando = false;
      const w = Math.floor(contenedor.clientWidth);
      if (w < 200 || w === anterior) return;
      anterior = w;
      contenedor.innerHTML = dibujar(datos, w);
    };
    pintar();
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => { if (!esperando) { esperando = true; requestAnimationFrame(pintar); } }).observe(contenedor);
    }
  }

  raiz.Graficos = { configuraciones, curvas, perdida, umbral, roc, cohortesTest, comparacion, calibracion, saturacion, folds,
                    bosque, montar, esc, lin, marcas, svg, texto, linea, circulo, rect, camino, ejeX };
})(window);
