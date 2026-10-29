/* Pagina de la prediccion (predecir.html): ficheros o ejemplos -> png.js (lectura y validacion) ->
 * visualizacion.js (mapa y curva, descriptivos) y modelo.js (la prediccion). Hace lo que hace hoy la app
 * de Streamlit: visor de las tres fases, mapa y curva de captacion, resultado con su escala y los 5
 * modelos, «¿cuanto fiarse?», comparacion con las clinicas, detalles y el JSON descargable.
 * Las cifras salen de datos.json y modelo.json; ninguna esta escrita aqui. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const C = window.Comun, G = window.Graficos, V = window.Visualizacion;
  const FASES = window.Png.FASES;
  const COHORTES = { spy1: "I-SPY1", spy2: "I-SPY2", duke: "Duke" };
  let meta = null, d = null, huella = "", umbralesEntrada = null;
  let ultimo = null;

  // ---------------------------------------------------------------------- //
  // Errores: el mensaje de la app tal cual y, debajo, que hacer (como mensaje_error de visualizacion.py)
  // ---------------------------------------------------------------------- //
  const ERRORES = [
    [/mide (\d+)x(\d+) px/, (f, m) => `${f} mide ${m[1]}×${m[2]} px; la web necesita 256×256, el tamaño con el que se entrenó el modelo. Sube el PNG original del dataset, sin redimensionar.`],
    [/vacio/, (f) => `El fichero de ${f} está vacío. Vuelve a exportarlo y súbelo otra vez.`],
    [/ocupa ([\d.]+) MB/, (f, m) => `${f} ocupa ${m[1].replace(".", ",")} MB y el máximo es 2 MB. Los PNG del dataset ocupan mucho menos: comprueba que es el fichero correcto.`],
    [/no es un PNG|no PNG/, (f) => `${f} no es un PNG. Sube el fichero .png de esa fase (no JPEG ni DICOM).`],
    [/tiene color/, (f) => `${f} tiene color; la web necesita una imagen en escala de grises. Sube el PNG original de la fase, sin colorear.`],
    [/modo de imagen (\S+);/, (f, m) => `${f} no es un PNG de grises de 8 bits (modo ${m[1]}). Sube el PNG original del dataset.`],
    [/corrupto/, (f) => `${f} está dañado o incompleto y no se puede leer. Descárgalo de nuevo.`],
  ];

  function ayudaError(texto) {
    const fase = /^(PRE|EARLY|LATE): /.exec(texto);
    for (const [patron, hacer] of ERRORES) {
      const m = patron.exec(texto);
      if (fase && m) return hacer(fase[1], m);
    }
    return "";
  }

  function mostrarError(texto) {
    const ayuda = ayudaError(texto);
    $("error").innerHTML = `<strong>${G.esc(texto)}</strong>` + (ayuda ? `<span class="ayuda">${G.esc(ayuda)}</span>` : "");
    $("error").hidden = false;
    limpiarSalida();
  }

  function limpiar() {
    $("error").hidden = true;
    $("avisos").textContent = "";
  }

  function limpiarSalida() {
    $("bloque-resultado").hidden = true;
    $("probabilidad").textContent = "";
    $("leyenda").hidden = true;
    $("nota-visor").hidden = true;
    $("vacio").hidden = false;
    for (const id of ["v-pre", "v-early", "v-late", "v-mapa"]) {
      const c = $(id);
      c.getContext("2d").clearRect(0, 0, c.width, c.height);
    }
    for (const f of FASES) $("n-" + f.toLowerCase()).textContent = "pendiente";
    ultimo = null;
  }

  // ---------------------------------------------------------------------- //
  // Visor
  // ---------------------------------------------------------------------- //
  function pintarGris(canvas, x) {
    const ctx = canvas.getContext("2d");
    const im = ctx.createImageData(256, 256);
    for (let i = 0; i < 65536; i++) { im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = x[i]; im.data[4 * i + 3] = 255; }
    ctx.putImageData(im, 0, 0);
  }

  function pintarVisor(fases, nombres, mapa) {
    FASES.forEach((f, i) => {
      pintarGris($("v-" + f.toLowerCase()), fases[i]);
      $("n-" + f.toLowerCase()).textContent = nombres[f] || "";
    });
    $("v-mapa").getContext("2d").putImageData(new ImageData(mapa.rgba, 256, 256), 0, 0);
    const pct = (v) => C.pct0(v);
    const ley = V.TIPOS.map((t) => `<span><i style="background:${V.COLORES[t]}"></i><b>${t}</b> ${pct(mapa.fracciones[t])}: ${V.DESCRIPCION[t]}</span>`).join("") +
      `<span><i style="background:#fff"></i><b>sin color</b>: satura (255) y su tipo no se puede medir</span>`;
    $("leyenda").innerHTML = ley;
    $("leyenda").hidden = false;
    $("nota-visor").innerHTML = `Mapa de captación: el ${100 - V.PERCENTIL_REALCE} % de la mama que más realza de PRE a EARLY, coloreado por cómo cambia de EARLY a LATE (suavizado para verlo). <strong>Es descriptivo: el modelo no lo usa.</strong> El momento en que se toma LATE cambia según el hospital, así que el tipo no es comparable entre cohortes.`;
    $("nota-visor").hidden = false;
    $("vacio").hidden = true;
  }

  // ---------------------------------------------------------------------- //
  // Graficos con la anchura real del contenedor
  // ---------------------------------------------------------------------- //
  function vivo(el) {
    let fn = null, ancho = -1;
    const pintar = () => {
      if (!fn) return;
      const w = Math.floor(el.clientWidth);
      if (w < 200) return;
      ancho = w;
      el.innerHTML = fn(w);
    };
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => { if (Math.floor(el.clientWidth) !== ancho) requestAnimationFrame(pintar); }).observe(el);
    }
    return (f) => { fn = f; ancho = -1; pintar(); };
  }

  const comaN = (v, n) => C.conComa(v.toFixed(n));

  /* Escala de 0 a 1: el corte (triangulo), los 5 modelos (puntos; aro el que no vio a la paciente), el
   * umbral (linea continua) y la tasa base (discontinua). Cada marca tiene forma y rotulo propios. */
  function escalaSvg(W, prob, umbral, tasa, porModelo, destacado) {
    const izq = 16, dcha = 16, x = (p) => izq + (W - izq - dcha) * Math.min(Math.max(p, 0), 1);
    const yB = 50, hB = 28, xu = x(umbral), xp = x(prob);
    let s = `<rect x="${izq}" y="${yB}" width="${(xu - izq).toFixed(1)}" height="${hB}" fill="#3D4045"/>` +
            `<rect x="${xu.toFixed(1)}" y="${yB}" width="${(W - dcha - xu).toFixed(1)}" height="${hB}" fill="#55595F"/>`;
    if (xu - izq > 150) s += G.texto(izq + 8, yB + 20, "no predice pCR", "g-t2");
    if (W - dcha - xu > 130) s += G.texto(W - dcha - 8, yB + 20, "predice pCR", "g-t2", "end");
    const yM = yB + hB + 20, yN = yB + hB + 54;               // fila de los modelos y fila de los numeros
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      s += G.linea(x(t), yM + 14, x(t), yM + 20, "g-eje");
      s += G.texto(x(t), yN, t === 0 ? "0" : t === 1 ? "1" : C.conComa(String(t)), "g-t2", "middle");
    }
    s += G.linea(izq, yM + 14, W - dcha, yM + 14, "g-eje");
    if (tasa !== null) {
      s += G.linea(x(tasa), yB - 8, x(tasa), yM + 14, "g-eje", ' stroke-dasharray="4 3" stroke-width="1.5"');
    }
    s += G.linea(xu, yB - 8, xu, yM + 14, "g-trazo", ' stroke-width="2"');
    porModelo.forEach((v, i) => { if (i !== destacado) s += G.circulo(x(v), yM, 4.5, "g-relleno2", ' fill-opacity=".9"'); });
    if (destacado !== null && destacado !== undefined) s += G.circulo(x(porModelo[destacado]), yM, 8, "g-fondo", ' stroke="var(--g-tx)" stroke-width="2.5"');
    s += `<path d="M${(xp - 10).toFixed(1)},22 L${(xp + 10).toFixed(1)},22 L${xp.toFixed(1)},44 Z" class="g-relleno"/>`;
    const xt = Math.min(Math.max(xp, izq + 50), W - dcha - 50);
    s += G.texto(xt, 14, "este corte", "g-t g-b", "middle");
    // leyenda de las dos lineas, apilada a la izquierda: con anchuras pequeñas los rotulos junto a cada linea se salen
    const yR = yB + hB + 84;
    s += G.linea(izq, yR - 5, izq + 28, yR - 5, "g-trazo", ' stroke-width="2"') + G.texto(izq + 38, yR, `umbral ${C.d4(umbral)}`, "g-t g-b");
    if (tasa !== null) {
      s += G.linea(izq, yR + 21, izq + 28, yR + 21, "g-eje", ' stroke-dasharray="4 3" stroke-width="1.5"');
      s += G.texto(izq + 38, yR + 26, `tasa base ${C.pct1(tasa)}`, "g-t2");
    }
    const desc = `Escala de probabilidad de 0 a 1. Este corte: ${C.d4(prob)}. Umbral: ${C.d4(umbral)}. ` +
      (tasa !== null ? `Tasa base: ${C.pct1(tasa)}. ` : "") + "Modelos: " + porModelo.map((v) => C.d3(v)).join("; ") + ".";
    return G.svg(W, yR + (tasa !== null ? 42 : 20), s, desc);
  }

  /* Curva de captacion: PRE, EARLY y LATE (0-255) de la zona que mas realza, con el tipo escrito al final. */
  function curvaSvg(W, curva) {
    const izq = 46, dcha = W >= 420 ? 128 : 100, arriba = 16, alto = 190, base = arriba + alto;
    const y = (v) => base - (alto * v) / 255;
    const xs = { PRE: izq + 14, EARLY: izq + (W - izq - dcha) / 2, LATE: W - dcha - 14 };
    const sub = { PRE: "antes", EARLY: "temprana", LATE: "tardía" };
    let s = "";
    for (const t of [0, 64, 128, 192]) {
      s += G.linea(izq, y(t), W - dcha, y(t), "g-rej") + G.texto(izq - 8, y(t) + 5, t, "g-t2", "end");
    }
    s += G.linea(izq, y(255), W - dcha, y(255), "g-eje", ' stroke-dasharray="4 3"') + G.texto(izq - 8, y(255) + 5, 255, "g-t2", "end");
    s += G.texto(W - dcha + 6, y(255) + 5, "techo", "g-t2");
    const color = curva.tipo ? V.COLORES[curva.tipo] : "var(--g-tx2)";
    const puntos = ["PRE", "EARLY", "LATE"].map((f) => `${xs[f].toFixed(1)},${y(curva.medias[f]).toFixed(1)}`).join(" ");
    s += `<polyline points="${puntos}" fill="none" stroke="${color}" stroke-width="3.5"${curva.tipo ? "" : ' stroke-dasharray="6 4"'}/>`;
    for (const f of ["PRE", "EARLY", "LATE"]) {
      const v = curva.medias[f];
      s += `<circle cx="${xs[f].toFixed(1)}" cy="${y(v).toFixed(1)}" r="6" fill="${color}" stroke="var(--visor-sup)" stroke-width="2"/>`;
      s += G.texto(xs[f], Math.min(y(v) + 24, base - 6), v.toFixed(0), "g-t", "middle");
      s += G.texto(xs[f], base + 26, f, "g-t g-b", "middle");
      s += G.texto(xs[f], base + 46, sub[f], "g-t2", "middle");
    }
    const yl = y(curva.medias.LATE);
    const signo = curva.cambio >= 0 ? "+" : C.MENOS;
    s += G.texto(W - dcha + 6, yl + 22, curva.tipo || "sin tipo", "g-t g-b");
    s += G.texto(W - dcha + 6, yl + 44, `${signo}${C.pct0(Math.abs(curva.cambio))}`, "g-t2");
    const desc = `Curva de captación. PRE ${curva.medias.PRE.toFixed(0)}, EARLY ${curva.medias.EARLY.toFixed(0)}, LATE ${curva.medias.LATE.toFixed(0)}. ` +
      `Cambio de EARLY a LATE ${signo}${C.pct0(Math.abs(curva.cambio))}: ${curva.tipo || "sin tipo"}.`;
    return G.svg(W, base + 56, s, desc);
  }

  const dibujarEscala = vivo($("escala"));
  const dibujarCurva = vivo($("curva"));

  // ---------------------------------------------------------------------- //
  // Prediccion
  // ---------------------------------------------------------------------- //
  async function bytesDe(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("No se pudo leer " + url);
    return new Uint8Array(await r.arrayBuffer());
  }

  async function sha256Texto(t) {
    return window.Png.sha256Hex(new TextEncoder().encode(t));
  }

  function resultadoJson(r, nombres) {
    const sha = {};
    meta.modelo.pesos.forEach((p) => { sha[p.nombre] = p.sha256; });
    const porModelo = {};
    meta.modelo.pesos.forEach((p, i) => { porModelo[p.nombre] = r.porModelo[i]; });
    return JSON.stringify({
      probabilidad_pcr: r.prob, clase: r.clase, umbral: r.umbral, regla: "pCR si probabilidad > umbral",
      por_modelo: porModelo, sha256: sha, huella_conjunto: huella, ficheros: nombres || {},
      dispositivo: "navegador, WebAssembly", aviso: "Uso educativo. Sin validez clínica.",
    }, null, 2);
  }

  /* datos: { PRE: Uint8Array, ... }; nombres: { PRE: "x_PRE.png", ... }; ejemplo: fila de modelo.json o null */
  async function predecirCorte(datos, nombres, origen, ejemplo) {
    limpiar();
    window.__resultado = undefined;
    try {
      const fases = [], avisos = [];
      for (const f of FASES) {
        const r = window.Png.leerFase(datos[f], f);        // lanza ErrorEntrada con el mensaje de la app
        fases.push(r.x);
        avisos.push(...r.avisos);
      }
      // que las tres fases sean el mismo corte de una RM (por nombre y por contenido); lanza ErrorEntrada si no
      avisos.push(...window.Png.validarCorte(fases, nombres, umbralesEntrada));
      const mapa = V.mapaCaptacion(fases), curva = V.curvaCaptacion(fases);
      pintarVisor(fases, nombres, mapa);
      for (const a of avisos) {
        const p = document.createElement("div");
        p.className = "aviso-corte";
        p.textContent = a;
        $("avisos").appendChild(p);
      }
      const r = await window.Modelo.predecir(fases);
      const tasa = d.fiarse.tasa_base;
      const destacado = ejemplo ? meta.modelo.pesos.findIndex((p) => d.modelo.pesos.find((q) => q.nombre === p.nombre).fold === ejemplo.fold) : null;

      $("bloque-resultado").hidden = false;
      $("de-donde").textContent = origen;
      $("probabilidad").textContent = r.texto;
      $("frase").textContent = r.pcr ? "Está por encima del umbral: el modelo predice pCR." : "Está por debajo del umbral: el modelo no predice pCR.";
      $("cerca").textContent = Math.abs(r.prob - r.umbral) < 0.02 ? "Está muy cerca del umbral: otro corte de la misma paciente podría dar la otra clase." : "";
      $("clase").textContent = r.clase;
      $("umbral").textContent = C.d4(r.umbral);
      $("tiempo").textContent = `${r.ms.toFixed(0)} ms`;
      dibujarEscala((w) => escalaSvg(w, r.prob, r.umbral, tasa, r.porModelo, destacado));
      dibujarCurva((w) => curvaSvg(w, curva));
      $("leyenda-escala").textContent = "Triángulo: este corte (media de los 5 modelos). Puntos: cada modelo" +
        (destacado !== null ? "; el aro es el modelo que no vio a esta paciente" : "") +
        ". Línea continua: el umbral, por encima del cual se predice pCR. Línea discontinua: la tasa base, la proporción de pacientes con pCR en train.";
      const sat = curva.saturadaEarly;
      $("nota-curva").innerHTML = `Intensidad media del ${C.pct0(V.FRACCION_ZONA)} de píxeles de la mama que más realza de PRE a EARLY (${curva.nPixeles} px). Tipo por el cambio de EARLY a LATE: más de +10 % persistente, ±10 % meseta, menos de −10 % lavado.` +
        (sat > 0.1 ? ` EARLY satura en el ${C.pct0(sat)} de esa zona: el cambio medido es una cota, y solo se afirma un tipo si la cota basta.` : "") +
        " <strong>Descriptivo: el modelo no lo usa</strong>, y el momento de LATE cambia según el hospital.";
      const real = $("real");
      if (ejemplo) {
        const nombre = meta.modelo.pesos[destacado].nombre;
        real.innerHTML = `<b>Resultado real: ${ejemplo.pcr ? "pCR" : "no pCR"}</b> (${COHORTES[ejemplo.cohorte] || ejemplo.cohorte}). La paciente es de train: solo el modelo de su fold (${ejemplo.fold}, <code>${G.esc(nombre)}</code>) no la vio, y le da ${C.d4(r.porModelo[destacado])}.`;
        real.hidden = false;
      } else {
        real.hidden = true;
      }
      const cuerpo = $("por-modelo").tBodies[0];
      cuerpo.textContent = "";
      meta.modelo.pesos.forEach((p, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${G.esc(p.nombre)}</td><td class="num">${C.d4(r.porModelo[i])}</td>`;
        cuerpo.appendChild(tr);
      });
      ultimo = { r, nombres };
      window.__resultado = { corte: ejemplo ? ejemplo.corte : null, texto: r.texto, clase: r.clase, ms: r.ms, prob: r.prob };
      return r;
    } catch (e) {
      mostrarError(e instanceof window.Png.ErrorEntrada ? e.message : "No se pudo predecir: " + e.message);
      return null;
    }
  }

  async function ficherosSubidos() {
    const datos = {}, nombres = {};
    for (const f of FASES) {
      const fichero = $("fichero-" + f).files[0];
      if (!fichero) throw new window.Png.ErrorEntrada(`${f}: falta el fichero.`);
      datos[f] = new Uint8Array(await fichero.arrayBuffer());
      nombres[f] = fichero.name;
    }
    return { datos, nombres };
  }

  // ---------------------------------------------------------------------- //
  // Autocomprobacion con el pipeline (ejemplos frente a la referencia de PyTorch)
  // ---------------------------------------------------------------------- //
  function fila(celdas, clases) {
    const tr = document.createElement("tr");
    celdas.forEach((c, i) => {
      const td = document.createElement("td");
      td.textContent = c;
      if (clases && clases[i]) td.className = clases[i];
      tr.appendChild(td);
    });
    return tr;
  }

  async function comprobar() {
    $("comprobar").disabled = true;
    window.__comprobacion = undefined;
    const cuerpo = $("tabla-comprobacion").tBodies[0];
    cuerpo.textContent = "";
    let correctos = 0, maxima = 0;
    for (const ej of meta.ejemplos) {
      const fases = [];
      for (const f of FASES) fases.push(window.Png.leerFase(await bytesDe(ej.ficheros[f]), f).x);
      const r = await window.Modelo.predecir(fases);
      const refs = [ej.referencia.prob, ...ej.referencia.por_modelo];
      const obt = [r.prob, ...r.porModelo];
      const dif = Math.max(...obt.map((v, i) => Math.abs(v - refs[i])));
      const bien = dif < 1e-5 && r.clase === ej.referencia.clase;
      if (bien) correctos++;
      maxima = Math.max(maxima, dif);
      cuerpo.appendChild(fila([ej.corte, comaN(r.prob, 6), comaN(ej.referencia.prob, 6), dif.toExponential(1).replace(".", ","),
                               bien ? "correcto" : "FALLO"], [null, "num", "num", "num", bien ? "bien" : "mal"]));
    }
    $("tabla-comprobacion").hidden = false;
    const total = meta.ejemplos.length;
    $("resumen-comprobacion").textContent = `${correctos} de ${total} correctos; diferencia máxima ${maxima.toExponential(1).replace(".", ",")} (tolerancia 1e-5, misma clase con el umbral).`;
    window.__comprobacion = { correctos, total, maxima };
    $("comprobar").disabled = false;
  }

  // ---------------------------------------------------------------------- //
  // Secciones de datos
  // ---------------------------------------------------------------------- //
  function barrasClinica() {
    const c = d.comparacion.clinica_e_imagen;
    const cont = $("barras-clinica");
    const filas = [["clinica", "Variables clínicas solas (subtipo, edad y volumen)"], ["imagen", "Imagen sola (esta red)"], ["combinada", "Las dos combinadas"]];
    for (const [clave, nombre] of filas) {
      const w = c[clave], ancho = Math.max(0, Math.min(1, (w.media - 0.5) / 0.3)) * 100;
      cont.insertAdjacentHTML("beforeend",
        `<div class="barra-clinica"><span>${nombre}</span><div class="pista"><i style="width:${ancho.toFixed(0)}%"></i></div>` +
        `<span class="valor">${C.d3(w.media)} ± ${C.d3(w.desviacion)}</span>` +
        (clave === "imagen" ? `<span class="nota pequeno">optimista: época elegida con la propia validación; sin elegir, ${C.d3(c.imagen_todas_las_epocas)}</span>` : "") +
        `</div>`);
    }
  }

  function detalles() {
    $("i-onnx").textContent = meta.modelo.sha256;
    $("opset").textContent = String(meta.modelo.opset);
    $("pct-realce").textContent = `${100 - V.PERCENTIL_REALCE} %`;
    const cuerpo = $("tabla-pesos").tBodies[0];
    for (const p of d.modelo.pesos) {
      const tr = fila([`models/${p.nombre}.pt`, String(p.fold), String(p.epoca), p.sha256], [null, "num", "num", "hash"]);
      cuerpo.appendChild(tr);
    }
    const u = d.seleccion.umbral, t = d.test, v = d.seleccion.validacion;
    const f = (x) => (x === undefined || x === null ? "—" : C.d3(x));
    const filas = [
      [`Validación fuera de fold, por paciente (${C.entero(u.pacientes.n)})`, v.global.auc, u.por_paciente.sensibilidad, u.por_paciente.especificidad, u.por_paciente.VPP, u.por_paciente.f1_macro],
      [`Validación fuera de fold, por corte (${C.entero(u.por_corte.n)})`, null, u.por_corte.sensibilidad, u.por_corte.especificidad, u.por_corte.VPP, null],
      [`Test, por paciente (${C.entero(t.pacientes.n)})`, t.auc.auc, t.por_paciente.sensibilidad, t.por_paciente.especificidad, t.por_paciente.VPP, t.por_paciente.f1_macro],
      [`Test, por corte (${C.entero(t.pacientes.cortes)})`, t.corte_suelto.auc.auc, t.corte_suelto.sensibilidad, t.corte_suelto.especificidad, t.corte_suelto.VPP, t.corte_suelto.f1_macro],
    ];
    const cm = $("tabla-metricas").tBodies[0];
    for (const [nombre, ...vals] of filas) cm.appendChild(fila([nombre, ...vals.map(f)], [null, "num", "num", "num", "num", "num"]));
    $("entrenamiento").innerHTML = C.entrenamientoHtml(d);
  }

  // ---------------------------------------------------------------------- //
  // Arranque
  // ---------------------------------------------------------------------- //
  /* Umbrales de la validacion del corte (src/utils/calibrar_entrada.py): sin ellos no se predice nada. */
  async function cargarUmbralesEntrada() {
    const r = await fetch("validacion_entrada.json");
    if (!r.ok) throw new Error("no se pudo cargar validacion_entrada.json");
    return r.json();
  }

  async function iniciar() {
    try {
      // si algo se atasca (red lenta, WebAssembly bloqueado), que la pagina lo diga en vez de quedarse en «Cargando…»
      const limite = new Promise((_, rechazar) => setTimeout(() => rechazar(new Error("tarda más de 45 s; recarga la página o prueba con otro navegador")), 45000));
      [meta, d, umbralesEntrada] = await Promise.race([Promise.all([window.Modelo.cargar(""), C.cargarDatos(), cargarUmbralesEntrada()]), limite]);
    } catch (e) {
      $("estado").textContent = "";
      mostrarError("No se pudo cargar el modelo: " + e.message);
      return;
    }
    C.rellenar(d);
    $("enlace-licencia").href = d.creditos.url_licencia;
    $("enlace-repo").href = d.creditos.repositorio;
    huella = (await sha256Texto(meta.modelo.pesos.map((p) => p.sha256).join(""))).slice(0, 12);
    $("huella").textContent = huella;
    barrasClinica();
    detalles();

    meta.ejemplos.forEach((ej, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.ejemplo = ej.corte;
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = `<img src="${ej.ficheros.EARLY}" alt="" width="256" height="256">` +
        `<span class="nombre">Ejemplo ${i + 1}</span><span class="sub">${COHORTES[ej.cohorte] || ej.cohorte} · pCR real: ${ej.pcr ? "sí" : "no"}</span>`;
      b.addEventListener("click", async () => {
        document.querySelectorAll(".ejemplos button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        const datos = {}, nombres = {};
        for (const f of FASES) {
          datos[f] = await bytesDe(ej.ficheros[f]);
          nombres[f] = ej.ficheros[f].split("/").pop();
        }
        await predecirCorte(datos, nombres, `Ejemplo ${i + 1}: ${ej.corte} (${COHORTES[ej.cohorte] || ej.cohorte}, paciente de train).`, ej);
      });
      $("ejemplos").appendChild(b);
    });

    for (const f of FASES) {
      $("fichero-" + f).addEventListener("change", (ev) => {
        $("nf-" + f).textContent = ev.target.files[0] ? ev.target.files[0].name : "ningún archivo";
      });
    }
    $("predecir").addEventListener("click", async () => {
      limpiar();
      document.querySelectorAll(".ejemplos button").forEach((x) => x.setAttribute("aria-pressed", "false"));
      try {
        const { datos, nombres } = await ficherosSubidos();
        await predecirCorte(datos, nombres, "Tus ficheros.", null);
      } catch (e) {
        mostrarError(e instanceof window.Png.ErrorEntrada ? e.message : "No se pudo leer: " + e.message);
      }
    });
    $("descargar").addEventListener("click", () => {
      if (!ultimo) return;
      const blob = new Blob([resultadoJson(ultimo.r, ultimo.nombres)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "resultado_pcr.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
    $("comprobar").addEventListener("click", () => comprobar().catch((e) => {
      $("resumen-comprobacion").textContent = "Error: " + e.message;
      $("comprobar").disabled = false;
    }));
    $("predecir").disabled = false;
    $("comprobar").disabled = false;
    $("estado").textContent = "Modelo cargado (ONNX Runtime Web, WebAssembly, 1 hilo).";
    window.__listo = true;
  }

  iniciar();
})();
