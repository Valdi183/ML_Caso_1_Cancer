/* Pagina del caso (index.html): rellena las cifras desde datos.json, dibuja las figuras y mueve la
 * portada. Ninguna cifra esta escrita aqui: todas salen de datos.json (python -m src.web.exportar). */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const C = window.Comun, G = window.Graficos;
  const reducido = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FASES = ["PRE", "EARLY", "LATE"];
  const SUBTITULO = { PRE: "antes del contraste", EARLY: "poco después", LATE: "unos minutos más tarde" };

  function html(tag, clase, contenido) {
    const el = document.createElement(tag);
    if (clase) el.className = clase;
    if (contenido !== undefined) el.innerHTML = contenido;
    return el;
  }

  const nombreCohorte = (d, id) => d.datos.cohortes.find((c) => c.id === id).nombre;

  // ---------------------------------------------------------------------- //
  // Portada: una sola transicion de PRE a EARLY y un control para recorrer las fases
  // ---------------------------------------------------------------------- //
  function portada(d) {
    const corte = d.portada.corte, imgs = {};
    const base = `ejemplos/${corte}_`;
    for (const f of FASES) {
      const img = $("img-" + f.toLowerCase());
      img.alt = `Corte de resonancia, fase ${f} (${SUBTITULO[f]})`;
      img.src = `${base}${f}.png`;
      imgs[f] = img;
    }
    const cohorte = nombreCohorte(d, d.portada.cohorte);
    const PIES = {
      PRE: "PRE: antes de inyectar el contraste.",
      EARLY: "EARLY: poco después. El contraste llega y el tumor se ilumina.",
      LATE: "LATE: unos minutos más tarde.",
    };
    let actual = null, temporizador = null;
    const botones = document.querySelectorAll(".fases button");

    function mostrar(fase, ms) {
      if (fase === actual) return;
      const marco = $("marco-portada");
      marco.style.setProperty("--duracion", `${reducido ? 0 : ms}ms`);
      for (const f of FASES) imgs[f].classList.remove("previa");
      if (actual) { imgs[actual].classList.remove("activa"); imgs[actual].classList.add("previa"); }
      imgs[fase].classList.add("activa");
      const anterior = actual;
      actual = fase;
      botones.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.fase === fase)));
      $("pie-portada").innerHTML = `${PIES[fase]} <span class="suave">Corte real de una paciente de train (${cohorte}).</span>`;
      if (anterior) setTimeout(() => { if (actual === fase) imgs[anterior].classList.remove("previa"); }, reducido ? 0 : ms + 50);
    }

    botones.forEach((b) => b.addEventListener("click", () => {
      clearTimeout(temporizador);
      mostrar(b.dataset.fase, 300);
    }));

    Promise.all(FASES.map((f) => (imgs[f].decode ? imgs[f].decode().catch(() => {}) : Promise.resolve()))).then(() => {
      mostrar("PRE", 0);
      if (!reducido) temporizador = setTimeout(() => mostrar("EARLY", 2400), 900);
    });
  }

  // ---------------------------------------------------------------------- //
  // 2. Datos
  // ---------------------------------------------------------------------- //
  function barrasCohortes(d) {
    const cont = $("barras-cohortes");
    const maximo = Math.max(...d.datos.cohortes.map((c) => c.pacientes));
    cont.appendChild(html("p", "leyenda-barras",
      '<span><i style="background:var(--tinta)"></i>train</span>' +
      '<span><i style="background:var(--tinta-2);background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.45) 0 3px,transparent 3px 7px)"></i>test</span>'));
    for (const c of d.datos.cohortes) {
      const fila = html("div", "barra");
      fila.appendChild(html("b", "", c.nombre));
      fila.appendChild(html("div", "pista",
        `<i class="train" style="width:${(100 * c.train) / maximo}%"></i><i class="test" style="width:${(100 * c.test) / maximo}%"></i>`));
      fila.appendChild(html("span", "texto-barra",
        `${C.entero(c.pacientes)} pacientes (${C.entero(c.train)} en train, ${C.entero(c.test)} en test)`));
      cont.appendChild(fila);
    }
  }

  function tresFases(d) {
    const base = `ejemplos/${d.portada.corte}_`;
    const cont = $("tres-fases");
    for (const f of FASES) {
      const fig = html("figure", "", `<img src="${base}${f}.png" alt="Corte de resonancia, fase ${f}" width="256" height="256">` +
        `<div class="rotulo"><b>${f}</b><span>${SUBTITULO[f]}</span></div>`);
      cont.appendChild(fig);
    }
  }

  function particion(d) {
    const t = d.datos.train, p = d.datos.test;
    const cont = $("particion");
    cont.innerHTML =
      `<div class="fila" style="grid-template-columns:${t.pacientes}fr ${p.pacientes}fr" aria-label="Reparto de las pacientes">` +
      `<div class="bloque train"><b>Train</b><span>${C.entero(t.pacientes)} pacientes</span></div>` +
      `<div class="bloque test"><b>Test</b><span>${C.entero(p.pacientes)}</span></div></div>` +
      `<div class="fila" style="grid-template-columns:${t.pacientes}fr ${p.pacientes}fr"><div><div class="folds">` +
      Array.from({ length: d.datos.folds }, (_, i) => `<div class="fold">fold ${i}</div>`).join("") +
      `</div><p class="nota-p">Cada paciente está en un solo fold. Se valida con uno y se entrena con los otros cuatro.</p></div>` +
      `<div><p class="nota-p"><b>Una sola vez</b>, al final.</p></div></div>`;
  }

  // ---------------------------------------------------------------------- //
  // 3. Modelo
  // ---------------------------------------------------------------------- //
  function iconoEtapa(etapa) {
    const numeros = (etapa.forma.match(/\d+/g) || []).map(Number);
    let cuerpo = "";
    if (numeros.length === 3) {
      const [canales, alto] = numeros;
      const lado = Math.round(14 + 36 * (alto / 256)), capas = Math.min(canales, 5);
      const x0 = 28 - lado / 2 - ((capas - 1) * 3) / 2, y0 = 28 - lado / 2 + ((capas - 1) * 3) / 2;
      for (let i = capas - 1; i >= 0; i--) {
        cuerpo += `<rect x="${x0 + i * 3}" y="${y0 - i * 3}" width="${lado}" height="${lado}" fill="var(--papel)" stroke="var(--tinta)" stroke-width="2"/>`;
      }
    } else if (numeros.length === 1) {
      for (let i = 0; i < 6; i++) cuerpo += `<rect x="${10 + i * 7}" y="${22 + (i % 2) * 3}" width="5" height="${12 - (i % 2) * 6}" fill="var(--tinta)"/>`;
    } else {
      cuerpo = `<circle cx="28" cy="28" r="14" fill="var(--tinta)"/><text x="28" y="34" text-anchor="middle" fill="var(--papel)" font-weight="700" font-size="18">p</text>`;
    }
    return `<svg viewBox="0 0 56 56" width="56" height="56" aria-hidden="true">${cuerpo}</svg>`;
  }

  function arquitectura(d) {
    const cont = $("arquitectura");
    const campo = Object.fromEntries(d.modelo.campo_receptivo.por_etapa.map((e) => [e.etapa, e.px]));
    for (const e of d.modelo.etapas) {
      const ve = campo[e.nombre.toLowerCase()];
      const detalle = e.detalle.replace(/\n/g, " ") + (ve ? ` · ve ${ve} × ${ve} px del corte` : "");
      cont.appendChild(html("li", "",
        `<span class="icono">${iconoEtapa(e)}</span>` +
        `<span><span class="nombre">${G.esc(e.nombre)}</span><span class="detalle">${G.esc(detalle)}</span></span>` +
        `<span class="medida"><b>${G.esc(e.forma)}</b><span>${e.parametros ? C.entero(e.parametros) + " parámetros" : "sin parámetros"}</span></span>`));
    }
  }

  async function campoReceptivo(d) {
    const cont = $("campo-receptivo");
    const px = d.modelo.campo_receptivo.px;
    const respuesta = await fetch(`ejemplos/${d.portada.corte}_EARLY.png`);
    const x = window.Png.leerFase(new Uint8Array(await respuesta.arrayBuffer()), "EARLY").x;
    // centro: el de la caja que encierra el tejido (mascara de la mama), no un punto elegido a ojo
    const mama = window.Visualizacion.mascaraMama([x, x, x]);
    let minX = 256, maxX = 0, minY = 256, maxY = 0;
    for (let i = 0; i < 65536; i++) {
      if (!mama[i]) continue;
      const px_ = i % 256, py_ = Math.floor(i / 256);
      if (px_ < minX) minX = px_;
      if (px_ > maxX) maxX = px_;
      if (py_ < minY) minY = py_;
      if (py_ > maxY) maxY = py_;
    }
    const cx = Math.min(Math.max((minX + maxX) / 2, px / 2), 256 - px / 2);
    const cy = Math.min(Math.max((minY + maxY) / 2, px / 2), 256 - px / 2);
    const lienzo = document.createElement("canvas");
    lienzo.width = lienzo.height = 256;
    lienzo.setAttribute("role", "img");
    lienzo.setAttribute("aria-label", `Corte de 256 por 256 píxeles con un cuadrado de ${px} por ${px} que marca lo que ve cada valor del mapa final`);
    const ctx = lienzo.getContext("2d");
    const imagen = ctx.createImageData(256, 256);
    for (let i = 0; i < 65536; i++) { imagen.data[i * 4] = imagen.data[i * 4 + 1] = imagen.data[i * 4 + 2] = x[i]; imagen.data[i * 4 + 3] = 255; }
    ctx.putImageData(imagen, 0, 0);
    ctx.fillStyle = "rgba(0,0,0,.5)";
    const x0 = Math.round(cx - px / 2), y0 = Math.round(cy - px / 2);
    ctx.fillRect(0, 0, 256, y0); ctx.fillRect(0, y0 + px, 256, 256 - y0 - px);
    ctx.fillRect(0, y0, x0, px); ctx.fillRect(x0 + px, y0, 256 - x0 - px, px);
    ctx.strokeStyle = "#E3E5E8"; ctx.lineWidth = 2; ctx.strokeRect(x0 + 1, y0 + 1, px - 2, px - 2);
    cont.appendChild(lienzo);
    const p = html("p", "", `Cada valor del mapa final ve solo el cuadrado marcado: <span data-v="modelo.campo_receptivo.px"></span> × <span data-v="modelo.campo_receptivo.px"></span> píxeles, el <span data-v="modelo.campo_receptivo.fraccion_area" data-f="pct1"></span> del corte. La red no ve el resto a la vez.`);
    cont.appendChild(p);
    C.rellenar(d, p);
  }

  function conjunto(d) {
    const cont = $("conjunto");
    const modelos = d.modelo.pesos.map((p, i) =>
      `<div class="modelo"><b>Modelo ${i + 1}</b><span>sin ver el fold ${p.fold}</span></div>`).join("");
    cont.innerHTML = `<div class="modelos">${modelos}</div>` +
      `<div class="paso">media de las ${d.modelo.n_modelos} probabilidades</div>` +
      `<div class="paso">¿supera el umbral ${C.d4(d.umbral)}?</div>` +
      `<div class="paso">pCR / no pCR</div>`;
  }

  function entrenamiento(d) {
    $("entrenamiento").innerHTML = C.entrenamientoHtml(d);
  }

  // ---------------------------------------------------------------------- //
  // 4. Umbral: otros criterios
  // ---------------------------------------------------------------------- //
  function tablaUmbrales(d) {
    const cuerpo = $("tabla-umbrales").tBodies[0];
    // solo los criterios: las filas «oficial», «siempre no pCR» y «azar» repiten o no son criterios
    const criterios = d.seleccion.umbral.alternativas.filter((f) => /^(especificidad|Youden)/.test(f.criterio));
    for (const f of criterios) {
      const oficial = Math.abs(f.umbral - d.seleccion.umbral.valor) < 1e-9;
      const criterio = f.criterio.replace(">=", "al menos").replace(/(\d)\.(\d)/g, "$1,$2");
      const tr = html("tr", "", `<td>${G.esc(criterio)}${oficial ? " (el elegido)" : ""}</td><td class="num">${C.d4(f.umbral)}</td>` +
        `<td class="num">${C.d3(f.sensibilidad)}</td><td class="num">${C.d3(f.especificidad)}</td><td class="num">${C.d3(f.VPP)}</td>`);
      if (oficial) tr.style.fontWeight = "700";
      cuerpo.appendChild(tr);
    }
  }

  // ---------------------------------------------------------------------- //
  // Indice lateral
  // ---------------------------------------------------------------------- //
  function indice() {
    const enlaces = [...document.querySelectorAll(".indice a[href^='#']")];
    const secciones = enlaces.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    if (!("IntersectionObserver" in window)) return;
    const marcar = (id) => enlaces.forEach((a) => {
      if (a.getAttribute("href") === "#" + id) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
    const visibles = new Set();
    const obs = new IntersectionObserver((entradas) => {
      for (const e of entradas) (e.isIntersecting ? visibles.add(e.target.id) : visibles.delete(e.target.id));
      const primera = secciones.find((s) => visibles.has(s.id));
      if (primera) marcar(primera.id);
    }, { rootMargin: "-15% 0px -65% 0px" });
    secciones.forEach((s) => obs.observe(s));
  }

  // ---------------------------------------------------------------------- //
  // Movimiento: solo lo que cambia o lo que controla el visitante
  // ---------------------------------------------------------------------- //
  const estado = { epoca: undefined, umbral: undefined };

  async function cargarOpcional(ruta) {
    try {
      const r = await fetch(ruta);
      return r.ok ? r.json() : null;
    } catch (e) {
      return null;
    }
  }

  /* Las barras de las cohortes crecen desde 0 al entrar en pantalla. */
  function animarBarras() {
    C.alEntrar($("barras-cohortes"), (cont) => {
      if (C.reducido()) return;
      const barras = [...cont.querySelectorAll(".pista i")];
      const finales = barras.map((b) => b.style.width);
      barras.forEach((b) => { b.style.transition = "none"; b.style.width = "0"; });
      cont.getBoundingClientRect();
      requestAnimationFrame(() => barras.forEach((b, i) => { b.style.transition = "width 800ms ease-out"; b.style.width = finales[i]; }));
    });
  }

  /* La matriz de confusion de test: las celdas aparecen contando hasta su recuento. */
  function animarMatriz(d) {
    C.alEntrar(document.querySelector('table.matriz[aria-label="Matriz de confusión en test"]'), (tabla) => {
      if (C.reducido()) return;
      tabla.querySelectorAll("td").forEach((td, i) => {
        const b = td.querySelector("b");
        td.style.animation = `aparecer 400ms ease-out ${i * 120}ms both`;
        setTimeout(() => C.contar(b, C.camino(d, b.dataset.v), C.entero, 700), i * 120);
      });
    });
  }

  /* Control de epoca: las curvas de train y validacion avanzan juntas y se ve su diferencia. */
  function controlEpoca(d, grafico) {
    if (!d.historial || !grafico) return;
    const folds = d.historial.folds, max = Math.max(...folds.map((f) => f.epoca[f.epoca.length - 1]));
    const deslizador = $("epoca"), boton = $("reproducir-epoca");
    let temporizador = null;
    const lectura = (e) => folds.map((f) => {
      const k = Math.min(f.epoca.filter((x) => x <= e).length, f.epoca.length) - 1;
      const dif = f.auc_train[k] - f.auc_val[k];
      const paro = e > f.epoca[f.epoca.length - 1] ? ` (paró en la época ${f.epoca[f.epoca.length - 1]})` : "";
      return `fold ${f.fold}: train ${C.d3(f.auc_train[k])}, validación ${C.d3(f.auc_val[k])}, diferencia ${dif >= 0 ? "+" : ""}${C.d3(dif)}${paro}`;
    }).join(" · ");
    const poner = (e) => {
      estado.epoca = e;
      deslizador.value = e;
      $("epoca-valor").textContent = `${e} de ${max}`;
      $("epoca-lectura").textContent = lectura(e);
      grafico.repintar();
    };
    const parar = () => {
      clearInterval(temporizador);
      temporizador = null;
      boton.textContent = "Reproducir";
      boton.setAttribute("aria-pressed", "false");
    };
    deslizador.max = max;
    deslizador.value = max;
    $("epoca-valor").textContent = `${max} de ${max}`;
    deslizador.addEventListener("input", () => { parar(); poner(Number(deslizador.value)); });
    boton.addEventListener("click", () => {
      if (temporizador) { parar(); return; }
      if (C.reducido()) { poner(max); return; }
      let e = estado.epoca === undefined || estado.epoca >= max ? 1 : estado.epoca;
      poner(e);
      boton.textContent = "Pausa";
      boton.setAttribute("aria-pressed", "true");
      temporizador = setInterval(() => { e += 1; poner(e); if (e >= max) parar(); }, 260);
    });
    $("control-epoca").hidden = false;
  }

  /* Explorador de umbral sobre las predicciones fuera de fold de VALIDACION (web/umbral_validacion.json). */
  function exploradorUmbral(pred, grafico) {
    if (!pred || !grafico) return;
    const pacientes = pred.pacientes, oficial = pred.umbral_oficial;
    const deslizador = $("umbral-deslizador"), volver = $("umbral-oficial");
    $("explorador-n").textContent = C.entero(pacientes.length);
    volver.textContent = `Volver al oficial (${C.d4(oficial)})`;
    const poner = (t, desdeDeslizador) => {
      estado.umbral = t;
      if (!desdeDeslizador) deslizador.value = t;
      let vp = 0, fp = 0, fn = 0, vn = 0;
      for (const [p, y] of pacientes) {
        if (p > t) { if (y) vp++; else fp++; } else if (y) fn++; else vn++;
      }
      for (const [id, v] of [["VP", vp], ["FP", fp], ["FN", fn], ["VN", vn]]) $("ex-" + id).textContent = C.entero(v);
      $("ex-sens").textContent = C.d3(vp / (vp + fn));
      $("ex-esp").textContent = C.d3(vn / (vn + fp));
      $("ex-vpp").textContent = vp + fp ? C.d3(vp / (vp + fp)) : "—";
      $("umbral-valor").textContent = C.d4(t) + (t === oficial ? " (oficial)" : "");
      volver.disabled = t === oficial;
      grafico.repintar();
    };
    deslizador.addEventListener("input", () => poner(Number(deslizador.value), true));
    volver.addEventListener("click", () => { poner(oficial); deslizador.focus(); });
    $("explorador-umbral").hidden = false;
    poner(oficial);
  }

  /* Recorrer la red: un corte pasa por las etapas y cambia de tamaño (alto x ancho) y de canales. */
  function recorrerRed(d) {
    const etapas = d.modelo.etapas, lis = [...$("arquitectura").children], visor = $("recorrido-tensor"), boton = $("recorrer");
    const CAPAS = 8, MAX = 200;
    visor.innerHTML = `<div class="tensor" aria-hidden="true">${Array.from({ length: CAPAS }, (_, i) =>
      `<i style="--i:${i}"></i>`).join("")}</div><p class="tensor-texto"></p>`;
    const capas = [...visor.querySelectorAll(".tensor i")], textoEl = visor.querySelector(".tensor-texto");
    visor.hidden = true;                                       // hasta que el visitante pulse «Recorrer»
    const portada = `url("ejemplos/${d.portada.corte}_EARLY.png")`;
    let temporizador = null;
    const mostrar = (i) => {
      const e = etapas[i], n = (e.forma.match(/\d+/g) || []).map(Number);
      lis.forEach((li, j) => li.classList.toggle("activo", j === i));
      // lado proporcional a la raiz del alto (256 px -> 200 px, 16 px -> 50 px), para que se vean todos
      let lado = 0, cuantas = 0, imagen = false;
      if (n.length === 3) { lado = Math.max(18, Math.round(MAX * Math.sqrt(n[1] / 256))); cuantas = Math.min(n[0], CAPAS); imagen = n[0] === 3; }
      else if (n.length === 1) { lado = 14; cuantas = CAPAS; }
      else { lado = 44; cuantas = 1; }                         // la salida: una sola probabilidad
      capas.forEach((c, k) => {
        const visible = k < cuantas;
        c.style.width = c.style.height = visible ? `${lado}px` : "0px";
        c.style.opacity = visible ? "1" : "0";
        c.style.borderRadius = n.length === 0 && k === 0 ? "50%" : "0";
        c.style.backgroundImage = imagen && k === 0 ? portada : "none";
      });
      textoEl.textContent = `${e.nombre}: ${e.forma}` + (n.length === 3 ? " (canales × alto × ancho)" : "");
    };
    boton.addEventListener("click", () => {
      clearInterval(temporizador);
      visor.hidden = false;
      if (C.reducido()) { mostrar(etapas.length - 1); return; }
      let i = 0;
      mostrar(i);
      boton.disabled = true;
      temporizador = setInterval(() => {
        i += 1;
        if (i >= etapas.length) { clearInterval(temporizador); boton.disabled = false; boton.textContent = "Recorrer otra vez"; boton.focus(); return; }
        mostrar(i);
      }, 1100);
    });
  }

  const DIBUJOS = {
    configuraciones: G.configuraciones, curvas: G.curvas, perdida: G.perdida, umbral: G.umbral, roc: G.roc,
    "cohortes-test": G.cohortesTest, calibracion: G.calibracion, comparacion: G.comparacion, saturacion: G.saturacion,
    folds: G.folds,
  };

  async function iniciar() {
    let d;
    try {
      d = await C.cargarDatos();
    } catch (e) {
      document.querySelector("main").insertAdjacentHTML("afterbegin",
        `<p class="aviso-caja" role="alert">No se pudieron cargar los datos de la página (${G.esc(e.message)}). Abre la web desde un servidor, no desde el fichero.</p>`);
      return;
    }
    C.rellenar(d);
    $("colecciones").textContent = d.creditos.colecciones.join(", ").replace(/, ([^,]*)$/, " y $1");
    $("enlace-licencia").href = d.creditos.url_licencia;
    $("enlace-repo").href = d.creditos.repositorio;
    portada(d);
    barrasCohortes(d);
    tresFases(d);
    particion(d);
    arquitectura(d);
    conjunto(d);
    entrenamiento(d);
    tablaUmbrales(d);
    const [historial, prediccionesVal] = await Promise.all([cargarOpcional("historial.json"), cargarOpcional("umbral_validacion.json")]);
    d.historial = historial;
    if (prediccionesVal) estado.umbral = prediccionesVal.umbral_oficial;
    const montados = {};
    document.querySelectorAll("[data-grafico]").forEach((el) => {
      const tipo = el.dataset.grafico;
      const dibujo = tipo === "curvas" ? (dd, w) => G.curvas(dd, w, estado.epoca)
        : tipo === "umbral" ? (dd, w) => G.umbral(dd, w, estado.umbral) : DIBUJOS[tipo];
      montados[tipo] = G.montar(el, dibujo, d);
    });
    controlEpoca(d, montados.curvas);
    exploradorUmbral(prediccionesVal, montados.umbral);
    recorrerRed(d);
    animarBarras();
    animarMatriz(d);
    indice();
    campoReceptivo(d).catch((e) => console.error(e));
    window.__listo = true;
  }

  iniciar();
})();
