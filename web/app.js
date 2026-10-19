import { forecast } from "./lstm.js";

const REC_COLORS = {
  greening: "#3a8d4f", cool_roofs: "#2a7ab0", protect_green: "#1f5f33", water: "#44a6c6", bare_land: "#c9a27c", monitor: "#b9b0a6",
};
const REC_SHORT = {
  greening: "Urban greening", cool_roofs: "Cool roofs", protect_green: "Protect green cover", water: "Water body",
  bare_land: "Bare / fallow land", monitor: "Monitor",
};
const MODEL_NAMES = {
  "city_median (no model)": "City median (no model)", linear_regression: "Linear regression",
  gradient_boosting: "Gradient boosting", pixel_mlp: "Pixel MLP (no spatial context)",
  unet: "U-Net", attention_unet: "Attention U-Net",
};
const SEQ = ["#fff5d6", "#fdd17a", "#f7a14a", "#e4572e", "#a3242a"];
const DIV = ["#2c6fb0", "#8fbbd9", "#f4f1ea", "#f5a35c", "#c0392b"];
const COOL = ["#eef5fb", "#9cc6e4", "#3d8dc4", "#18527f"];

const $ = (s) => document.querySelector(s);
const fmt = (v, d = 1, plus = false) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : `${plus && v > 0 ? "+" : ""}${v.toFixed(d)}`;

function lerpColor(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const p = t * (stops.length - 1), i = Math.min(Math.floor(p), stops.length - 2), f = p - i;
  const a = stops[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = stops[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
const gradient = (stops) => `linear-gradient(90deg, ${stops.join(",")})`;

let GRID, METRICS, FCMODEL, map, cellLayer, hotLayer, overlays = {}, layer = "priority", opacity = 0.7;
const polys = new Map();

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function init() {
  [GRID, METRICS] = await Promise.all([getJSON("data/grid.json"), getJSON("data/metrics.json")]);
  FCMODEL = await getJSON("data/forecast_model.json").catch(() => null);
  const m = GRID.meta;
  if (m.demo || METRICS.demo) $("#demo-banner").hidden = false;
  $("#subtitle").textContent = `Deep learning heat-risk map · ${m.city} · summers ${m.years[0]}–${m.years[m.years.length - 1]}`;
  initMap();
  renderKPIs();
  renderSuhi();
  renderRecCounts();
  renderRanking();
  renderModel();
  setupTabs();
  setupControls();
  if (FCMODEL) renderSkill();
}

/* ---------------- map ---------------- */
function initMap() {
  const b = GRID.meta.bounds;
  map = L.map("map", { preferCanvas: true, zoomControl: true }).fitBounds(b);
  const streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', maxZoom: 19,
  }).addTo(map);
  const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics", maxZoom: 19,
  });
  L.control.layers({ "Street map": streets, "Satellite": satellite }, null, { position: "bottomright" }).addTo(map);
  for (const k of ["heat", "lst", "cooling"]) overlays[k] = L.imageOverlay(`data/${k}.png`, b, { opacity, interactive: false });

  const renderer = L.canvas({ padding: 0.3 });
  cellLayer = L.layerGroup().addTo(map);
  hotLayer = L.layerGroup().addTo(map);
  for (const c of GRID.cells) {
    const p = L.polygon(c.c, { renderer, weight: 0, fillOpacity: opacity });
    p.on("click", () => showDetail(c));
    p.on("mouseover", () => p.setStyle({ weight: 1.5, color: "#1d1a17" }));
    p.on("mouseout", () => p.setStyle({ weight: 0 }));
    polys.set(c.id, p);
    cellLayer.addLayer(p);
    if (c.hot) hotLayer.addLayer(L.polygon(c.c, { renderer, color: "#a3242a", weight: 1.4, fill: false, interactive: false }));
  }
  restyle();
}

function cellColor(c) {
  const r = GRID.meta.overlay_ranges;
  switch (layer) {
    case "priority": return lerpColor(SEQ, c.priority / 100);
    case "rec": return REC_COLORS[c.rec];
    case "trend": return c.trend === null ? "#cccccc" : lerpColor(DIV, (c.trend + 0.5) / 1.0);
    case "heat": return lerpColor(DIV, (c.heat - r.heat[0]) / (r.heat[1] - r.heat[0]));
    default: return "#000";
  }
}

function restyle() {
  const raster = ["heat", "lst", "cooling"].includes(layer);
  for (const [k, o] of Object.entries(overlays)) {
    if (k === layer) { o.setOpacity(opacity); o.addTo(map); } else map.removeLayer(o);
  }
  for (const c of GRID.cells) {
    const p = polys.get(c.id);
    if (raster) p.setStyle({ fillColor: "#000", fillOpacity: 0.01 });
    else p.setStyle({ fillColor: cellColor(c), fillOpacity: opacity });
  }
  renderLegend();
}

function renderLegend() {
  const r = GRID.meta.overlay_ranges, el = $("#legend");
  const scale = (stops, lo, hi, title) => `<div class="small muted" style="margin:8px 0 0">${title}</div>
    <div class="legend-scale" style="background:${gradient(stops)}"></div>
    <div class="legend-ends"><span>${lo}</span><span>${hi}</span></div>`;
  if (layer === "priority") el.innerHTML = scale(SEQ, "low", "high", "Priority score (0–100)");
  else if (layer === "heat") el.innerHTML = scale(DIV, `${fmt(r.heat[0])}°C`, `+${fmt(r.heat[1])}°C`, "vs. city median");
  else if (layer === "lst") el.innerHTML = scale(["#000004", "#57106e", "#bc3754", "#f98e09", "#fcffa4"], `${fmt(r.lst[0])}°C`, `${fmt(r.lst[1])}°C`, `Land surface temp, ${GRID.meta.reference_scene}`);
  else if (layer === "cooling") el.innerHTML = scale(COOL, "0°C", `−${fmt(r.cooling[1])}°C`, "Best-scenario surface cooling");
  else if (layer === "trend") el.innerHTML = scale(DIV, "−0.5", "+0.5", "°C per year (anomaly)");
  else el.innerHTML = Object.entries(REC_SHORT).map(([k, v]) => `<div class="legend-item"><span class="sw" style="background:${REC_COLORS[k]}"></span>${v}</div>`).join("");
}

/* ---------------- detail card ---------------- */
function sparkline(vals, years) {
  const pts = vals.map((v, i) => [i, v]).filter(([, v]) => v !== null);
  if (pts.length < 2) return "";
  const W = 320, H = 60, xs = (i) => 8 + (i / (vals.length - 1)) * (W - 16);
  const lo = Math.min(...pts.map((p) => p[1]), 0), hi = Math.max(...pts.map((p) => p[1]), 0.1);
  const ys = (v) => H - 10 - ((v - lo) / (hi - lo)) * (H - 20);
  const d = pts.map(([i, v], k) => `${k ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
    <line x1="0" x2="${W}" y1="${ys(0)}" y2="${ys(0)}" stroke="#d8d0c5" stroke-dasharray="3 3"/>
    <path d="${d}" fill="none" stroke="#e4572e" stroke-width="2"/>
    ${pts.map(([i, v]) => `<circle cx="${xs(i)}" cy="${ys(v)}" r="3" fill="#e4572e"><title>${years[i]}: ${fmt(v, 2, true)}°C</title></circle>`).join("")}
    <text x="4" y="${H - 1}" font-size="10" fill="#6f675f">${years[0]}</text>
    <text x="${W - 4}" y="${H - 1}" font-size="10" fill="#6f675f" text-anchor="end">${years[years.length - 1]}</text></svg>`;
}

function driverBars(drivers) {
  const max = Math.max(0.1, ...Object.values(drivers).map(Math.abs));
  return Object.entries(drivers).map(([g, v]) => {
    const w = (Math.abs(v) / max) * 50;
    const left = v >= 0 ? 50 : 50 - w;
    return `<div class="bar-row"><span>${g}</span>
      <div class="bar"><span style="left:${left}%;width:${w}%;background:${v >= 0 ? "#e4572e" : "#2a7ab0"}"></span></div>
      <span class="num">${fmt(v, 2, true)}°</span></div>`;
  }).join("");
}

function showDetail(c) {
  const m = GRID.meta, el = $("#detail");
  const rank = [...GRID.cells].sort((a, b) => b.priority - a.priority).findIndex((x) => x.id === c.id) + 1;
  const ctr = c.c.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]);
  const best = c.rec === "greening" ? c.cool_green : c.rec === "cool_roofs" ? c.cool_roof : null;
  const recClass = c.rec === "cool_roofs" ? "roof" : ["monitor", "water"].includes(c.rec) ? "none" : "";
  el.innerHTML = `<button class="x" aria-label="close">×</button>
    <h3>Area ${c.id} ${c.hot ? '<span class="pill" style="background:#a3242a">HOTSPOT</span>' : ""}</h3>
    <div class="muted">Priority rank <b>#${rank}</b> of ${GRID.cells.length} · ${ctr[0].toFixed(4)}, ${ctr[1].toFixed(4)}
      · <a href="https://www.google.com/maps/@${ctr[0]},${ctr[1]},16z" target="_blank" rel="noopener">open map</a></div>
    <div class="grid2">
      <div><b>${fmt(c.heat, 1, true)}°C</b><span>heat vs city median (${m.recent_years[0]}–${m.recent_years[m.recent_years.length - 1]})</span></div>
      <div><b>${fmt(c.lst)}°C</b><span>surface temp on ${m.reference_scene}</span></div>
      <div><b>${fmt(c.priority, 0)}</b><span>priority score / 100</span></div>
      <div><b>${c.trend === null ? "–" : fmt(c.trend, 2, true)}</b><span>°C / year trend</span></div>
    </div>
    <div class="rec ${recClass}"><b>${m.recs[c.rec]}</b>
      ${best !== null ? `<div class="small">Model-estimated surface cooling: <b>${fmt(best, 1)}°C</b></div>` : ""}
      <div class="small muted" style="margin:4px 0 0">Scenario estimates: greening ${c.cool_green < 0 ? fmt(c.cool_green, 2) + "°C" : "no cooling"} ·
        cool roofs ${c.cool_roof < 0 ? fmt(c.cool_roof, 2) + "°C" : "no cooling"}</div></div>
    <h3 style="margin-top:10px">Why is it hot?</h3>
    <p class="muted">Each factor's contribution to this area's predicted temperature (red heats, blue cools).</p>
    ${driverBars(c.drivers)}
    <h3 style="margin-top:10px">Heat anomaly by year</h3>
    ${sparkline(c.years, m.years)}
    <div class="small muted">Land cover: built-up ${Math.round(c.built * 100)}% · green ${Math.round(c.green * 100)}%${c.fallow !== undefined ? ` · bare/fallow ${Math.round(c.fallow * 100)}%` : ""} · water ${Math.round(c.water * 100)}% · NDVI ${fmt(c.ndvi, 2)}</div>`;
  el.hidden = false;
  el.querySelector(".x").onclick = () => (el.hidden = true);
}

/* ---------------- sidebar ---------------- */
function renderKPIs() {
  const m = GRID.meta, lst = METRICS.lst.results[METRICS.lst.best_model];
  const suhi = m.suhi_by_year.filter((s) => s.suhi_c !== null);
  const last = suhi[suhi.length - 1];
  const cu = METRICS.cooling.greening_urban_hotspots_c ?? METRICS.cooling.greening_hotspots_c;
  const cr = METRICS.cooling.roofs_urban_hotspots_c ?? METRICS.cooling.roofs_hotspots_c;
  const cool = Math.min(cu, cr);
  const urban = m.n_urban_hotspots ?? m.n_hotspots;
  $("#kpis").innerHTML = [
    ["hot", urban, `hotspots in built-up areas (of ${m.n_hotspots} ≥ +${fmt(m.hotspot_threshold_c)}°C)`],
    ["hot", last ? `${fmt(last.suhi_c)}°C` : "–", `built-up vs. green land, ${last ? last.year : ""}`],
    ["", `${fmt(lst.rmse, 2)}°C`, `model error (RMSE) · R² ${fmt(lst.r2, 2)}`],
    ["cool", `${fmt(cool, 1)}°C`, "avg. modelled cooling, urban hotspots"],
  ].map(([cls, v, l]) => `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function renderSuhi() {
  const s = GRID.meta.suhi_by_year;
  const ds = [{ label: "vs. green land", data: s.map((x) => x.suhi_c), borderColor: "#e4572e", backgroundColor: "#e4572e", tension: 0.3 }];
  if (s.some((x) => x.urban_minus_fallow_c !== undefined && x.urban_minus_fallow_c !== null))
    ds.push({ label: "vs. bare/fallow land", data: s.map((x) => x.urban_minus_fallow_c), borderColor: "#a07a4a", backgroundColor: "#a07a4a", tension: 0.3 });
  new Chart($("#suhi-chart"), {
    type: "line",
    data: { labels: s.map((x) => x.year), datasets: ds },
    options: { maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } }, scales: { y: { title: { display: true, text: "°C" } } } },
  });
  const hb = GRID.meta.heat_by_land_cover || {};
  const vals = Object.values(hb).map((v) => Math.abs(v.mean)), mx = Math.max(0.5, ...vals);
  $("#lc-heat").innerHTML = Object.entries(hb).sort((a, b) => b[1].mean - a[1].mean).map(([k, v]) => {
    const w = (Math.abs(v.mean) / mx) * 50, left = v.mean >= 0 ? 50 : 50 - w;
    return `<div class="bar-row"><span>${k}</span><div class="bar"><span style="left:${left}%;width:${w}%;background:${v.mean >= 0 ? "#e4572e" : "#2a7ab0"}"></span></div><span class="num">${fmt(v.mean, 1, true)}°</span></div>`;
  }).join("") + (GRID.meta.land_cover_source ? `<div class="small muted" style="margin-top:6px">Land cover: ${GRID.meta.land_cover_source}</div>` : "");
}

function renderRecCounts() {
  const rc = GRID.meta.recommendation_counts, tot = Object.values(rc).reduce((a, b) => a + b, 0) || 1;
  $("#rec-counts").innerHTML = Object.entries(rc).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
    `<div class="bar-row"><span>${REC_SHORT[k]}</span><div class="bar"><span style="left:0;width:${(v / tot) * 100}%;background:${REC_COLORS[k]}"></span></div><span class="num">${v}</span></div>`).join("");
}

function renderRanking() {
  const top = [...GRID.cells].sort((a, b) => b.priority - a.priority).slice(0, 20);
  $("#rank-list").innerHTML = top.map((c) => `<li data-id="${c.id}"><div><div class="t">Area ${c.id} · ${fmt(c.heat, 1, true)}°C</div>
    <div class="s">${REC_SHORT[c.rec]}${c.rec === "greening" ? ` (${fmt(c.cool_green, 1)}°C)` : c.rec === "cool_roofs" ? ` (${fmt(c.cool_roof, 1)}°C)` : ""}</div></div>
    <span class="pill" style="background:${lerpColor(SEQ, c.priority / 100)};color:#1d1a17">${fmt(c.priority, 0)}</span></li>`).join("");
  $("#rank-list").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => {
    const c = GRID.cells.find((x) => x.id === li.dataset.id);
    map.fitBounds(polys.get(c.id).getBounds(), { maxZoom: 15 });
    showDetail(c);
  }));
}

function renderModel() {
  const res = METRICS.lst.results, best = METRICS.lst.best_model;
  $("#model-table").innerHTML = `<tr><th>Model</th><th>RMSE °C</th><th>MAE</th><th>R²</th><th>Hotspot F1</th></tr>` +
    Object.entries(res).map(([k, r]) => `<tr class="${k === best ? "best" : ""}"><td>${MODEL_NAMES[k] || k}${k === best ? " ★" : ""}</td>
      <td>${fmt(r.rmse, 2)}</td><td>${fmt(r.mae, 2)}</td><td>${fmt(r.r2, 3)}</td><td>${fmt(r.hotspot_f1, 3)}</td></tr>`).join("");
  const imp = METRICS.importance, max = Math.max(...Object.values(imp), 0.01);
  $("#importance").innerHTML = Object.entries(imp).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
    `<div class="bar-row"><span>${k}</span><div class="bar"><span style="left:0;width:${Math.max(0, v / max) * 100}%;background:#3a8d4f"></span></div><span class="num">${fmt(v, 2)}</span></div>`).join("");
  const l = METRICS.lst;
  $("#data-notes").innerHTML = `<b>Data:</b> ${l.n_scenes} Landsat 8/9 summer scenes (${l.scenes[0]} → ${l.scenes[l.scenes.length - 1]}),
    Sentinel-2 L2A, Copernicus DEM (Microsoft Planetary Computer); ERA5 daily weather (Open-Meteo).
    Training patches: ${l.patches.train}, validation ${l.patches.val}, test ${l.patches.test}.
    ★ = selected on validation error. Cooling numbers are model-based <i>surface</i> temperature estimates, not engineering guarantees.
    Generated ${GRID.meta.generated}.`;
}

/* ---------------- forecast ---------------- */
function renderSkill() {
  const f = METRICS.forecast;
  if (!f) return;
  const colors = { persistence: "#a59d94", climatology: "#d8b46a", ridge_regression: "#2a7ab0", attention_lstm: "#e4572e" };
  new Chart($("#fc-skill"), {
    type: "line",
    data: { labels: f.metrics.attention_lstm.mae_by_lead.map((_, i) => `+${i + 1}d`),
      datasets: Object.entries(f.metrics).map(([k, v]) => ({ label: k.replace("_", " "), data: v.mae_by_lead, borderColor: colors[k], backgroundColor: colors[k], tension: 0.2 })) },
    options: { maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } }, scales: { y: { title: { display: true, text: "MAE °C" } } } },
  });
}

let fcLoaded = false;
async function runLiveForecast() {
  if (fcLoaded || !FCMODEL) return;
  fcLoaded = true;
  const vars = FCMODEL.features.filter((f) => !f.startsWith("doy"));
  const [lat, lon] = FCMODEL.point;
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=${vars.join(",")}&past_days=45&forecast_days=8&timezone=Asia%2FKolkata`;
  try {
    const j = await (await fetch(url)).json();
    const d = j.daily, today = new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });
    const rows = d.time.map((t, i) => Object.fromEntries([["date", t], ...vars.map((v) => [v, d[v][i]])]));
    const past = rows.filter((r) => r.date < today);
    for (const v of vars) { // fill occasional gaps
      for (let i = 0; i < past.length; i++) if (past[i][v] === null) past[i][v] = i ? past[i - 1][v] : 0;
    }
    const { pred } = forecast(FCMODEL, past);
    const fut = rows.filter((r) => r.date >= today).slice(0, FCMODEL.horizon);
    const thr = FCMODEL.heat_threshold;
    const hist = past.slice(-21);
    const labels = [...hist.map((r) => r.date.slice(5)), ...fut.map((r) => r.date.slice(5))];
    const pad = (arr, before) => [...Array(before).fill(null), ...arr];
    new Chart($("#fc-chart"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Recent observed max", data: [...hist.map((r) => r.temperature_2m_max), ...Array(fut.length).fill(null)], borderColor: "#1d1a17", pointRadius: 0, tension: 0.2 },
        { label: "Our LSTM forecast", data: pad(pred.slice(0, fut.length), hist.length), borderColor: "#e4572e", backgroundColor: "#e4572e", tension: 0.2 },
        { label: "Open-Meteo NWP forecast", data: pad(fut.map((r) => r.temperature_2m_max), hist.length), borderColor: "#2a7ab0", borderDash: [5, 4], pointRadius: 2, tension: 0.2 },
        { label: "Heat-stress threshold", data: labels.map(() => thr), borderColor: "#a3242a", borderWidth: 1, borderDash: [2, 3], pointRadius: 0 },
      ] },
      options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } }, scales: { y: { title: { display: true, text: "°C" } } } },
    });
    $("#fc-table").innerHTML = `<tr><th>Date</th><th>LSTM °C</th><th>NWP °C</th><th>Heat stress?</th></tr>` +
      fut.map((r, i) => { const hot = pred[i] >= thr || r.temperature_2m_max >= thr;
        return `<tr class="${hot ? "heat" : ""}"><td>${r.date}</td><td>${fmt(pred[i])}</td><td>${fmt(r.temperature_2m_max)}</td><td>${hot ? "⚠ yes" : "no"}</td></tr>`; }).join("");
    $("#fc-status").textContent = `Inputs: ${past[past.length - FCMODEL.window].date} → ${past[past.length - 1].date}. Heat-stress threshold ${thr}°C (${METRICS.forecast?.threshold_definition || "90th percentile of summer max"}). Test MAE at +1 day: ${fmt(FCMODEL.test_mae[0], 2)}°C.`;
  } catch (e) {
    $("#fc-chart").parentElement.hidden = true;
    $("#fc-status").textContent = `Could not load live weather (${e.message}). Showing test-period skill only.`;
  }
}

/* ---------------- controls ---------------- */
function setupTabs() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${t.dataset.tab}`));
    if (t.dataset.tab === "forecast") runLiveForecast();
    if (t.dataset.tab === "hotspots") { layer = "priority"; document.querySelector('input[value="priority"]').checked = true; restyle(); }
  }));
}

function setupControls() {
  document.querySelectorAll('input[name="layer"]').forEach((r) => r.addEventListener("change", () => { layer = r.value; restyle(); }));
  $("#opacity").addEventListener("input", (e) => { opacity = +e.target.value; restyle(); });
  $("#hot-outline").addEventListener("change", (e) => (e.target.checked ? hotLayer.addTo(map) : map.removeLayer(hotLayer)));
}

init().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin", `<div class="banner">Could not load data: ${e.message}. Run the pipeline to generate web/data.</div>`);
  console.error(e);
});
