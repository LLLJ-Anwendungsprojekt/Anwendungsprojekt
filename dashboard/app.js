/* ===================================================================
   GPR & Aktienmärkte — Dashboard Rendering (Plotly.js)
   Liest window.DASHBOARD_DATA (aus data.js) und rendert alle Abschnitte.
   =================================================================== */

const D = window.DASHBOARD_DATA;

const COLORS = {
  bg: "#FFFFFF", surface: "#F7F7F7", border: "#E5E5E5",
  text: "#1A1A1A", muted: "#888888",
  accent: "#1C4E80", accent2: "#A31621",
  positive: "#2E8B57", negative: "#A31621",
  region: { "China": "#D62728", "Europa": "#1F77B4", "Nordamerika": "#2CA02C", "Asien-Pazifik": "#FF7F0E" },
  cluster: ["#1C4E80", "#A31621", "#2E8B57", "#FF7F0E", "#7B2D8B"],
};

const BASE_LAYOUT = {
  template: "plotly_white",
  font: { family: "-apple-system, Segoe UI, sans-serif", size: 12, color: COLORS.text },
  plot_bgcolor: COLORS.bg,
  paper_bgcolor: COLORS.bg,
  margin: { l: 55, r: 30, t: 30, b: 50 },
  legend: { bgcolor: "rgba(0,0,0,0)", borderwidth: 0 },
  xaxis: { showgrid: true, gridcolor: "#F0F0F0", zeroline: false },
  yaxis: { showgrid: true, gridcolor: "#F0F0F0", zeroline: false },
  hoverlabel: { font: { size: 12 } },
};
const CONFIG = { displayModeBar: false, responsive: true };

function L(extra) { return Object.assign(JSON.parse(JSON.stringify(BASE_LAYOUT)), extra || {}); }
function md(m) { return m + "-01"; }                       // "YYYY-MM" -> Datum
function fmtPct(v, d = 2) { return (v == null ? "–" : v.toFixed(d) + " %"); }
function fmtNum(v, d = 2) { return (v == null ? "–" : v.toFixed(d)); }

/* Vertikale Event-Linien als shapes + annotations */
function eventShapes(yref = "paper") {
  return D.meta.events.map(e => ({
    type: "line", x0: md(e.month), x1: md(e.month), xref: "x",
    y0: 0, y1: 1, yref: "paper",
    line: { color: COLORS.muted, width: 1, dash: "dot" },
  }));
}
function eventAnnos() {
  return D.meta.events.map(e => ({
    x: md(e.month), y: 1, xref: "x", yref: "paper",
    text: e.label, showarrow: false, font: { size: 10, color: COLORS.muted },
    yanchor: "bottom", textangle: 0,
  }));
}

/* =================== 1. ÜBERSICHT =================== */
function renderOverview() {
  const o = D.overview, k = o.kpis;
  // KPI-Karten
  document.getElementById("kpi-row").innerHTML = [
    `<div class="kpi"><div class="value accent">${k.n_months}</div><div class="label">Monate · 2000–2021</div></div>`,
    `<div class="kpi"><div class="value accent">${k.n_indices}</div><div class="label">Indizes · ${k.n_regions} Regionen</div></div>`,
    `<div class="kpi"><div class="value neg">${fmtPct(k.avg_return_spike)}</div><div class="label">Ø Rendite bei GPR-Spikes (vs. ${fmtPct(k.avg_return_normal)})</div></div>`,
    `<div class="kpi"><div class="value neg">${fmtNum(k.corr_threat, 3)}</div><div class="label">Korr. THREAT → Markt (ACT: ${fmtNum(k.corr_act, 3)})</div></div>`,
  ].join("");

  drawOverviewTS("gprd_mean");
  // Scatter
  const sc = o.scatter;
  const traces = D.meta.region_order.map(r => ({
    x: sc[r].x, y: sc[r].y, mode: "markers", type: "scattergl", name: r,
    marker: { color: COLORS.region[r], size: 5, opacity: 0.55 },
    text: sc[r].name.map((n, i) => `${n} · ${sc[r].month[i]}`),
    hovertemplate: "%{text}<br>GPR Δ %{x:.1f}%<br>Rendite %{y:.2f}%<extra></extra>",
  }));
  traces.push({
    x: o.ols.x, y: o.ols.y, mode: "lines", type: "scatter", name: "OLS",
    line: { color: COLORS.muted, width: 2, dash: "dash" }, hoverinfo: "skip",
  });
  Plotly.react("ov-scatter", traces, L({
    xaxis: { title: "GPR-Änderung (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    yaxis: { title: "Aktienrendite (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.18 },
    annotations: [{ x: 0.02, y: 0.98, xref: "paper", yref: "paper", showarrow: false,
      text: `R² = ${o.ols.r2.toFixed(3)}`, font: { size: 11, color: COLORS.muted }, align: "left" }],
  }), CONFIG);

  // Box: spike vs normal
  Plotly.react("ov-box", [
    { y: o.spike_box.normal, type: "violin", name: "Normal", line: { color: "#7FA8D0" },
      fillcolor: "rgba(127,168,208,0.3)", box: { visible: true }, meanline: { visible: true }, points: false },
    { y: o.spike_box.spike, type: "violin", name: "GPR-Spike", line: { color: COLORS.accent2 },
      fillcolor: "rgba(163,22,33,0.3)", box: { visible: true }, meanline: { visible: true }, points: false },
  ], L({
    yaxis: { title: "Aktienrendite (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    annotations: [{ x: 0.5, y: 1.0, xref: "paper", yref: "paper", showarrow: false,
      text: `Ø ${fmtPct(k.avg_return_spike)} vs. ${fmtPct(k.avg_return_normal)}`,
      font: { size: 12, color: COLORS.text } }],
    showlegend: false,
  }), CONFIG);
}

function drawOverviewTS(gprKey) {
  const o = D.overview;
  const x = o.months.map(md);
  Plotly.react("ov-timeseries", [
    { x, y: o.avg_stock_return, type: "bar", name: "Ø Aktienrendite", yaxis: "y2",
      marker: { color: o.avg_stock_return.map(v => v >= 0 ? "rgba(46,139,87,0.55)" : "rgba(163,22,33,0.55)") },
      hovertemplate: "%{x|%b %Y}<br>Rendite %{y:.2f}%<extra></extra>" },
    { x, y: o[gprKey], type: "scatter", mode: "lines", name: "GPR", line: { color: COLORS.accent, width: 2 },
      hovertemplate: "%{x|%b %Y}<br>GPR %{y:.1f}<extra></extra>" },
  ], L({
    shapes: eventShapes(), annotations: eventAnnos(),
    xaxis: { type: "date", gridcolor: "#F0F0F0", zeroline: false },
    yaxis: { title: "GPR-Index", gridcolor: "#F0F0F0", zeroline: false },
    yaxis2: { title: "Ø Rendite (%)", overlaying: "y", side: "right", showgrid: false, zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.15 },
    margin: { l: 55, r: 60, t: 30, b: 50 },
  }), CONFIG);
}

/* =================== 2. ZEITREIHEN =================== */
let tsIndexInit = false;
function renderTimeseries() {
  const o = D.overview, x = o.months.map(md);
  // GPR-Komponenten
  Plotly.react("ts-components", [
    { x, y: o.gprd_mean, name: "GPR gesamt", type: "scatter", mode: "lines",
      line: { color: COLORS.muted, width: 1.8 }, fill: "tozeroy", fillcolor: "rgba(136,136,136,0.08)" },
    { x, y: o.gprd_act_mean, name: "GPR-ACT", type: "scatter", mode: "lines",
      line: { color: COLORS.accent, width: 1.6 } },
    { x, y: o.gprd_threat_mean, name: "GPR-THREAT", type: "scatter", mode: "lines",
      line: { color: "#6FB1E0", width: 1.6 } },
  ], L({
    shapes: eventShapes(), annotations: eventAnnos(),
    xaxis: { type: "date", gridcolor: "#F0F0F0" }, yaxis: { title: "GPR-Index", gridcolor: "#F0F0F0" },
    legend: { orientation: "h", y: -0.15 },
  }), CONFIG);

  if (!tsIndexInit) {
    const box = document.getElementById("ts-index-checks");
    const defaults = ["IXIC", "GDAXI", "HSI", "000001.SS"];
    box.innerHTML = "<label>Indizes:</label>" + Object.keys(D.meta.indices).map(c =>
      `<label class="chk"><input type="checkbox" value="${c}" ${defaults.includes(c) ? "checked" : ""}> ${D.meta.indices[c].name}</label>`
    ).join("");
    box.addEventListener("change", drawIndexCompare);
    tsIndexInit = true;
  }
  drawIndexCompare();

  // Rollierende Korrelation
  const rc = D.timeseries.rolling_corr;
  const rtraces = D.meta.region_order.map(r => ({
    x, y: rc[r], name: r, type: "scatter", mode: "lines",
    line: { color: COLORS.region[r], width: 1.8 },
  }));
  Plotly.react("ts-rolling", rtraces, L({
    shapes: [{ type: "line", x0: x[0], x1: x[x.length - 1], y0: 0, y1: 0, xref: "x", yref: "y",
      line: { color: COLORS.muted, width: 1, dash: "dash" } }],
    xaxis: { type: "date", gridcolor: "#F0F0F0" },
    yaxis: { title: "Korrelation (12M rollierend)", gridcolor: "#F0F0F0", range: [-1, 1] },
    legend: { orientation: "h", y: -0.15 },
  }), CONFIG);
}

function drawIndexCompare() {
  const checks = [...document.querySelectorAll("#ts-index-checks input:checked")].map(i => i.value).slice(0, 6);
  const x = D.overview.months.map(md);
  const traces = [];
  checks.forEach((c, idx) => {
    const series = D.timeseries.close[c];
    if (!series) return;
    let base = null;
    const reb = series.map(v => {
      if (v == null) return null;
      if (base == null) base = v;
      return base ? (v / base) * 100 : null;
    });
    traces.push({ x, y: reb, name: D.meta.indices[c].name, type: "scatter", mode: "lines",
      line: { width: 1.8 } });
  });
  // GPR auf zweiter Achse
  traces.push({ x, y: D.overview.gprd_mean, name: "GPR", type: "scatter", mode: "lines", yaxis: "y2",
    line: { color: COLORS.muted, width: 1, dash: "dot" }, opacity: 0.7 });
  Plotly.react("ts-index", traces, L({
    xaxis: { type: "date", gridcolor: "#F0F0F0" },
    yaxis: { title: "Kurs (rebasiert = 100)", gridcolor: "#F0F0F0" },
    yaxis2: { title: "GPR", overlaying: "y", side: "right", showgrid: false },
    legend: { orientation: "h", y: -0.15 },
    margin: { l: 55, r: 55, t: 30, b: 50 },
  }), CONFIG);
}

/* =================== 3. REGIONEN =================== */
function renderRegions() {
  const r = D.regions, x = D.overview.months.map(md);
  // Heatmap
  Plotly.react("rg-heatmap", [{
    z: r.heatmap_z, x, y: r.labels, type: "heatmap",
    zmid: 0, zmin: -10, zmax: 10,
    colorscale: [[0, "#A31621"], [0.5, "#FFFFFF"], [1, "#2E8B57"]],
    colorbar: { title: "%", thickness: 12, len: 0.8 },
    hovertemplate: "%{y}<br>%{x|%b %Y}<br>%{z:.2f}%<extra></extra>",
  }], L({
    xaxis: { type: "date", gridcolor: "#F0F0F0" },
    yaxis: { automargin: true, autorange: "reversed" },
    margin: { l: 110, r: 30, t: 20, b: 50 },
  }), CONFIG);

  // Korrelationsmatrix
  Plotly.react("rg-corr", [{
    z: r.corr_z, x: r.corr_labels, y: r.corr_labels, type: "heatmap",
    zmin: -1, zmax: 1, zmid: 0,
    colorscale: [[0, "#1C4E80"], [0.5, "#FFFFFF"], [1, "#A31621"]],
    colorbar: { thickness: 12, len: 0.8 },
    hovertemplate: "%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>",
  }], L({
    xaxis: { tickangle: -45, automargin: true },
    yaxis: { automargin: true, autorange: "reversed" },
    margin: { l: 110, r: 30, t: 20, b: 110 },
  }), CONFIG);

  // Region-Vergleich
  const regs = r.region_stats.map(s => s.region);
  Plotly.react("rg-bars", [
    { x: regs, y: r.region_stats.map(s => s.avg_return), name: "Ø Rendite (%)", type: "bar",
      marker: { color: COLORS.accent }, hovertemplate: "%{x}<br>Ø Rendite %{y:.2f}%<extra></extra>" },
    { x: regs, y: r.region_stats.map(s => s.loss_share * 100), name: "Verlustmonate (%)", type: "bar",
      marker: { color: COLORS.accent2 }, hovertemplate: "%{x}<br>Verlustmonate %{y:.1f}%<extra></extra>" },
    { x: regs, y: r.region_stats.map(s => s.gpr_corr * 100), name: "GPR-Korr. (×100)", type: "bar",
      marker: { color: COLORS.muted }, hovertemplate: "%{x}<br>GPR-Korr %{y:.1f}<extra></extra>" },
  ], L({
    barmode: "group", yaxis: { gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.18 },
  }), CONFIG);
}

/* =================== 4. MARKTREGIMES =================== */
function renderRegime() {
  const km = D.kmeans;
  document.getElementById("km-badge").textContent = `k = ${km.best_k} · Silhouette ${km.silhouette}`;
  document.getElementById("km-pca-sub").textContent =
    `PC1 (${(km.pca_var[0] * 100).toFixed(1)} %) · PC2 (${(km.pca_var[1] * 100).toFixed(1)} %)`;

  const x = km.timeline_months.map(md);
  // Regime-Zeitstrahl: Rendite-Linie + farbige Hintergrund-Bänder
  const shapes = [];
  for (let i = 0; i < km.timeline_dominant.length; i++) {
    const c = km.timeline_dominant[i];
    if (c == null) continue;
    const x0 = md(km.timeline_months[i]);
    const x1 = i + 1 < km.timeline_months.length ? md(km.timeline_months[i + 1]) : x0;
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0, x1, y0: 0, y1: 1,
      fillcolor: COLORS.cluster[c], opacity: 0.12, line: { width: 0 }, layer: "below" });
  }
  const legendTraces = [];
  for (let c = 0; c < km.best_k; c++) {
    legendTraces.push({ x: [null], y: [null], type: "scatter", mode: "markers",
      name: km.cluster_labels[c], marker: { color: COLORS.cluster[c], size: 10, symbol: "square" } });
  }
  Plotly.react("km-timeline", [
    { x, y: km.timeline_return, type: "scatter", mode: "lines", name: "Ø Rendite",
      line: { color: COLORS.text, width: 1.5 }, hovertemplate: "%{x|%b %Y}<br>%{y:.2f}%<extra></extra>" },
    ...legendTraces,
  ], L({
    shapes,
    xaxis: { type: "date", gridcolor: "#F0F0F0" },
    yaxis: { title: "Ø Aktienrendite (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.18 },
  }), CONFIG);

  // PCA-Scatter
  const ptraces = [];
  for (let c = 0; c < km.best_k; c++) {
    const p = km.pca_points[c];
    ptraces.push({ x: p.x, y: p.y, type: "scattergl", mode: "markers", name: `Regime ${c}`,
      marker: { color: COLORS.cluster[c], size: 5, opacity: 0.6 },
      text: p.name.map((n, i) => `${n} · ${p.month[i]}`),
      hovertemplate: "%{text}<extra>Regime " + c + "</extra>" });
  }
  Plotly.react("km-pca", ptraces, L({
    xaxis: { title: `PC1 (${(km.pca_var[0] * 100).toFixed(1)} %)`, gridcolor: "#F0F0F0" },
    yaxis: { title: `PC2 (${(km.pca_var[1] * 100).toFixed(1)} %)`, gridcolor: "#F0F0F0" },
    legend: { orientation: "h", y: -0.18 },
  }), CONFIG);

  // Cluster-Profile (gruppierte Balken je Feature)
  const featLabels = { GPRD_mean: "GPR Ø", Stock_monthly_pct: "Rendite", stock_vol6: "Volatilität",
    GPRD_ACT_mean: "GPR-ACT", GPRD_THREAT_mean: "GPR-THREAT" };
  const proftraces = [];
  for (let c = 0; c < km.best_k; c++) {
    proftraces.push({ x: km.profile_features.map(f => featLabels[f] || f), y: km.profiles[c],
      type: "bar", name: `Regime ${c}`, marker: { color: COLORS.cluster[c] },
      hovertemplate: "%{x}<br>z = %{y:.2f}<extra>Regime " + c + "</extra>" });
  }
  Plotly.react("km-profiles", proftraces, L({
    barmode: "group", yaxis: { title: "z-Score vs. Ø", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.18 },
  }), CONFIG);

  // Tabelle
  const head = "<tr><th>Regime</th><th>n</th><th>Ø Rendite</th><th>Ø GPR</th><th>Ø Vol.</th><th>Verlustmon.</th><th>GPR-Spike</th></tr>";
  const rows = km.stats.map(s => {
    const rc = s.avg_return < 0 ? "neg" : "pos";
    return `<tr><td><span style="color:${COLORS.cluster[s.cluster]}">●</span> Regime ${s.cluster}</td>`
      + `<td>${s.n}</td><td class="${rc}">${fmtPct(s.avg_return)}</td>`
      + `<td>${fmtNum(s.avg_gpr, 1)}</td><td>${fmtNum(s.avg_vol)}</td>`
      + `<td>${(s.loss_share * 100).toFixed(0)} %</td><td>${(s.spike_share * 100).toFixed(0)} %</td></tr>`;
  }).join("");
  document.getElementById("km-table").innerHTML = head + rows;
}

/* =================== 5. VORHERSAGE (KNN) =================== */
function renderForecast() {
  const knn = D.knn;
  const card = (dir) => {
    const m = knn[dir];
    return `<div class="kpi"><div class="value accent">AUC ${m.test_auc.toFixed(3)}</div>`
      + `<div class="label"><b>${m.label}</b><br>F1 ${m.f1.toFixed(3)} · Accuracy ${m.accuracy.toFixed(3)} · CV-AUC ${m.cv_auc.toFixed(3)}<br>`
      + `k=${m.best_k} · ${m.best_metric} · PCA=${m.n_pca}</div></div>`;
  };
  document.getElementById("knn-kpi").innerHTML = card("A") + card("B");

  // ROC
  Plotly.react("knn-roc", [
    { x: knn.A.roc.fpr, y: knn.A.roc.tpr, type: "scatter", mode: "lines", name: `A · AUC ${knn.A.test_auc.toFixed(3)}`,
      line: { color: COLORS.accent2, width: 2.2 }, fill: "tozeroy", fillcolor: "rgba(163,22,33,0.06)" },
    { x: knn.B.roc.fpr, y: knn.B.roc.tpr, type: "scatter", mode: "lines", name: `B · AUC ${knn.B.test_auc.toFixed(3)}`,
      line: { color: COLORS.accent, width: 2.2 } },
    { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Zufall",
      line: { color: COLORS.muted, width: 1, dash: "dash" }, hoverinfo: "skip" },
  ], L({
    xaxis: { title: "Falsch-Positiv-Rate", gridcolor: "#F0F0F0", range: [0, 1] },
    yaxis: { title: "Richtig-Positiv-Rate", gridcolor: "#F0F0F0", range: [0, 1.02] },
    legend: { x: 0.98, y: 0.05, xanchor: "right", yanchor: "bottom" },
  }), CONFIG);

  drawFeatureImportance("A");

  // Konfusionsmatrizen
  drawCM("knn-cm-a", knn.A.cm, ["Steigt", "Fällt"], [[0, "#FFFFFF"], [1, COLORS.accent2]]);
  drawCM("knn-cm-b", knn.B.cm, ["GPR fällt", "GPR steigt"], [[0, "#FFFFFF"], [1, COLORS.accent]]);

  const dir = knn.delta_auc < 0 ? "Richtung B (Aktien → GPR)" : "Richtung A (GPR → Aktien)";
  document.getElementById("knn-info").innerHTML =
    `<b>Δ AUC (A − B) = ${knn.delta_auc >= 0 ? "+" : ""}${knn.delta_auc.toFixed(3)}.</b> `
    + `Beide Richtungen liegen über der Zufallsbaseline (AUC &gt; 0,50). `
    + `${dir} ist geringfügig besser klassifizierbar — konsistent mit der Hypothese, dass `
    + `Aktienmärkte und GPR sich gegenseitig nur schwach vorhersagen lassen.`;
}

function drawFeatureImportance(dir) {
  const m = D.knn[dir];
  const entries = Object.entries(m.mi_scores).sort((a, b) => a[1] - b[1]);
  const sel = new Set(m.selected_features);
  const baseColor = dir === "A" ? COLORS.accent2 : COLORS.accent;
  Plotly.react("knn-fi", [{
    type: "bar", orientation: "h",
    x: entries.map(e => e[1]), y: entries.map(e => e[0]),
    marker: { color: entries.map(e => sel.has(e[0]) ? baseColor : "#DDDDDD") },
    hovertemplate: "%{y}<br>MI %{x:.4f}<extra></extra>",
  }], L({
    xaxis: { title: "Mutual-Information-Score", gridcolor: "#F0F0F0" },
    yaxis: { automargin: true, tickfont: { size: 10 } },
    margin: { l: 150, r: 20, t: 20, b: 45 },
  }), CONFIG);
}

function drawCM(id, cm, labels, scale) {
  const total = cm.flat().reduce((a, b) => a + b, 0);
  const text = cm.map(row => row.map(v => `${v}<br>${(v / total * 100).toFixed(0)} %`));
  Plotly.react(id, [{
    z: cm, x: labels.map(l => "Pred: " + l), y: labels.map(l => "True: " + l),
    type: "heatmap", colorscale: scale, showscale: false,
    text, texttemplate: "%{text}", textfont: { size: 14 },
    hovertemplate: "%{y} / %{x}<br>n = %{z}<extra></extra>",
  }], L({
    xaxis: { side: "top" }, yaxis: { autorange: "reversed", automargin: true },
    margin: { l: 90, r: 20, t: 40, b: 20 },
  }), CONFIG);
}

/* =================== 6. EVENT-STUDIE =================== */
function renderEvents() {
  drawEventDir("A");
  // ACT vs THREAT
  const ev = D.events, rel = ev.ACT.rel_days;
  Plotly.react("ev-actthreat", [
    { x: rel, y: ev.ACT.mean_car, type: "scatter", mode: "lines+markers", name: `ACT (n=${ev.ACT.n})`,
      line: { color: COLORS.accent2, width: 2.2 } },
    { x: rel, y: ev.ACT.mean_car.map((v, i) => v + ev.ACT.ci_car[i]), type: "scatter", mode: "lines",
      line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: rel, y: ev.ACT.mean_car.map((v, i) => v - ev.ACT.ci_car[i]), type: "scatter", mode: "lines",
      fill: "tonexty", fillcolor: "rgba(163,22,33,0.12)", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: rel, y: ev.THREAT.mean_car, type: "scatter", mode: "lines+markers", name: `THREAT (n=${ev.THREAT.n})`,
      line: { color: COLORS.accent, width: 2.2 } },
    { x: rel, y: ev.THREAT.mean_car.map((v, i) => v + ev.THREAT.ci_car[i]), type: "scatter", mode: "lines",
      line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: rel, y: ev.THREAT.mean_car.map((v, i) => v - ev.THREAT.ci_car[i]), type: "scatter", mode: "lines",
      fill: "tonexty", fillcolor: "rgba(28,78,128,0.12)", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
  ], L({
    shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: COLORS.muted, dash: "dot", width: 1 } }],
    xaxis: { title: "Tage relativ zum Event", gridcolor: "#F0F0F0", dtick: 1 },
    yaxis: { title: "Ø Kumulatives AR (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.2 },
    annotations: [{ x: 0.5, y: 1.05, xref: "paper", yref: "paper", showarrow: false,
      text: `ACT CAR_post ${fmtPct(ev.ACT.car_post)} · THREAT CAR_post ${fmtPct(ev.THREAT.car_post)}`,
      font: { size: 11, color: COLORS.muted } }],
  }), CONFIG);

  // Regression B
  const rb = ev.regB_scatter;
  document.getElementById("ev-reg-sub").textContent =
    `β = ${rb.slope} · R² = ${rb.r2} · p = ${rb.p.toExponential(2)} · n = ${rb.x.length}`;
  Plotly.react("ev-reg", [
    { x: rb.x, y: rb.y, type: "scattergl", mode: "markers", name: "Events",
      marker: { color: COLORS.accent, size: 5, opacity: 0.4 }, hoverinfo: "skip" },
    { x: rb.line_x, y: rb.line_y, type: "scatter", mode: "lines", name: "Regression",
      line: { color: COLORS.accent2, width: 2.2 } },
  ], L({
    xaxis: { title: "Aktien-Schock am Event-Tag (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    yaxis: { title: "CAR GPR (t+1…t+5, %)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.2 },
  }), CONFIG);
}

function drawEventDir(dir) {
  const e = D.events[dir], rel = e.rel_days;
  const target = dir === "A" ? "Aktien" : "GPR";
  const thr = dir === "A" ? D.events.thr_a : D.events.thr_b;
  document.getElementById("ev-ar-sub").textContent =
    `Ziel: ${target} · n = ${e.n} Events · Schwelle ${thr}`;
  document.getElementById("ev-car-sub").textContent =
    `CAR_post = ${fmtPct(e.car_post)} · p = ${e.p.toExponential(2)}`;

  Plotly.react("ev-ar", [{
    x: rel, y: e.mean_ar, type: "bar", name: `Ø AR ${target}`,
    marker: { color: e.mean_ar.map(v => v >= 0 ? COLORS.positive : COLORS.negative) },
    error_y: { type: "data", array: e.ci_ar, color: COLORS.muted, thickness: 1, width: 3 },
    hovertemplate: "t=%{x}<br>AR %{y:.3f}%<extra></extra>",
  }], L({
    shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: COLORS.accent2, dash: "dot", width: 1.2 } }],
    xaxis: { title: "Tage relativ zum Event", gridcolor: "#F0F0F0", dtick: 1 },
    yaxis: { title: `Ø Abnormal Return ${target} (%)`, gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    showlegend: false,
  }), CONFIG);

  Plotly.react("ev-car", [
    { x: rel, y: e.mean_car, type: "scatter", mode: "lines+markers", name: `Ø CAR ${target}`,
      line: { color: COLORS.accent2, width: 2.2 } },
    { x: rel, y: e.mean_car.map((v, i) => v + e.ci_car[i]), type: "scatter", mode: "lines",
      line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: rel, y: e.mean_car.map((v, i) => v - e.ci_car[i]), type: "scatter", mode: "lines",
      fill: "tonexty", fillcolor: "rgba(163,22,33,0.12)", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
  ], L({
    shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: COLORS.accent, dash: "dot", width: 1.2 } }],
    xaxis: { title: "Tage relativ zum Event", gridcolor: "#F0F0F0", dtick: 1 },
    yaxis: { title: `Ø Kumulatives AR ${target} (%)`, gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    showlegend: false,
  }), CONFIG);
}

/* =================== 7. METHODIK =================== */
function renderMethod() {
  const km = D.kmeans, knn = D.knn, ev = D.events;
  document.getElementById("method-grid").innerHTML = [
    `<div class="method"><span class="tag">Unüberwacht</span><h3>K-Means Clustering</h3>
      <p>Erkennung wiederkehrender Marktregime aus monatlichen GPR- und Aktien-Features.</p>
      <div class="metric">k = ${km.best_k} · Silhouette ${km.silhouette}</div></div>`,
    `<div class="method"><span class="tag">Überwacht</span><h3>KNN-Klassifikation</h3>
      <p>Bidirektionale Vorhersagbarkeit mit Feature-Engineering, MI-Selektion und PCA.</p>
      <div class="metric">AUC A = ${knn.A.test_auc.toFixed(3)} · B = ${knn.B.test_auc.toFixed(3)}</div></div>`,
    `<div class="method"><span class="tag">Ereignisbasiert</span><h3>Event-Studie</h3>
      <p>Abnormale Renditen im ±5-Tage-Fenster um GPR-Schocks (95. Perzentil).</p>
      <div class="metric">n = ${ev.A.n} Events · CAR_post ${fmtPct(ev.A.car_post)}</div></div>`,
    `<div class="method"><span class="tag">Überwacht</span><h3>Random Forest</h3>
      <p>Ergänzende nichtlineare Klassifikation (Jupyter-Notebooks).</p>
      <div class="metric">Siehe results/random_forest/</div></div>`,
  ].join("");

  document.getElementById("pipeline").innerHTML =
`Rohdaten (indexData.csv + GPR.xls)
   <span class="arrow">↓ build_dataset.py</span>
stocks_gpr_<span class="out">daily</span> / <span class="out">monthly</span> / <span class="out">features</span>.csv
   <span class="arrow">├→</span> Event-Studie   <span class="arrow">→</span> results/lineare_regression/
   <span class="arrow">├→</span> K-Means        <span class="arrow">→</span> results/k_means/
   <span class="arrow">├→</span> KNN            <span class="arrow">→</span> results/knn/
   <span class="arrow">└→</span> Random Forest  <span class="arrow">→</span> results/random_forest/`;
}

/* =================== NAVIGATION + INIT =================== */
const RENDERERS = {
  overview: renderOverview, timeseries: renderTimeseries, regions: renderRegions,
  regime: renderRegime, forecast: renderForecast, events: renderEvents, method: renderMethod,
};
const rendered = {};

function navigate(target) {
  document.querySelectorAll(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.target === target));
  document.querySelectorAll(".section").forEach(s => s.classList.toggle("active", s.id === target));
  if (!rendered[target]) { RENDERERS[target](); rendered[target] = true; }
  // Resize, da Plotly in versteckten Containern keine Größe kennt
  setTimeout(() => window.dispatchEvent(new Event("resize")), 30);
}

function initToggles() {
  // Übersicht GPR-Toggle
  document.getElementById("ov-gpr-toggle").addEventListener("click", e => {
    if (e.target.tagName !== "BUTTON") return;
    [...e.currentTarget.children].forEach(b => b.classList.toggle("active", b === e.target));
    drawOverviewTS(e.target.dataset.v);
  });
  // KNN Feature-Importance-Toggle
  document.getElementById("knn-fi-toggle").addEventListener("click", e => {
    if (e.target.tagName !== "BUTTON") return;
    [...e.currentTarget.children].forEach(b => b.classList.toggle("active", b === e.target));
    drawFeatureImportance(e.target.dataset.v);
  });
  // Event-Richtungs-Toggle
  document.getElementById("ev-dir-toggle").addEventListener("click", e => {
    if (e.target.tagName !== "BUTTON") return;
    [...e.currentTarget.children].forEach(b => b.classList.toggle("active", b === e.target));
    drawEventDir(e.target.dataset.v);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("nav").addEventListener("click", e => {
    const item = e.target.closest(".nav-item");
    if (item) navigate(item.dataset.target);
  });
  document.getElementById("foot").innerHTML =
    `${D.meta.months.length} Monate · 12 Indizes<br>Stand: ${D.meta.generated}`;
  initToggles();
  navigate("overview");
});
