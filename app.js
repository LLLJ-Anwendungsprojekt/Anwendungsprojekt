/* ===================================================================
   GPR & Aktienmärkte — Dashboard Rendering (Plotly.js)
   Liest window.DASHBOARD_DATA (aus data.js) und rendert alle Abschnitte.
   Struktur: Übersicht · Lineare Regression · K-Means · Random Forest · KNN · Synthese.
   Die vier Verfahren werden gleichberechtigt in je einem Tab dargestellt.
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
function eventShapes() {
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
  document.getElementById("kpi-row").innerHTML = [
    `<div class="kpi"><div class="value accent">${k.n_months}</div><div class="label">Monate · 2000–2021</div></div>`,
    `<div class="kpi"><div class="value accent">${k.n_indices}</div><div class="label">Indizes · ${k.n_regions} Regionen</div></div>`,
    `<div class="kpi"><div class="value neg">${fmtPct(k.avg_return_spike)}</div><div class="label">Ø Rendite bei GPR-Spikes (vs. ${fmtPct(k.avg_return_normal)})</div></div>`,
    `<div class="kpi"><div class="value neg">${fmtNum(k.corr_threat, 3)}</div><div class="label">Korr. THREAT → Markt (ACT: ${fmtNum(k.corr_act, 3)})</div></div>`,
  ].join("");

  renderDataBasis();
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
      fillcolor: "rgba(127,168,208,0.3)", box: { visible: true }, meanline: { visible: true }, points: false,
      hoverinfo: "skip" },
    { y: o.spike_box.spike, type: "violin", name: "GPR-Spike", line: { color: COLORS.accent2 },
      fillcolor: "rgba(163,22,33,0.3)", box: { visible: true }, meanline: { visible: true }, points: false,
      hoverinfo: "skip" },
  ], L({
    yaxis: { title: "Aktienrendite (%)", gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    annotations: [{ x: 0.5, y: 1.0, xref: "paper", yref: "paper", showarrow: false,
      text: `Ø ${fmtPct(k.avg_return_spike)} vs. ${fmtPct(k.avg_return_normal)}`,
      font: { size: 12, color: COLORS.text } }],
    showlegend: false,
  }), CONFIG);
}

/* Datengrundlage: Indizes je Region, Eckdaten und Begriffsdefinitionen */
function renderDataBasis() {
  const m = D.meta, k = D.overview.kpis, months = m.months;
  const range = `${months[0]} – ${months[months.length - 1]}`;

  const byRegion = m.region_order.map(r => {
    const names = Object.values(m.indices).filter(x => x.region === r).map(x => x.name);
    return `<li><span style="color:${COLORS.region[r]}">●</span> <b>${r}</b> `
      + `<span class="db-muted">(${names.length})</span>: ${names.join(", ")}</li>`;
  }).join("");

  const facts = [
    ["Zeitraum", `${range} · ${k.n_months} Monate`],
    ["Umfang", `${k.n_indices} Indizes · ${k.n_regions} Regionen`],
    ["Granularität", "Tagesdaten (Lineare Regression) · Monatspanel (übrige Verfahren)"],
    ["Quelle", "Börsenkurse (Yahoo Finance) · GPR-Index (Caldara & Iacoviello)"],
  ].map(f => `<tr><td>${f[0]}</td><td>${f[1]}</td></tr>`).join("");

  const defs = [
    ["GPR", "Geopolitical Risk Index — Häufigkeit geopolitischer Risikobegriffe in der Presse"],
    ["GPR-ACT", "Realisierte Ereignisse — tatsächliche geopolitische Vorfälle"],
    ["GPR-THREAT", "Bedrohungen — angedrohte bzw. befürchtete Ereignisse"],
    ["GPR-Spike", `Monat mit |GPR-Änderung| > 1 SD (~${(k.spike_share * 100).toFixed(0)} % der Monate)`],
  ].map(d => `<tr><td><b>${d[0]}</b></td><td>${d[1]}</td></tr>`).join("");

  document.getElementById("ov-databasis").innerHTML =
    `<div class="db-grid">
       <div>
         <div class="db-head">Indizes nach Region</div>
         <ul class="db-list">${byRegion}</ul>
         <div class="db-head" style="margin-top:14px">Eckdaten</div>
         <table class="db-table">${facts}</table>
       </div>
       <div>
         <div class="db-head">Begriffe</div>
         <table class="db-table">${defs}</table>
       </div>
     </div>`;
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

/* =================== 2. K-MEANS =================== */
function renderKmeans() {
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
    legend: { orientation: "h", y: -0.22, font: { size: 10 } },
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

  // Kernaussage
  const hi = km.stats.reduce((a, b) => b.avg_gpr > a.avg_gpr ? b : a);
  const lo = km.stats.reduce((a, b) => b.avg_gpr < a.avg_gpr ? b : a);
  document.getElementById("km-verdict").innerHTML =
    `K-Means trennt den Zeitraum in <b>${km.best_k} stabile Marktregime</b> (Silhouette ${km.silhouette}). `
    + `Das <b>Hoch-GPR-Regime</b> (Regime ${hi.cluster}, Ø GPR ${fmtNum(hi.avg_gpr, 0)}) ist mit Ø Rendite `
    + `${fmtPct(hi.avg_return)}, ${(hi.loss_share * 100).toFixed(0)} % Verlustmonaten und erhöhter Volatilität `
    + `das ungünstigste Umfeld; das Niedrig-GPR-Regime (Regime ${lo.cluster}) liegt bei ${fmtPct(lo.avg_return)}. `
    + `Erhöhte geopolitische Anspannung fällt also systematisch mit <b>schwächeren, volatileren Märkten</b> zusammen — `
    + `ein belegter Zusammenhang, keine bewiesene Kausalität.`;
}

/* =================== Klassifikator-Bausteine (gemeinsam für KNN & RF) ====== */
function drawROC(elId, m, colorA, colorB, suffix) {
  Plotly.react(elId, [
    { x: m.A.roc.fpr, y: m.A.roc.tpr, type: "scatter", mode: "lines", name: `Richtung A · ${m.A.test_auc.toFixed(3)}${suffix}`,
      line: { color: colorA, width: 2.2 } },
    { x: m.B.roc.fpr, y: m.B.roc.tpr, type: "scatter", mode: "lines", name: `Richtung B · ${m.B.test_auc.toFixed(3)}${suffix}`,
      line: { color: colorB, width: 2.2 } },
    { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Zufall",
      line: { color: COLORS.muted, width: 1, dash: "dot" }, hoverinfo: "skip" },
  ], L({
    xaxis: { title: "Falsch-Positiv-Rate", gridcolor: "#F0F0F0", range: [0, 1] },
    yaxis: { title: "Richtig-Positiv-Rate", gridcolor: "#F0F0F0", range: [0, 1.02] },
    legend: { x: 0.98, y: 0.02, xanchor: "right", yanchor: "bottom", font: { size: 11 } },
  }), CONFIG);
}

function drawFI(elId, entries, colors, xtitle) {
  Plotly.react(elId, [{
    type: "bar", orientation: "h",
    x: entries.map(e => e[1]), y: entries.map(e => e[0]),
    marker: { color: colors },
    hovertemplate: "%{y}<br>%{x:.4f}<extra></extra>",
  }], L({
    xaxis: { title: xtitle, gridcolor: "#F0F0F0" },
    yaxis: { automargin: true, tickfont: { size: 10 } },
    margin: { l: 155, r: 20, t: 20, b: 45 },
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

/* =================== 3. KNN =================== */
let knnFiDir = "A";
function renderKnn() {
  const knn = D.knn;
  document.getElementById("knn-table").innerHTML =
    "<tr><th>Kennzahl</th><th>Richtung A: GPR→Aktien</th><th>Richtung B: Aktien→GPR</th></tr>"
    + `<tr><td>Test-AUC (Holdout)</td><td>${knn.A.test_auc.toFixed(3)}</td><td>${knn.B.test_auc.toFixed(3)}</td></tr>`
    + `<tr><td>CV-AUC</td><td>${knn.A.cv_auc.toFixed(3)}</td><td>${knn.B.cv_auc.toFixed(3)}</td></tr>`
    + `<tr><td>F1</td><td>${knn.A.f1.toFixed(3)}</td><td>${knn.B.f1.toFixed(3)}</td></tr>`
    + `<tr><td>Accuracy</td><td>${knn.A.accuracy.toFixed(3)}</td><td>${knn.B.accuracy.toFixed(3)}</td></tr>`
    + `<tr><td>Bestes k · Metrik</td><td>${knn.A.best_k} · ${knn.A.best_metric}</td><td>${knn.B.best_k} · ${knn.B.best_metric}</td></tr>`
    + `<tr><td>PCA-Komponenten</td><td>${knn.A.n_pca}</td><td>${knn.B.n_pca}</td></tr>`;

  drawROC("knn-roc", knn, COLORS.accent2, COLORS.accent, "");
  drawKnnFI();
  drawCM("knn-cm-a", knn.A.cm, ["Steigt", "Fällt"], [[0, "#FFFFFF"], [1, COLORS.accent2]]);
  drawCM("knn-cm-b", knn.B.cm, ["GPR fällt", "GPR steigt"], [[0, "#FFFFFF"], [1, COLORS.accent]]);

  // Kernaussage
  const strongerDir = Math.abs(knn.delta_auc) < 0.01 ? "ist keine Richtung klar überlegen (≈ symmetrisch)"
    : (knn.delta_auc > 0 ? "liegt Richtung A (GPR → Aktien) knapp vorn" : "liegt Richtung B (Aktien → GPR) knapp vorn");
  document.getElementById("knn-verdict").innerHTML =
    `Auf dem temporalen 20 %-Holdout erreichen beide Richtungen `
    + `AUC ${knn.A.test_auc.toFixed(2)} (A) bzw. ${knn.B.test_auc.toFixed(2)} (B) — nur knapp über der Zufallslinie (0,50). `
    + `Im direkten Vergleich ${strongerDir} (Δ AUC = ${knn.delta_auc.toFixed(3)}). `
    + `Die Bewertung erfolgt <b>out-of-sample</b> und beschreibt damit die Übertragung auf ungesehene Zeiträume; `
    + `wegen der Nähe zum Zufall ist die erreichte Vorhersagbarkeit gering.`;
}

function drawKnnFI() {
  const m = D.knn[knnFiDir];
  const baseColor = knnFiDir === "A" ? COLORS.accent2 : COLORS.accent;
  const entries = Object.entries(m.mi_scores).sort((a, b) => a[1] - b[1]);
  const sel = new Set(m.selected_features);
  const colors = entries.map(e => sel.has(e[0]) ? baseColor : "#DDDDDD");
  drawFI("knn-fi", entries, colors, "Mutual-Information-Score");
}

/* =================== 4. RANDOM FOREST =================== */
let rfFiDir = "A";
function renderRf() {
  const rf = D.rf;
  document.getElementById("rf-table").innerHTML =
    "<tr><th>Kennzahl</th><th>Richtung A: GPR→Aktien</th><th>Richtung B: Aktien→GPR</th></tr>"
    + `<tr><td>AUC (in-sample)</td><td>${rf.A.test_auc.toFixed(3)}</td><td>${rf.B.test_auc.toFixed(3)}</td></tr>`
    + `<tr><td>F1</td><td>${rf.A.f1.toFixed(3)}</td><td>${rf.B.f1.toFixed(3)}</td></tr>`
    + `<tr><td>Accuracy</td><td>${rf.A.accuracy.toFixed(3)}</td><td>${rf.B.accuracy.toFixed(3)}</td></tr>`
    + `<tr><td>Precision</td><td>${rf.A.precision.toFixed(3)}</td><td>${rf.B.precision.toFixed(3)}</td></tr>`
    + `<tr><td>Recall</td><td>${rf.A.recall.toFixed(3)}</td><td>${rf.B.recall.toFixed(3)}</td></tr>`
    + `<tr><td>n (Beobachtungen)</td><td>${rf.A.n}</td><td>${rf.B.n}</td></tr>`;

  drawROC("rf-roc", rf, COLORS.accent2, COLORS.accent, " (i.s.)");
  drawRfFI();
  drawCM("rf-cm-a", rf.A.cm, ["Steigt", "Fällt"], [[0, "#FFFFFF"], [1, COLORS.accent2]]);
  drawCM("rf-cm-b", rf.B.cm, ["GPR fällt", "GPR steigt"], [[0, "#FFFFFF"], [1, COLORS.accent]]);

  document.getElementById("rf-info").innerHTML =
    `Der Random Forest wird hier <b>in-sample</b> trainiert und ausgewertet (Training und Test auf demselben `
    + `Zeitraum 2001–2021). Die hohe AUC (A ${rf.A.test_auc.toFixed(2)} / B ${rf.B.test_auc.toFixed(2)}) ist daher `
    + `<b>optimistisch verzerrt</b> und beschreibt nur, wie gut das Modell die Trainingsdaten erfasst — sie ist `
    + `<b>keine</b> Prognosegüte. Der direkte Kontrast zum out-of-sample bewerteten KNN `
    + `(AUC A ${D.knn.A.test_auc.toFixed(2)} / B ${D.knn.B.test_auc.toFixed(2)}) illustriert die In-Sample-Falle: `
    + `Aus In-Sample-Güte lässt sich nicht auf echte Vorhersagekraft schließen.`;
}

function drawRfFI() {
  const m = D.rf[rfFiDir];
  const baseColor = rfFiDir === "A" ? COLORS.accent2 : COLORS.accent;
  const entries = Object.entries(m.importances).sort((a, b) => a[1] - b[1]);
  const colors = entries.map(() => baseColor);
  drawFI("rf-fi", entries, colors, "Gini-Importance");
}

/* =================== 5. LINEARE REGRESSION (Event-Studie) =================== */
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

  // Kernaussage
  const sig = p => p < 0.01 ? "hoch signifikant" : p < 0.05 ? "signifikant" : p < 0.1 ? "schwach signifikant" : "nicht signifikant";
  document.getElementById("ev-verdict").innerHTML =
    `<b>Kleine, gerichtete Effekte.</b> Nach einem GPR-Schock (Richtung A) beträgt die kumulierte Aktien-Reaktion `
    + `CAR_post ${fmtPct(ev.A.car_post)} (${sig(ev.A.p)}); nach einem Aktien-Crash (Richtung B) reagiert der GPR mit `
    + `CAR_post ${fmtPct(ev.B.car_post)} (${sig(ev.B.p)}). `
    + `Die Regression der kumulierten Reaktion auf die Schockstärke erklärt jedoch kaum Streuung `
    + `(R² A ${fmtNum(ev.A.reg_r2, 4)} / B ${fmtNum(ev.B.reg_r2, 4)}) — der Zusammenhang ist `
    + `<b>nachweisbar, aber ökonomisch klein</b>. Tendenziell wirken <b>THREAT-Schocks stärker als ACT-Schocks</b>.`;
}

function drawEventDir(dir) {
  const e = D.events[dir], rel = e.rel_days;
  const target = dir === "A" ? "Aktien" : "GPR";
  const thr = dir === "A" ? D.events.thr_a : D.events.thr_b;
  document.getElementById("ev-ar-sub").textContent =
    `Ziel: ${target} · n = ${e.n} Events · Schwelle ${thr}`;
  document.getElementById("ev-car-sub").textContent =
    `CAR_post = ${fmtPct(e.car_post)} · p = ${e.p.toExponential(2)}`;

  drawEventReg(dir);

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

/* Lineare Regression: kumulierte Reaktion (CAR_post) ~ Schockstärke am Event-Tag */
function drawEventReg(dir) {
  const e = D.events[dir], sc = e.reg_scatter;
  const target = dir === "A" ? "Aktien" : "GPR";
  const xtitle = dir === "A" ? "GPR-Schock am Event-Tag (%)" : "Aktien-Schock am Event-Tag (%)";
  const intercept = e.reg_intercept != null ? e.reg_intercept
    : (sc.line_y[0] - e.reg_slope * sc.line_x[0]);
  const sign = intercept >= 0 ? "+" : "−";

  document.getElementById("ev-reg-sub").textContent =
    `OLS · β = ${fmtNum(e.reg_slope, 4)} · R² = ${fmtNum(e.reg_r2, 4)} · p = ${e.reg_p.toExponential(2)} · n = ${sc.x.length}`;

  Plotly.react("ev-reg", [
    { x: sc.x, y: sc.y, type: "scattergl", mode: "markers", name: "Events",
      marker: { color: COLORS.accent, size: 5, opacity: 0.45 },
      hovertemplate: "Schock %{x:.2f}%<br>CAR %{y:.2f}%<extra></extra>" },
    { x: sc.line_x, y: sc.line_y, type: "scatter", mode: "lines", name: "Regressionsgerade",
      line: { color: COLORS.accent2, width: 2.6 }, hoverinfo: "skip" },
  ], L({
    shapes: [
      { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: COLORS.muted, dash: "dot", width: 1 } },
    ],
    xaxis: { title: xtitle, gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    yaxis: { title: `CAR ${target} (t+1 … t+5, %)`, gridcolor: "#F0F0F0", zeroline: true, zerolinecolor: "#E0E0E0" },
    legend: { orientation: "h", y: -0.18 },
    annotations: [{ x: 0.02, y: 0.98, xref: "paper", yref: "paper", showarrow: false, align: "left",
      text: `y = ${fmtNum(e.reg_slope, 4)}·x ${sign} ${fmtNum(Math.abs(intercept), 3)}<br>R² = ${fmtNum(e.reg_r2, 4)}`,
      font: { size: 11, color: COLORS.muted } }],
  }), CONFIG);
}

/* =================== 6. SYNTHESE =================== */
function renderSynthesis() {
  const knn = D.knn, rf = D.rf, ev = D.events, km = D.kmeans, k = D.overview.kpis;
  const sig = p => p < 0.01 ? "***" : p < 0.05 ? "**" : p < 0.1 ? "*" : "n.s.";
  const sigWord = p => p < 0.01 ? "hoch signifikant" : p < 0.05 ? "signifikant" : p < 0.1 ? "schwach signifikant" : "nicht signifikant";

  // ── ① Antwort-Banner ──────────────────────────────────────────────
  const dirWord = Math.abs(knn.delta_auc) < 0.005 ? "Keine Richtung dominiert klar"
    : (knn.delta_auc < 0 ? "Aktien laufen dem GPR leicht voraus" : "GPR läuft den Aktien leicht voraus");
  document.getElementById("syn-banner").innerHTML =
    `<div class="lead">Gesamtbefund</div>`
    + `<b>${dirWord}</b> — doch alle Vorhersagesignale sind schwach (beste out-of-sample-AUC ≈ ${knn.A.test_auc.toFixed(2)}, `
    + `nur knapp über dem Zufall). Geopolitisches Risiko bewegt die Märkte spürbar vor allem in `
    + `<b>Spike-Phasen</b> (Ø ${fmtPct(k.avg_return_spike)} vs. ${fmtPct(k.avg_return_normal)} sonst).`;

  // ── ② Verfahren-Vergleich (neutral, ohne Rangfolge) ────────────────
  const evBest = ev.A.p < ev.B.p ? ev.A : ev.B;        // signifikantere Richtung
  const evDir = ev.A.p < ev.B.p ? "A" : "B";
  const methods = [
    { name: "Lineare Regression", type: "deskriptiv", metric: fmtPct(evBest.car_post),
      label: `CAR_post · Richtung ${evDir} (${sigWord(evBest.p)})`,
      take: "Gerichteter Effekt nachweisbar, aber ökonomisch klein (R² ≈ 0)." },
    { name: "K-Means", type: "unüberwacht", metric: `${km.best_k} Regime`,
      label: `Silhouette ${km.silhouette}`,
      take: "Gut getrennte Regime: Hoch-GPR = schwache, volatile Märkte." },
    { name: "Random Forest", type: "in-sample", metric: rf.A.test_auc.toFixed(2),
      label: "AUC (Richtung A) · in-sample",
      take: "Hohe AUC, aber in-sample — beschreibt nur die Trainingsdaten." },
    { name: "KNN", type: "out-of-sample", metric: knn.A.test_auc.toFixed(2),
      label: "AUC (Richtung A) · Holdout",
      take: "Sauber out-of-sample bewertet; Signal knapp über dem Zufall." },
  ];
  document.getElementById("syn-methods").innerHTML = methods.map(m =>
    `<div class="method-card">`
    + `<div class="m-name">${m.name}</div><div class="m-type">${m.type}</div>`
    + `<div class="m-metric">${m.metric}</div><div class="m-metric-label">${m.label}</div>`
    + `<div class="m-take">${m.take}</div></div>`
  ).join("");

  // Kennzahlen-Vergleichstabelle (alle 4, heterogene Metriken)
  document.getElementById("syn-table").innerHTML =
    "<tr><th>Verfahren</th><th>Typ</th><th>Kernfrage</th><th>Kennzahl</th><th>Aussage</th></tr>"
    + `<tr><td>Lineare Regression</td><td>deskriptiv</td><td>Reaktion auf Schock?</td>`
    + `<td>CAR_post ${fmtPct(evBest.car_post)} (${sig(evBest.p)})</td><td>klein, gerichtet</td></tr>`
    + `<tr><td>K-Means</td><td>unüberwacht</td><td>Welche Marktregime?</td>`
    + `<td>${km.best_k} Regime · Sil. ${km.silhouette}</td><td>klar getrennt</td></tr>`
    + `<tr><td>Random Forest</td><td>in-sample</td><td>Klassifizierbar?</td>`
    + `<td>AUC ${rf.A.test_auc.toFixed(3)} / ${rf.B.test_auc.toFixed(3)}</td><td>nur deskriptiv</td></tr>`
    + `<tr><td>KNN</td><td>out-of-sample</td><td>Vorhersagbar?</td>`
    + `<td>AUC ${knn.A.test_auc.toFixed(3)} / ${knn.B.test_auc.toFixed(3)}</td><td>schwach, aber echt</td></tr>`;

  // AUC-Vergleichsbalken (KNN out-of-sample vs RF in-sample)
  Plotly.react("syn-auc", [
    { x: ["Richtung A", "Richtung B"], y: [knn.A.test_auc, knn.B.test_auc], type: "bar", name: "KNN (o.o.s.)",
      marker: { color: COLORS.accent }, hovertemplate: "%{x}<br>AUC %{y:.3f}<extra>KNN</extra>" },
    { x: ["Richtung A", "Richtung B"], y: [rf.A.test_auc, rf.B.test_auc], type: "bar", name: "RF (in-sample)",
      marker: { color: COLORS.muted, pattern: { shape: "/" } }, hovertemplate: "%{x}<br>AUC %{y:.3f}<extra>RF</extra>" },
  ], L({
    barmode: "group",
    shapes: [{ type: "line", x0: -0.5, x1: 1.5, y0: 0.5, y1: 0.5, xref: "x", yref: "y",
      line: { color: COLORS.accent2, width: 1, dash: "dash" } }],
    yaxis: { title: "AUC", gridcolor: "#F0F0F0", range: [0.4, 1] },
    annotations: [{ x: 1.45, y: 0.5, xref: "x", yref: "y", text: "Zufall", showarrow: false,
      font: { size: 10, color: COLORS.accent2 }, yanchor: "bottom", xanchor: "right" }],
    legend: { orientation: "h", y: -0.15 },
  }), CONFIG);

  // Warum die Verfahren auseinanderliegen (ohne Rangfolge)
  document.getElementById("syn-verdict").innerHTML =
    `Die Verfahren sind <b>nicht direkt vergleichbar</b> — sie beantworten unterschiedliche Fragen mit unterschiedlichen `
    + `Metriken. <b>K-Means</b> ist unüberwacht (Silhouette misst Trennschärfe, keine Vorhersage). `
    + `<b>Random Forest</b> wurde <b>in-sample</b> ausgewertet: Die hohe AUC (${rf.A.test_auc.toFixed(2)}) beschreibt die `
    + `Trainingsdaten, nicht die Prognose — der Kontrast zur <b>out-of-sample</b> bewerteten KNN-AUC `
    + `(${knn.A.test_auc.toFixed(2)}) illustriert diese In-Sample-Falle. Die lineare Regression liefert die Effektgröße `
    + `(CAR), nicht die Klassifikationsgüte. Die Kennzahlen ergänzen sich also, statt sich zu einer Rangliste zu fügen.`;

  // ── ③ Datenbefunde (Auswirkungen, modellunabhängig) ────────────────
  document.getElementById("syn-data").innerHTML =
    `<b>Wie stark beeinflussen sich GPR und Aktien?</b> Der Zusammenhang ist <b>negativ, aber klein</b>: `
    + `GPR-THREAT korreliert mit den Aktienrenditen mit ${fmtNum(k.corr_threat, 3)}, GPR-ACT mit ${fmtNum(k.corr_act, 3)}. `
    + `Spürbar wird der Effekt vor allem in Stressphasen — in GPR-Spike-Monaten fällt die Ø Rendite auf `
    + `<b>${fmtPct(k.avg_return_spike)}</b> (vs. ${fmtPct(k.avg_return_normal)} sonst).<br><br>`
    + `<b>Was ist belastbar?</b> <b>Belastbar</b> sind die gleichzeitigen Muster: der Spike-Renditeeffekt und `
    + `<b>THREAT > ACT</b> (antizipierte Risiken bewegen Märkte stärker als realisierte) sind über den gesamten `
    + `Zeitraum konsistent. <b>Wenig belastbar</b> ist dagegen die vorlaufende Wirkung — wer wen vorhersagt, `
    + `liegt nahe am Zufall und taugt nicht als verlässliches Signal.`;

  const hiGpr = km.stats.reduce((a, b) => b.avg_gpr > a.avg_gpr ? b : a);
  document.getElementById("syn-regime").innerHTML =
    `Das Hoch-GPR-Regime (Regime ${hiGpr.cluster}, Ø GPR ${fmtNum(hiGpr.avg_gpr, 0)}) weist mit `
    + `<b>${(hiGpr.loss_share * 100).toFixed(0)} % Verlustmonaten</b> und ${(hiGpr.spike_share * 100).toFixed(0)} % GPR-Spikes `
    + `das ungünstigste Marktumfeld auf. Die gerichteten Effekte sind also vor allem ein Phänomen `
    + `erhöhter geopolitischer Anspannung — in ruhigen Phasen verschwinden sie weitgehend.`;

  // ── ④ Einordnung & Limitationen ───────────────────────────────────
  document.getElementById("syn-limits").innerHTML =
    `<b>Vorsicht bei der Interpretation.</b> Die out-of-sample-Vorhersagekraft liegt <b>nahe am Zufall</b> — `
    + `kein handelbares Signal. Random-Forest-Kennzahlen sind <b>in-sample</b> und keine Prognosegüte. `
    + `Alle Befunde belegen <b>Zusammenhänge, keine Kausalität</b>. Effekte sind teils statistisch signifikant, `
    + `ökonomisch jedoch klein.`;
}

/* =================== NAVIGATION + INIT =================== */
const RENDERERS = {
  overview: renderOverview, kmeans: renderKmeans, knn: renderKnn,
  rf: renderRf, events: renderEvents, synthesis: renderSynthesis,
};
const rendered = {};

function navigate(target) {
  document.querySelectorAll(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.target === target));
  document.querySelectorAll(".section").forEach(s => s.classList.toggle("active", s.id === target));
  if (!rendered[target]) { RENDERERS[target](); rendered[target] = true; }
  // Resize, da Plotly in versteckten Containern keine Größe kennt
  setTimeout(() => window.dispatchEvent(new Event("resize")), 30);
}

function bindToggle(id, onPick) {
  document.getElementById(id).addEventListener("click", e => {
    if (e.target.tagName !== "BUTTON") return;
    [...e.currentTarget.children].forEach(b => b.classList.toggle("active", b === e.target));
    onPick(e.target.dataset.v);
  });
}

function initToggles() {
  bindToggle("ov-gpr-toggle", v => drawOverviewTS(v));
  bindToggle("knn-fi-toggle", v => { knnFiDir = v; drawKnnFI(); });
  bindToggle("rf-fi-toggle", v => { rfFiDir = v; drawRfFI(); });
  bindToggle("ev-dir-toggle", v => drawEventDir(v));
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
