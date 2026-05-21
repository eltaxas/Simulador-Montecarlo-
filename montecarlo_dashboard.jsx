import { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart, BarChart, Bar, Cell } from "recharts";

// ─── Monte Carlo Engine ───────────────────────────────────────────────────────
function seedRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function runSimulation({ initialBank, stake, winRate, yieldPct, nPicks, nSims, evFilter }) {
  const yieldDecimal = yieldPct / 100;
  const evPerBet     = yieldDecimal * stake;
  const gMedia       = (evPerBet + (1 - winRate / 100) * stake) / (winRate / 100);
  const simulations  = [];
  const rng          = seedRandom(42);

  for (let i = 0; i < nSims; i++) {
    const path = [initialBank];
    let bank   = initialBank;
    for (let j = 0; j < nPicks; j++) {
      const r    = rng();
      const win  = r < winRate / 100;
      const gain = win
        ? Math.max(0.25 * stake, Math.min(3.5 * stake, gMedia + (rng() - 0.5) * 10))
        : -stake;
      bank += gain;
      path.push(Math.max(0, bank));
    }
    simulations.push(path);
  }

  // Percentiles en cada paso (submuestreo cada N picks para rendimiento)
  const step     = Math.max(1, Math.floor(nPicks / 200));
  const indices  = [];
  for (let i = 0; i <= nPicks; i += step) indices.push(i);

  const chartData = indices.map(idx => {
    const vals = simulations.map(s => s[idx]).sort((a, b) => a - b);
    const pct  = (p) => vals[Math.floor((p / 100) * (vals.length - 1))];
    return {
      picks: idx,
      p5:    Math.round(pct(5)),
      p25:   Math.round(pct(25)),
      p50:   Math.round(pct(50)),
      p75:   Math.round(pct(75)),
      p95:   Math.round(pct(95)),
      media: Math.round(vals.reduce((a, b) => a + b, 0) / vals.length),
    };
  });

  // Stats finales
  const finals      = simulations.map(s => s[s.length - 1]);
  const sortedFinals = [...finals].sort((a, b) => a - b);
  const pct         = (p) => sortedFinals[Math.floor((p / 100) * (sortedFinals.length - 1))];
  const mean        = finals.reduce((a, b) => a + b, 0) / finals.length;

  const drawdowns = simulations.map(path => {
    let maxDD = 0, peak = path[0];
    for (const v of path) {
      if (v > peak) peak = v;
      const dd = peak - v;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
  });
  const sortedDD = [...drawdowns].sort((a, b) => a - b);

  // Distribución del bankroll final para histograma
  const buckets = 40;
  const minF    = sortedFinals[0];
  const maxF    = sortedFinals[sortedFinals.length - 1];
  const bSize   = (maxF - minF) / buckets;
  const hist    = Array.from({ length: buckets }, (_, i) => ({
    range:  Math.round(minF + i * bSize),
    rangeK: `${Math.round((minF + i * bSize) / 1000)}k`,
    count:  0,
    isLoss: (minF + i * bSize) < initialBank,
  }));
  finals.forEach(f => {
    const idx = Math.min(buckets - 1, Math.floor((f - minF) / bSize));
    if (idx >= 0) hist[idx].count++;
  });

  return {
    chartData,
    hist,
    stats: {
      mean:       Math.round(mean),
      p5:         Math.round(pct(5)),
      p10:        Math.round(pct(10)),
      p25:        Math.round(pct(25)),
      p50:        Math.round(pct(50)),
      p75:        Math.round(pct(75)),
      p90:        Math.round(pct(90)),
      p95:        Math.round(pct(95)),
      probRuin:   (finals.filter(f => f <= 0).length / finals.length * 100).toFixed(2),
      probLoss:   (finals.filter(f => f < initialBank).length / finals.length * 100).toFixed(1),
      probDouble: (finals.filter(f => f >= initialBank * 2).length / finals.length * 100).toFixed(1),
      probTriple: (finals.filter(f => f >= initialBank * 3).length / finals.length * 100).toFixed(1),
      probX5:     (finals.filter(f => f >= initialBank * 5).length / finals.length * 100).toFixed(1),
      avgDD:      Math.round(drawdowns.reduce((a, b) => a + b, 0) / drawdowns.length),
      p90DD:      Math.round(sortedDD[Math.floor(0.9 * sortedDD.length)]),
      maxDD:      Math.round(sortedDD[sortedDD.length - 1]),
      evPerBet:   evPerBet.toFixed(3),
      gMedia:     gMedia.toFixed(2),
    }
  };
}

// ─── Formatters ──────────────────────────────────────────────────────────────
const fmtEuro = (v) => v >= 1000 ? `${(v/1000).toFixed(1)}k€` : `${v}€`;
const fmtFull = (v) => `${v?.toLocaleString("es-ES")}€`;

// ─── Tooltip personalizado ───────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label, initialBank }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1117", border: "1px solid #30363d",
      borderRadius: 8, padding: "10px 14px", fontSize: 12
    }}>
      <div style={{ color: "#8b949e", marginBottom: 6 }}>Pick #{label?.toLocaleString()}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{fmtFull(p.value)}</strong>
        </div>
      ))}
      {initialBank && (
        <div style={{ color: "#555", marginTop: 4, borderTop: "1px solid #222", paddingTop: 4 }}>
          Capital inicial: {fmtFull(initialBank)}
        </div>
      )}
    </div>
  );
};

// ─── Slider component ────────────────────────────────────────────────────────
const Slider = ({ label, value, min, max, step, onChange, format, color = "#58a6ff" }) => (
  <div style={{ marginBottom: 16 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
      <span style={{ color: "#8b949e", fontSize: 12, fontFamily: "monospace" }}>{label}</span>
      <span style={{ color, fontSize: 13, fontWeight: 700, fontFamily: "monospace" }}>
        {format ? format(value) : value}
      </span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(Number(e.target.value))}
      style={{ width: "100%", accentColor: color, cursor: "pointer" }} />
  </div>
);

// ─── Stat card ───────────────────────────────────────────────────────────────
const StatCard = ({ label, value, color = "#e6edf3", sub }) => (
  <div style={{
    background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
    padding: "12px 14px", textAlign: "center"
  }}>
    <div style={{ color: "#8b949e", fontSize: 10, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>
      {label}
    </div>
    <div style={{ color, fontSize: 18, fontWeight: 800, fontFamily: "monospace" }}>{value}</div>
    {sub && <div style={{ color: "#555", fontSize: 10, marginTop: 2 }}>{sub}</div>}
  </div>
);

// ─── Líneas visibles toggle ──────────────────────────────────────────────────
const LINE_CONFIG = [
  { key: "p5",    name: "P5 Pesimista",  color: "#f85149" },
  { key: "p25",   name: "P25",           color: "#e3b341" },
  { key: "p50",   name: "Mediana P50",   color: "#58a6ff" },
  { key: "media", name: "Media",         color: "#ffffff" },
  { key: "p75",   name: "P75",           color: "#56d364" },
  { key: "p95",   name: "P95 Optimista", color: "#3fb950" },
];

// ─── App principal ────────────────────────────────────────────────────────────
export default function MontecarloApp() {
  const [params, setParams] = useState({
    initialBank: 1500,
    stake:       15,
    winRate:     62,
    yieldPct:    6.5,
    nPicks:      6000,
    nSims:       500,
  });
  const [result, setResult]     = useState(null);
  const [running, setRunning]   = useState(false);
  const [visLines, setVisLines] = useState({ p5: true, p25: true, p50: true, media: true, p75: true, p95: true });
  const [activeTab, setActiveTab] = useState("trajectories");

  const set = (k) => (v) => setParams(prev => ({ ...prev, [k]: v }));

  const run = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const r = runSimulation(params);
      setResult(r);
      setRunning(false);
    }, 50);
  }, [params]);

  useEffect(() => { run(); }, []);

  const toggleLine = (key) => setVisLines(prev => ({ ...prev, [key]: !prev[key] }));

  const yAxisFmt = (v) => fmtEuro(v);

  return (
    <div style={{
      minHeight: "100vh", background: "#0d1117", color: "#e6edf3",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "24px 28px"
    }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: running ? "#f85149" : "#3fb950",
            boxShadow: running ? "0 0 8px #f85149" : "0 0 8px #3fb950"
          }} />
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, letterSpacing: -0.5 }}>
            MONTECARLO SIMULATOR
          </h1>
          <span style={{
            background: "#21262d", border: "1px solid #30363d",
            borderRadius: 4, padding: "2px 8px", fontSize: 10, color: "#8b949e"
          }}>
            PROYECTO 500 BALAS
          </span>
        </div>
        <p style={{ margin: 0, color: "#8b949e", fontSize: 12 }}>
          Sistema de value betting · Filtro EV≥3.5% · Simulación interactiva en tiempo real
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20 }}>

        {/* Panel de controles */}
        <div>
          <div style={{
            background: "#161b22", border: "1px solid #30363d",
            borderRadius: 10, padding: "18px 16px", marginBottom: 16
          }}>
            <div style={{ color: "#58a6ff", fontSize: 11, fontWeight: 700, marginBottom: 16,
                          textTransform: "uppercase", letterSpacing: 2 }}>
              ⚙ Parámetros
            </div>

            <Slider label="Bankroll inicial" value={params.initialBank}
              min={500} max={10000} step={100} onChange={set("initialBank")}
              format={v => `${v.toLocaleString("es-ES")}€`} color="#58a6ff" />
            <Slider label="Stake base" value={params.stake}
              min={5} max={100} step={5} onChange={set("stake")}
              format={v => `${v}€`} color="#58a6ff" />
            <Slider label="Win Rate" value={params.winRate}
              min={45} max={75} step={0.5} onChange={set("winRate")}
              format={v => `${v}%`} color="#e3b341" />
            <Slider label="Yield picks filtrados" value={params.yieldPct}
              min={1} max={15} step={0.5} onChange={set("yieldPct")}
              format={v => `${v}%`} color="#3fb950" />
            <Slider label="Nº de picks" value={params.nPicks}
              min={500} max={10000} step={500} onChange={set("nPicks")}
              format={v => v.toLocaleString("es-ES")} color="#bc8cff" />
            <Slider label="Simulaciones" value={params.nSims}
              min={100} max={1000} step={100} onChange={set("nSims")}
              format={v => v.toLocaleString("es-ES")} color="#8b949e" />

            <button onClick={run} disabled={running} style={{
              width: "100%", padding: "10px 0", marginTop: 8,
              background: running ? "#21262d" : "#238636",
              border: "1px solid " + (running ? "#30363d" : "#2ea043"),
              borderRadius: 6, color: running ? "#8b949e" : "#fff",
              fontSize: 13, fontWeight: 700, cursor: running ? "not-allowed" : "pointer",
              fontFamily: "monospace", letterSpacing: 1,
              transition: "all 0.2s"
            }}>
              {running ? "⟳ CALCULANDO..." : "▶ SIMULAR"}
            </button>
          </div>

          {/* Toggle líneas */}
          <div style={{
            background: "#161b22", border: "1px solid #30363d",
            borderRadius: 10, padding: "14px 16px"
          }}>
            <div style={{ color: "#58a6ff", fontSize: 11, fontWeight: 700, marginBottom: 12,
                          textTransform: "uppercase", letterSpacing: 2 }}>
              ◉ Líneas visibles
            </div>
            {LINE_CONFIG.map(l => (
              <div key={l.key} onClick={() => toggleLine(l.key)}
                style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "6px 8px", borderRadius: 6, cursor: "pointer",
                  marginBottom: 4, opacity: visLines[l.key] ? 1 : 0.35,
                  background: visLines[l.key] ? "#0d1117" : "transparent",
                  border: "1px solid " + (visLines[l.key] ? "#30363d" : "transparent"),
                  transition: "all 0.15s"
                }}>
                <div style={{ width: 16, height: 3, background: l.color, borderRadius: 2 }} />
                <span style={{ color: l.color, fontSize: 11, fontWeight: 600 }}>{l.name}</span>
                {result && (
                  <span style={{ marginLeft: "auto", color: "#8b949e", fontSize: 10 }}>
                    {fmtEuro(result.chartData[result.chartData.length - 1][l.key])}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Panel principal */}
        <div>
          {/* Stats en la parte superior */}
          {result && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8, marginBottom: 16 }}>
              <StatCard label="Media final"   value={fmtEuro(result.stats.mean)}    color="#ffffff" />
              <StatCard label="P5 pesimista"  value={fmtEuro(result.stats.p5)}      color="#f85149" />
              <StatCard label="Mediana"       value={fmtEuro(result.stats.p50)}     color="#58a6ff" />
              <StatCard label="P95 optimista" value={fmtEuro(result.stats.p95)}     color="#3fb950" />
              <StatCard label="Riesgo ruina"  value={`${result.stats.probRuin}%`}   color={result.stats.probRuin > 1 ? "#f85149" : "#3fb950"} />
              <StatCard label="Prob doblar"   value={`${result.stats.probDouble}%`} color="#56d364" />
            </div>
          )}

          {/* Tabs */}
          <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
            {[
              { key: "trajectories", label: "📈 Trayectorias" },
              { key: "distribution", label: "📊 Distribución final" },
              { key: "milestones",   label: "🏁 Hitos" },
              { key: "risk",         label: "⚠️ Riesgo" },
            ].map(t => (
              <button key={t.key} onClick={() => setActiveTab(t.key)} style={{
                padding: "7px 14px", borderRadius: 6, fontSize: 11, fontWeight: 600,
                cursor: "pointer", fontFamily: "monospace",
                background: activeTab === t.key ? "#21262d" : "transparent",
                border: "1px solid " + (activeTab === t.key ? "#58a6ff" : "#30363d"),
                color: activeTab === t.key ? "#58a6ff" : "#8b949e",
                transition: "all 0.15s"
              }}>{t.label}</button>
            ))}
          </div>

          {/* Gráficos */}
          <div style={{
            background: "#161b22", border: "1px solid #30363d",
            borderRadius: 10, padding: "16px"
          }}>
            {!result && (
              <div style={{ height: 380, display: "flex", alignItems: "center", justifyContent: "center", color: "#8b949e" }}>
                Ejecuta la simulación para ver resultados
              </div>
            )}

            {result && activeTab === "trajectories" && (
              <>
                <div style={{ color: "#8b949e", fontSize: 11, marginBottom: 12 }}>
                  Percentiles del bankroll a lo largo de {params.nPicks.toLocaleString()} apuestas · {params.nSims} simulaciones
                </div>
                <ResponsiveContainer width="100%" height={380}>
                  <LineChart data={result.chartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                    <XAxis dataKey="picks" stroke="#555"
                      tickFormatter={v => v >= 1000 ? `${v/1000}k` : v}
                      tick={{ fontSize: 10, fill: "#8b949e" }} />
                    <YAxis stroke="#555" tickFormatter={yAxisFmt}
                      tick={{ fontSize: 10, fill: "#8b949e" }} />
                    <Tooltip content={<CustomTooltip initialBank={params.initialBank} />} />
                    <ReferenceLine y={params.initialBank} stroke="#555" strokeDasharray="4 4"
                      label={{ value: "Capital inicial", position: "right", fill: "#555", fontSize: 10 }} />
                    <ReferenceLine y={0} stroke="#f85149" strokeOpacity={0.4} />
                    {LINE_CONFIG.map(l => visLines[l.key] && (
                      <Line key={l.key} type="monotone" dataKey={l.key} name={l.name}
                        stroke={l.color} strokeWidth={l.key === "media" ? 2.5 : 1.5}
                        dot={false} activeDot={{ r: 4 }}
                        strokeDasharray={l.key === "p5" || l.key === "p95" ? "5 3" : undefined} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}

            {result && activeTab === "distribution" && (
              <>
                <div style={{ color: "#8b949e", fontSize: 11, marginBottom: 12 }}>
                  Distribución de bankrolls al final de {params.nPicks.toLocaleString()} apuestas
                </div>
                <ResponsiveContainer width="100%" height={380}>
                  <BarChart data={result.hist} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                    <XAxis dataKey="rangeK" stroke="#555" tick={{ fontSize: 9, fill: "#8b949e" }}
                      label={{ value: "Bankroll final (€)", position: "insideBottom", offset: -10, fill: "#8b949e", fontSize: 11 }} />
                    <YAxis stroke="#555" tick={{ fontSize: 10, fill: "#8b949e" }} />
                    <Tooltip formatter={(v, n) => [`${v} simulaciones`, ""]}
                      contentStyle={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8 }} />
                    <ReferenceLine x={Math.round(params.initialBank/1000) + "k"} stroke="#58a6ff"
                      strokeDasharray="4 4" />
                    <Bar dataKey="count" name="Simulaciones" radius={[3, 3, 0, 0]}>
                      {result.hist.map((entry, i) => (
                        <Cell key={i} fill={entry.isLoss ? "#f85149" : entry.range >= params.initialBank * 3 ? "#3fb950" : "#58a6ff"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 8 }}>
                  {[["#f85149", "Pérdida"], ["#58a6ff", "Ganancia"], ["#3fb950", "x3 o más"]].map(([c, l]) => (
                    <div key={l} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#8b949e" }}>
                      <div style={{ width: 12, height: 12, background: c, borderRadius: 2 }} /> {l}
                    </div>
                  ))}
                </div>
              </>
            )}

            {result && activeTab === "milestones" && (
              <>
                <div style={{ color: "#8b949e", fontSize: 11, marginBottom: 16 }}>
                  Evolución del bankroll en hitos clave
                </div>
                {[500, 1000, 2000, 3000, Math.round(params.nPicks * 0.75), params.nPicks].filter(h => h <= params.nPicks).map(h => {
                  const idx  = result.chartData.findIndex(d => d.picks >= h);
                  const d    = result.chartData[idx >= 0 ? idx : result.chartData.length - 1];
                  const pct  = (d.media - params.initialBank) / params.initialBank * 100;
                  return (
                    <div key={h} style={{
                      display: "grid", gridTemplateColumns: "80px 1fr 1fr 1fr 1fr 80px",
                      gap: 8, padding: "10px 12px", marginBottom: 6, borderRadius: 8,
                      background: "#0d1117", border: "1px solid #21262d",
                      alignItems: "center"
                    }}>
                      <div style={{ color: "#8b949e", fontSize: 11, fontWeight: 700 }}>
                        {h >= 1000 ? `${h/1000}k` : h} picks
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ color: "#f85149", fontSize: 12, fontWeight: 700 }}>{fmtFull(d.p5)}</div>
                        <div style={{ color: "#555", fontSize: 9 }}>P5</div>
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ color: "#58a6ff", fontSize: 12, fontWeight: 700 }}>{fmtFull(d.p50)}</div>
                        <div style={{ color: "#555", fontSize: 9 }}>Mediana</div>
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ color: "#fff", fontSize: 12, fontWeight: 700 }}>{fmtFull(d.media)}</div>
                        <div style={{ color: "#555", fontSize: 9 }}>Media</div>
                      </div>
                      <div style={{ textAlign: "center" }}>
                        <div style={{ color: "#3fb950", fontSize: 12, fontWeight: 700 }}>{fmtFull(d.p95)}</div>
                        <div style={{ color: "#555", fontSize: 9 }}>P95</div>
                      </div>
                      <div style={{
                        textAlign: "right", fontSize: 12, fontWeight: 700,
                        color: pct >= 0 ? "#3fb950" : "#f85149"
                      }}>
                        {pct >= 0 ? "+" : ""}{pct.toFixed(0)}%
                      </div>
                    </div>
                  );
                })}
              </>
            )}

            {result && activeTab === "risk" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                {/* Probabilidades */}
                <div>
                  <div style={{ color: "#3fb950", fontSize: 11, fontWeight: 700, marginBottom: 12,
                                textTransform: "uppercase", letterSpacing: 1 }}>
                    Probabilidades de éxito
                  </div>
                  {[
                    { label: "Prob. terminar en pérdida", value: `${result.stats.probLoss}%`,   color: result.stats.probLoss > 5 ? "#f85149" : "#3fb950" },
                    { label: "Riesgo de ruina total",      value: `${result.stats.probRuin}%`,   color: result.stats.probRuin > 0 ? "#f85149" : "#3fb950" },
                    { label: "Prob. doblar (×2)",          value: `${result.stats.probDouble}%`, color: "#56d364" },
                    { label: "Prob. triplicar (×3)",       value: `${result.stats.probTriple}%`, color: "#3fb950" },
                    { label: "Prob. quintuplicar (×5)",    value: `${result.stats.probX5}%`,     color: "#3fb950" },
                  ].map(row => (
                    <div key={row.label} style={{
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                      padding: "9px 12px", marginBottom: 6, borderRadius: 6,
                      background: "#0d1117", border: "1px solid #21262d"
                    }}>
                      <span style={{ color: "#8b949e", fontSize: 12 }}>{row.label}</span>
                      <span style={{ color: row.color, fontWeight: 700, fontSize: 14 }}>{row.value}</span>
                    </div>
                  ))}
                </div>

                {/* Drawdown */}
                <div>
                  <div style={{ color: "#e3b341", fontSize: 11, fontWeight: 700, marginBottom: 12,
                                textTransform: "uppercase", letterSpacing: 1 }}>
                    Análisis de drawdown
                  </div>
                  {[
                    { label: "Drawdown medio esperado",      value: fmtFull(result.stats.avgDD), pct: `${(result.stats.avgDD/params.initialBank*100).toFixed(0)}%`, color: "#8b949e" },
                    { label: "Drawdown típico malo (P90)",   value: fmtFull(result.stats.p90DD), pct: `${(result.stats.p90DD/params.initialBank*100).toFixed(0)}%`, color: "#e3b341" },
                    { label: "Peor drawdown simulado",       value: fmtFull(result.stats.maxDD), pct: `${(result.stats.maxDD/params.initialBank*100).toFixed(0)}%`, color: "#f85149" },
                    { label: "Bankroll mínimo recomendado",  value: fmtFull(result.stats.maxDD * 1.5), pct: "buffer ×1.5", color: "#58a6ff" },
                  ].map(row => (
                    <div key={row.label} style={{
                      padding: "9px 12px", marginBottom: 6, borderRadius: 6,
                      background: "#0d1117", border: "1px solid #21262d"
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span style={{ color: "#8b949e", fontSize: 12 }}>{row.label}</span>
                        <span style={{ color: row.color, fontWeight: 700, fontSize: 13 }}>{row.value}</span>
                      </div>
                      <div style={{ color: "#555", fontSize: 10, textAlign: "right" }}>{row.pct} del bankroll</div>
                    </div>
                  ))}

                  <div style={{
                    marginTop: 12, padding: "10px 12px", borderRadius: 6,
                    background: "#0d1117", border: "1px solid #e3b34140"
                  }}>
                    <div style={{ color: "#e3b341", fontSize: 11, fontWeight: 700, marginBottom: 6 }}>
                      ⚠ Gestión de bankroll
                    </div>
                    <div style={{ color: "#8b949e", fontSize: 11, lineHeight: 1.6 }}>
                      Stake del {((params.stake / params.initialBank) * 100).toFixed(1)}% del bankroll por apuesta.
                      {(params.stake / params.initialBank) > 0.03
                        ? " Considera reducir el stake — supera el 3% recomendado."
                        : " Dentro del rango recomendado (≤3%)."}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Info EV */}
          {result && (
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginTop: 12
            }}>
              {[
                { label: "EV por apuesta", value: `+${result.stats.evPerBet}€` },
                { label: "Stake / Bankroll", value: `${((params.stake/params.initialBank)*100).toFixed(1)}%` },
                { label: "Total apostado", value: `${(params.nPicks*params.stake).toLocaleString("es-ES")}€` },
                { label: "Beneficio esperado", value: `+${((result.stats.mean - params.initialBank)).toLocaleString("es-ES")}€` },
              ].map(s => (
                <div key={s.label} style={{
                  background: "#161b22", border: "1px solid #30363d",
                  borderRadius: 8, padding: "8px 12px",
                  display: "flex", justifyContent: "space-between", alignItems: "center"
                }}>
                  <span style={{ color: "#8b949e", fontSize: 10 }}>{s.label}</span>
                  <span style={{ color: "#58a6ff", fontWeight: 700, fontSize: 12 }}>{s.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
