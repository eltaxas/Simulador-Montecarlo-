import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  CONFIG PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Montecarlo · Proyecto 500 Balas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo oscuro */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* Métricas */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 11px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px !important; font-weight: 800 !important; }

    /* Headers */
    h1, h2, h3 { color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace !important; }

    /* Sliders */
    .stSlider > div > div > div { background: #58a6ff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #8b949e !important; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }

    /* Botón */
    .stButton button {
        background: #238636 !important;
        border: 1px solid #2ea043 !important;
        color: white !important;
        font-weight: 700 !important;
        font-family: monospace !important;
        width: 100%;
        border-radius: 6px;
    }
    .stButton button:hover { background: #2ea043 !important; }

    /* Dataframe */
    .stDataFrame { background: #161b22; }

    /* Info boxes */
    .stat-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Divider */
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MOTOR MONTECARLO
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_simulation(initial_bank, stake, win_rate_pct, yield_pct, n_picks, n_sims, seed=42):
    """Simulación Montecarlo con bootstrapping paramétrico."""
    np.random.seed(seed)

    win_rate    = win_rate_pct / 100
    yield_dec   = yield_pct / 100
    ev_per_bet  = yield_dec * stake
    g_media     = (ev_per_bet + (1 - win_rate) * stake) / win_rate
    g_media     = max(g_media, stake * 0.1)

    # Matriz de simulaciones
    results = np.zeros((n_sims, n_picks + 1))
    results[:, 0] = initial_bank

    for i in range(n_sims):
        wins      = np.random.random(n_picks) < win_rate
        ganancias = np.random.normal(g_media, stake * 0.35, n_picks)
        ganancias = np.clip(ganancias, stake * 0.15, stake * 3.5)
        pl        = np.where(wins, ganancias, -stake)
        results[i, 1:] = initial_bank + np.cumsum(pl)
        results[i]     = np.maximum(0, results[i])

    # Percentiles submuestreados para rendimiento
    step    = max(1, n_picks // 300)
    indices = np.arange(0, n_picks + 1, step)

    pcts = {
        "p5":    np.percentile(results[:, indices], 5,  axis=0),
        "p10":   np.percentile(results[:, indices], 10, axis=0),
        "p25":   np.percentile(results[:, indices], 25, axis=0),
        "p50":   np.percentile(results[:, indices], 50, axis=0),
        "p75":   np.percentile(results[:, indices], 75, axis=0),
        "p90":   np.percentile(results[:, indices], 90, axis=0),
        "p95":   np.percentile(results[:, indices], 95, axis=0),
        "media": np.mean(results[:, indices], axis=0),
    }

    # Stats finales
    finals     = results[:, -1]
    sorted_f   = np.sort(finals)
    drawdowns  = np.array([
        np.max(np.maximum.accumulate(results[i]) - results[i])
        for i in range(n_sims)
    ])

    # Histograma
    counts, bin_edges = np.histogram(finals, bins=50)

    stats = {
        "mean":        np.mean(finals),
        "p5":          np.percentile(finals, 5),
        "p10":         np.percentile(finals, 10),
        "p25":         np.percentile(finals, 25),
        "p50":         np.percentile(finals, 50),
        "p75":         np.percentile(finals, 75),
        "p90":         np.percentile(finals, 90),
        "p95":         np.percentile(finals, 95),
        "prob_ruin":   np.mean(np.any(results <= 0, axis=1)) * 100,
        "prob_loss":   np.mean(finals < initial_bank) * 100,
        "prob_double": np.mean(finals >= initial_bank * 2) * 100,
        "prob_triple": np.mean(finals >= initial_bank * 3) * 100,
        "prob_x5":     np.mean(finals >= initial_bank * 5) * 100,
        "prob_x10":    np.mean(finals >= initial_bank * 10) * 100,
        "avg_dd":      np.mean(drawdowns),
        "p90_dd":      np.percentile(drawdowns, 90),
        "max_dd":      np.max(drawdowns),
        "ev_per_bet":  ev_per_bet,
        "g_media":     g_media,
        "counts":      counts,
        "bin_edges":   bin_edges,
        "indices":     indices,
    }

    return pcts, stats, results


# ─────────────────────────────────────────────
#  SIDEBAR — PARÁMETROS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parámetros del sistema")
    st.markdown("---")

    initial_bank = st.slider("💰 Bankroll inicial (€)", 200, 20000, 1500, 100)
    stake        = st.slider("🎯 Stake base (€)", 1, 200, 15, 1)
    win_rate_pct = st.slider("✅ Win Rate (%)", 40.0, 80.0, 62.0, 0.5)
    yield_pct    = st.slider("📈 Yield picks filtrados (%)", 1.0, 20.0, 6.5, 0.5)
    n_picks      = st.slider("🎲 Número de picks", 500, 15000, 6000, 500)
    n_sims       = st.slider("🔄 Simulaciones", 100, 2000, 500, 100)

    st.markdown("---")

    # Info calculada
    ev_per_bet   = (yield_pct / 100) * stake
    stake_pct    = stake / initial_bank * 100
    total_staked = n_picks * stake

    st.markdown(f"""
    **📊 Métricas calculadas**
    - EV por apuesta: `+{ev_per_bet:.3f}€`
    - Stake/Bankroll: `{stake_pct:.1f}%`
    - Total apostado: `{total_staked:,.0f}€`
    - Filtro: `EV ≥ 3.5%`
    """)

    if stake_pct > 3:
        st.warning(f"⚠️ Stake al {stake_pct:.1f}% — considera reducirlo al ≤3%")

    st.markdown("---")
    simular = st.button("▶ SIMULAR", type="primary")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 📈 MONTECARLO SIMULATOR")
    st.markdown(f"**Proyecto 500 Balas** · Bot + Filtro EV≥3.5% · {n_sims} trayectorias · {n_picks:,} picks")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    status = "🟢 Listo" if not simular else "🔴 Calculando..."
    st.markdown(f"**Estado:** {status}")

st.markdown("---")


# ─────────────────────────────────────────────
#  EJECUTAR SIMULACIÓN
# ─────────────────────────────────────────────

with st.spinner("⟳ Ejecutando simulación..."):
    pcts, stats, results = run_simulation(
        initial_bank, stake, win_rate_pct, yield_pct, n_picks, n_sims
    )

indices = stats["indices"]

# ─────────────────────────────────────────────
#  MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Media final",     f"{stats['mean']:,.0f}€",   f"+{stats['mean']-initial_bank:,.0f}€")
c2.metric("P5 pesimista",    f"{stats['p5']:,.0f}€",     f"+{stats['p5']-initial_bank:,.0f}€")
c3.metric("Mediana",         f"{stats['p50']:,.0f}€",    f"+{stats['p50']-initial_bank:,.0f}€")
c4.metric("P95 optimista",   f"{stats['p95']:,.0f}€",    f"+{stats['p95']-initial_bank:,.0f}€")
c5.metric("Riesgo ruina",    f"{stats['prob_ruin']:.2f}%")
c6.metric("Prob. doblar",    f"{stats['prob_double']:.1f}%")

st.markdown("---")


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trayectorias",
    "📊 Distribución final",
    "🏁 Hitos",
    "⚠️ Riesgo & Drawdown"
])


# ── TAB 1: TRAYECTORIAS ──────────────────────────────────────────────────────

with tab1:
    st.markdown("#### Percentiles del bankroll a lo largo de las apuestas")

    # Selector de líneas
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        show_p5  = st.checkbox("P5 Pesimista",  value=True)
        show_p25 = st.checkbox("P25",           value=True)
    with col_l2:
        show_p50  = st.checkbox("Mediana P50",  value=True)
        show_mean = st.checkbox("Media",        value=True)
    with col_l3:
        show_p75 = st.checkbox("P75",           value=True)
        show_p95 = st.checkbox("P95 Optimista", value=True)

    # N trayectorias individuales
    n_traj = st.slider("Trayectorias individuales visibles", 0, min(50, n_sims), 20)

    fig = go.Figure()

    # Área entre P25 y P75
    fig.add_trace(go.Scatter(
        x=np.concatenate([indices, indices[::-1]]),
        y=np.concatenate([pcts["p75"], pcts["p25"][::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango P25–P75", showlegend=True
    ))

    # Área entre P5 y P95
    fig.add_trace(go.Scatter(
        x=np.concatenate([indices, indices[::-1]]),
        y=np.concatenate([pcts["p95"], pcts["p5"][::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango P5–P95", showlegend=True
    ))

    # Trayectorias individuales (submuestra aleatoria)
    if n_traj > 0:
        sampled = np.random.choice(n_sims, min(n_traj, n_sims), replace=False)
        for i in sampled:
            traj = results[i, indices]
            color = "rgba(63,185,80,0.15)" if traj[-1] >= initial_bank * 2 else "rgba(88,166,255,0.12)"
            fig.add_trace(go.Scatter(
                x=indices, y=traj,
                mode="lines", line=dict(width=0.5, color=color),
                showlegend=False, hoverinfo="skip"
            ))

    # Líneas percentiles
    lineas = [
        ("p5",    "P5 Pesimista",   "#f85149", show_p5,  "dash"),
        ("p25",   "P25",            "#e3b341", show_p25, "dot"),
        ("p50",   "Mediana P50",    "#58a6ff", show_p50, "solid"),
        ("media", "Media",          "#ffffff", show_mean,"solid"),
        ("p75",   "P75",            "#56d364", show_p75, "dot"),
        ("p95",   "P95 Optimista",  "#3fb950", show_p95, "dash"),
    ]
    for key, name, color, visible, dash in lineas:
        if visible:
            width = 2.5 if key == "media" else 1.8
            fig.add_trace(go.Scatter(
                x=indices, y=pcts[key],
                mode="lines", name=name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=f"<b>{name}</b><br>Pick: %{{x}}<br>Bankroll: %{{y:,.0f}}€<extra></extra>"
            ))

    # Línea capital inicial
    fig.add_hline(y=initial_bank, line_dash="dot", line_color="#555",
                  annotation_text=f"Capital inicial: {initial_bank:,}€",
                  annotation_position="right", annotation_font_color="#555")

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="monospace"),
        xaxis=dict(title="Número de apuestas", gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(title="Bankroll (€)", gridcolor="#21262d", color="#8b949e",
                   tickformat=",.0f"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        hovermode="x unified",
        height=480,
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: DISTRIBUCIÓN FINAL ────────────────────────────────────────────────

with tab2:
    st.markdown("#### Distribución del bankroll final")

    counts    = stats["counts"]
    bin_edges = stats["bin_edges"]
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2

    colors = []
    for mid in bin_mids:
        if mid < initial_bank:
            colors.append("#f85149")
        elif mid >= initial_bank * 3:
            colors.append("#3fb950")
        else:
            colors.append("#58a6ff")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bin_mids, y=counts,
        marker_color=colors,
        name="Simulaciones",
        hovertemplate="Bankroll: %{x:,.0f}€<br>Frecuencia: %{y}<extra></extra>"
    ))

    # Líneas verticales de referencia
    for val, name, color in [
        (stats["p5"],    "P5",          "#f85149"),
        (stats["p50"],   "Mediana",     "#58a6ff"),
        (stats["mean"],  "Media",       "#ffffff"),
        (stats["p95"],   "P95",         "#3fb950"),
        (initial_bank,   "Inicial",     "#888888"),
    ]:
        fig2.add_vline(x=val, line_color=color, line_dash="dash", line_width=1.5,
                       annotation_text=f"{name}: {val:,.0f}€",
                       annotation_font_color=color, annotation_font_size=10)

    fig2.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="monospace"),
        xaxis=dict(title="Bankroll final (€)", gridcolor="#21262d", color="#8b949e", tickformat=",.0f"),
        yaxis=dict(title="Número de simulaciones", gridcolor="#21262d", color="#8b949e"),
        height=420,
        margin=dict(t=20, b=40),
        bargap=0.05,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Leyenda de colores
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("🔴 **Pérdida** — bankroll < inicial")
    col_b.markdown("🔵 **Ganancia** — hasta ×3")
    col_c.markdown("🟢 **Excelente** — ×3 o más")


# ── TAB 3: HITOS ─────────────────────────────────────────────────────────────

with tab3:
    st.markdown("#### Evolución del bankroll en hitos clave")

    hitos = [h for h in [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000] if h <= n_picks]
    if n_picks not in hitos:
        hitos.append(n_picks)

    tabla = []
    for h in hitos:
        idx_pos = np.searchsorted(indices, h)
        idx_pos = min(idx_pos, len(indices) - 1)
        p5v     = int(pcts["p5"][idx_pos])
        p25v    = int(pcts["p25"][idx_pos])
        p50v    = int(pcts["p50"][idx_pos])
        p75v    = int(pcts["p75"][idx_pos])
        p95v    = int(pcts["p95"][idx_pos])
        meanv   = int(pcts["media"][idx_pos])
        pct_ret = (meanv - initial_bank) / initial_bank * 100
        tabla.append({
            "Picks": f"{h:,}",
            "P5 😰": f"{p5v:,}€",
            "P25": f"{p25v:,}€",
            "Mediana 📊": f"{p50v:,}€",
            "Media ⚖️": f"{meanv:,}€",
            "P75": f"{p75v:,}€",
            "P95 🚀": f"{p95v:,}€",
            "Retorno medio": f"+{pct_ret:.0f}%" if pct_ret >= 0 else f"{pct_ret:.0f}%",
        })

    import pandas as pd
    df = pd.DataFrame(tabla)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Gráfico de barras agrupado por hitos
    st.markdown("#### Comparativa visual por hitos")
    hitos_plot = [h for h in [500, 1000, 2000, 3000, n_picks] if h <= n_picks]
    x_labels   = [f"{h:,}" for h in hitos_plot]

    fig3 = go.Figure()
    capas = [("P5", "#f85149"), ("Mediana", "#58a6ff"), ("Media", "#ffffff"), ("P95", "#3fb950")]
    for label, color in capas:
        key = {"P5": "p5", "Mediana": "p50", "Media": "media", "P95": "p95"}[label]
        vals = []
        for h in hitos_plot:
            ip = np.searchsorted(indices, h)
            ip = min(ip, len(indices)-1)
            vals.append(int(pcts[key][ip]))
        fig3.add_trace(go.Bar(name=label, x=x_labels, y=vals, marker_color=color))

    fig3.add_hline(y=initial_bank, line_dash="dot", line_color="#555",
                   annotation_text=f"Capital inicial: {initial_bank:,}€")
    fig3.update_layout(
        barmode="group",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="monospace"),
        xaxis=dict(title="Número de picks", gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(title="Bankroll (€)", gridcolor="#21262d", color="#8b949e", tickformat=",.0f"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        height=380, margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── TAB 4: RIESGO ────────────────────────────────────────────────────────────

with tab4:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### 🎯 Probabilidades de éxito")
        probs = [
            ("Prob. terminar en pérdida",  f"{stats['prob_loss']:.1f}%",   "🔴" if stats["prob_loss"] > 5  else "🟢"),
            ("Riesgo de ruina total",       f"{stats['prob_ruin']:.2f}%",   "🔴" if stats["prob_ruin"] > 0  else "🟢"),
            ("Prob. doblar bankroll (×2)",  f"{stats['prob_double']:.1f}%", "🟢"),
            ("Prob. triplicar (×3)",        f"{stats['prob_triple']:.1f}%", "🟢"),
            ("Prob. quintuplicar (×5)",     f"{stats['prob_x5']:.1f}%",     "🟢"),
            ("Prob. ×10",                   f"{stats['prob_x10']:.1f}%",    "🟢"),
        ]
        for label, valor, icon in probs:
            st.markdown(f"""
            <div class="stat-box">
                <span style="color:#8b949e;font-size:13px">{icon} {label}</span>
                <span style="color:#e6edf3;font-weight:800;font-size:16px;font-family:monospace">{valor}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_r2:
        st.markdown("#### ⚠️ Análisis de Drawdown")
        dds = [
            ("Drawdown medio esperado",     f"{stats['avg_dd']:,.0f}€",
             f"{stats['avg_dd']/initial_bank*100:.0f}% del bankroll", "#8b949e"),
            ("Drawdown típico malo (P90)",  f"{stats['p90_dd']:,.0f}€",
             f"{stats['p90_dd']/initial_bank*100:.0f}% del bankroll", "#e3b341"),
            ("Peor drawdown simulado",      f"{stats['max_dd']:,.0f}€",
             f"{stats['max_dd']/initial_bank*100:.0f}% del bankroll", "#f85149"),
            ("Bankroll mínimo recomendado", f"{stats['max_dd']*1.5:,.0f}€",
             "buffer ×1.5 sobre el máximo", "#58a6ff"),
        ]
        for label, valor, sub, color in dds:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                        padding:12px 16px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="color:#8b949e;font-size:12px">{label}</span>
                    <span style="color:{color};font-weight:800;font-size:16px;font-family:monospace">{valor}</span>
                </div>
                <div style="color:#555;font-size:11px;text-align:right">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        # Aviso gestión bankroll
        stake_pct_val = stake / initial_bank * 100
        if stake_pct_val > 3:
            st.warning(f"⚠️ Stake al **{stake_pct_val:.1f}%** del bankroll — supera el 3% recomendado para value betting.")
        else:
            st.success(f"✅ Stake al **{stake_pct_val:.1f}%** del bankroll — dentro del rango óptimo (≤3%).")

    # Drawdown chart
    st.markdown("#### Distribución de drawdowns máximos")
    dd_vals = [
        np.max(np.maximum.accumulate(results[i]) - results[i])
        for i in range(min(n_sims, 500))
    ]
    dd_counts, dd_edges = np.histogram(dd_vals, bins=40)
    dd_mids = (dd_edges[:-1] + dd_edges[1:]) / 2

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=dd_mids, y=dd_counts,
        marker_color=[
            "#3fb950" if m < initial_bank * 0.2
            else "#e3b341" if m < initial_bank * 0.5
            else "#f85149"
            for m in dd_mids
        ],
        hovertemplate="Drawdown: %{x:,.0f}€<br>Freq: %{y}<extra></extra>"
    ))
    fig4.add_vline(x=stats["avg_dd"], line_color="#ffffff", line_dash="dash",
                   annotation_text="Media", annotation_font_color="#ffffff")
    fig4.add_vline(x=stats["p90_dd"], line_color="#e3b341", line_dash="dash",
                   annotation_text="P90", annotation_font_color="#e3b341")
    fig4.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="monospace"),
        xaxis=dict(title="Drawdown máximo (€)", gridcolor="#21262d", color="#8b949e", tickformat=",.0f"),
        yaxis=dict(title="Frecuencia", gridcolor="#21262d", color="#8b949e"),
        height=300, margin=dict(t=20, b=40), bargap=0.05,
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;color:#555;font-size:11px;font-family:monospace">
    <span>📈 Montecarlo Simulator · Proyecto 500 Balas</span>
    <span>Filtro EV≥3.5% · {n_sims} simulaciones · {n_picks:,} picks · Yield {yield_pct}%</span>
    <span>Stake: {stake}€ · Bankroll: {initial_bank:,}€</span>
</div>
""", unsafe_allow_html=True)
