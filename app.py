"""
AI-Driven Energy & Failure Prediction Dashboard
Run: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from energy_optimizer.bill_parser import load_bill_csv, load_sample_bill
from energy_optimizer.explainability import explain_anomalies, explain_demand, top_reasons
from energy_optimizer.integration import run_full_analysis
from energy_optimizer.wastage_rules import PRESETS, rules_from_sidebar

# ── Bold dark palette ────────────────────────────────────────────────────────
C = {
    "bg": "#08080c",
    "card": "#12121a",
    "border": "#2a2a3a",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "cyan": "#00f5ff",
    "magenta": "#ff2d6a",
    "gold": "#ffc300",
    "lime": "#39ff14",
    "orange": "#ff6b35",
    "purple": "#a855f7",
    "red": "#ff3366",
    "before": "#4a4a5c",
    "after_waste": "#39ff14",
    "after_profit": "#00f5ff",
    "grad_a": "#00f5ff",
    "grad_b": "#ff2d6a",
}

DARK_XAXIS = dict(
    gridcolor="#252532",
    linecolor=C["border"],
    zerolinecolor=C["border"],
    tickfont=dict(color=C["muted"]),
)
DARK_YAXIS = dict(
    gridcolor="#252532",
    linecolor=C["border"],
    zerolinecolor=C["border"],
    tickfont=dict(color=C["muted"]),
)
DARK_BASE = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="Segoe UI, sans-serif", size=13),
    title_font=dict(size=18, color=C["text"], family="Segoe UI Black, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
    margin=dict(l=50, r=30, t=70, b=50),
)


def dark_layout(**extra) -> dict:
    """Merge dark theme axes without duplicate yaxis keyword conflicts."""
    layout = dict(DARK_BASE)
    xaxis_extra = extra.pop("xaxis", {})
    yaxis_extra = extra.pop("yaxis", {})
    yaxis2_extra = extra.pop("yaxis2", None)
    layout["xaxis"] = {**DARK_XAXIS, **xaxis_extra}
    layout["yaxis"] = {**DARK_YAXIS, **yaxis_extra}
    if yaxis2_extra is not None:
        layout["yaxis2"] = {**DARK_YAXIS, "gridcolor": "rgba(0,0,0,0)", **yaxis2_extra}
    layout.update(extra)
    return layout

ANIM_CONFIG = {
    "frame": {"duration": 120, "redraw": True},
    "transition": {"duration": 200, "easing": "cubic-in-out"},
}

PLAY_BTN = [
    {
        "label": "Play",
        "method": "animate",
        "args": [
            None,
            {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True, "transition": {"duration": 180}},
        ],
    },
    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
]

st.set_page_config(
    page_title="AI Energy & Failure Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(165deg, {C['bg']} 0%, #0f0f18 45%, #141420 100%);
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0c0c12 0%, #161622 100%);
        border-right: 2px solid {C['cyan']}55;
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {C['cyan']} !important;
        font-weight: 800 !important;
    }}
    /* ── Sidebar labels (sliders, inputs) — high contrast ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider p,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectSlider label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] p {{
        color: {C['text']} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {C['muted']} !important;
    }}
    /* Slider thumb value (e.g. "Normal", "40") */
    [data-testid="stSidebar"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"] {{
        color: {C['cyan']} !important;
        font-weight: 700 !important;
    }}
    /* Slider track — cyan accent instead of hard-to-read red */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {{
        background: {C['cyan']} !important;
    }}
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {{
        background: #2a2a3e !important;
    }}
    [data-testid="stSidebar"] .stSlider [role="slider"] {{
        background: {C['gold']} !important;
        border: 2px solid {C['text']} !important;
    }}
    /* Number inputs — dark field, light text */
    [data-testid="stSidebar"] input {{
        background-color: #1a1a28 !important;
        color: {C['text']} !important;
        border: 2px solid {C['cyan']}66 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
        background-color: #252532 !important;
        color: {C['cyan']} !important;
        border-color: {C['cyan']}44 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {{
        background-color: {C['cyan']}33 !important;
        color: {C['text']} !important;
    }}
    /* Select slider dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: #1a1a28 !important;
        border-color: {C['cyan']}66 !important;
        color: {C['text']} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] span {{
        color: {C['text']} !important;
    }}
    .main-title {{
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, {C['cyan']}, {C['magenta']}, {C['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 40px {C['cyan']}44;
    }}
    .subtitle {{
        color: {C['muted']};
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
    }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {C['card']} 0%, #1a1a28 100%);
        border: 2px solid {C['cyan']}44;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 0 24px {C['cyan']}15;
    }}
    [data-testid="stMetricLabel"] {{
        color: {C['cyan']} !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetricValue"] {{
        color: {C['text']} !important;
        font-weight: 900 !important;
        font-size: 1.6rem !important;
    }}
    .rec-card {{
        background: linear-gradient(135deg, #14141f 0%, #1c1c2e 100%);
        border-left: 5px solid {C['magenta']};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 24px rgba(0,245,255,0.08);
    }}
    .rec-card h4 {{ margin: 0 0 0.4rem 0; color: {C['cyan']}; font-weight: 800; }}
    .rec-card p {{ margin: 0; color: {C['muted']}; font-size: 0.92rem; }}
    .rec-card strong {{ color: {C['gold']}; }}
    [data-testid="stTabs"] button {{
        font-weight: 800 !important;
        color: {C['muted']} !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {C['cyan']} !important;
        border-bottom-color: {C['cyan']} !important;
    }}
    h2, h3 {{ color: {C['text']} !important; font-weight: 800 !important; }}
  </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, title: str, height: int = 400, **layout_kw) -> go.Figure:
    fig.update_layout(
        **dark_layout(**layout_kw),
        title=dict(text=f"<b>{title}</b>", x=0.02),
        height=height,
    )
    return fig


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def build_forecast_chart(data: pd.DataFrame) -> go.Figure:
    x = list(range(1, len(data) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["EnergyOutput"],
            name="Actual output",
            mode="lines+markers",
            line=dict(color=C["cyan"], width=3, shape="spline"),
            marker=dict(size=6, color=C["gold"], line=dict(width=1, color=C["text"])),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["predicted_output"],
            name="Forecast",
            mode="lines",
            line=dict(color=C["magenta"], width=3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=data["forecast_error"].abs(),
            name="Forecast error",
            marker_color=C["gold"],
            opacity=0.45,
            yaxis="y2",
        )
    )
    fig.update_layout(
        xaxis_title="Sample index",
        yaxis=dict(title="Energy (MW)"),
        yaxis2=dict(title="Error (MW)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
    )
    return style_fig(fig, "Demand Forecast Analysis", height=420)


def build_wastage_chart(summary: pd.DataFrame, top_n: int = 10) -> go.Figure:
    plot_data = summary.head(top_n)
    fig = go.Figure(
        go.Bar(
            x=plot_data["Product ID"].astype(str),
            y=plot_data["total_wastage_inr"],
            marker_color=C["magenta"],
            text=[format_inr(v) for v in plot_data["total_wastage_inr"]],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Machine ID", yaxis_title="Wastage (₹)")
    return style_fig(fig, "Top machine wastage hotspots", height=420)


def build_risk_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=df["failure_risk"] * 100,
            nbinsx=20,
            marker_color=C["cyan"],
            opacity=0.8,
            name="Failure risk",
        )
    )
    fig.update_layout(xaxis_title="Failure risk (%)", yaxis_title="Count")
    return style_fig(fig, "Failure risk distribution", height=380)


def build_feature_importance_chart(importance: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=importance["mean_abs_shap"],
            y=importance["feature"],
            orientation="h",
            marker_color=C["gold"],
            text=importance["mean_abs_shap"].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Mean |SHAP|", yaxis_title="Feature")
    return style_fig(fig, title, height=380)


def add_play_slider(fig: go.Figure, n_frames: int, prefix: str = "frame") -> go.Figure:
    fig.update_layout(
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.02, "y": 1.12, "buttons": PLAY_BTN}],
        sliders=[
            {
                "active": n_frames - 1,
                "currentvalue": {"prefix": "Frame: ", "font": {"color": C["cyan"]}},
                "pad": {"t": 40},
                "bgcolor": C["card"],
                "bordercolor": C["border"],
                "tickcolor": C["cyan"],
                "font": {"color": C["text"]},
                "steps": [
                    {
                        "args": [[f"{prefix}{i}"], {"frame": {"duration": 150, "redraw": True}, "mode": "immediate"}],
                        "label": str(i + 1),
                        "method": "animate",
                    }
                    for i in range(n_frames)
                ],
            }
        ],
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### <span style='color:{C['cyan']}'>⚡ Parameters</span>", unsafe_allow_html=True)
    animate_speed = st.select_slider("Chart playback", ["Slow", "Normal", "Fast"], value="Normal")
    speed_map = {"Slow": 250, "Normal": 150, "Fast": 80}
    frame_ms = speed_map[animate_speed]

    st.markdown(f"**<span style='color:{C['gold']}'>Economics</span>**", unsafe_allow_html=True)
    tariff = st.number_input("Tariff (₹/kWh)", 4.0, 20.0, 8.0, 0.1)
    preset = st.selectbox("Wastage rules preset", list(PRESETS.keys()) + ["Custom"], index=0)
    if preset == "Custom":
        idle_torque_ratio = st.slider("Idle torque ratio", 0.20, 0.60, 0.40, 0.01)
        anomaly_risk_threshold = st.slider("Anomaly risk threshold", 0.10, 0.90, 0.48, 0.01)
        wastage_penalty_factor = st.slider("Wastage penalty factor", 0.05, 0.35, 0.22, 0.01)
        demo_wastage_scale = st.slider("Wastage scale", 1.0, 20.0, 1.0, 0.5)
    else:
        preset_base = PRESETS[preset]
        idle_torque_ratio = st.slider("Idle torque ratio", 0.20, 0.60, float(preset_base.idle_torque_ratio), 0.01)
        anomaly_risk_threshold = st.slider("Anomaly risk threshold", 0.10, 0.90, float(preset_base.anomaly_risk_threshold), 0.01)
        wastage_penalty_factor = st.slider("Wastage penalty factor", 0.05, 0.35, float(preset_base.wastage_penalty_factor), 0.01)
        demo_wastage_scale = st.slider("Wastage scale", 1.0, 20.0, float(preset_base.demo_wastage_scale), 0.5)
    rules = rules_from_sidebar(preset, idle_torque_ratio, anomaly_risk_threshold, wastage_penalty_factor, demo_wastage_scale)

    st.markdown(f"**<span style='color:{C['magenta']}'>Data inputs</span>**", unsafe_allow_html=True)
    bill_file = st.file_uploader("Upload bill CSV", type=["csv"], help="Optional: compare actual billing with model forecast.")
    bills = None
    if bill_file is not None:
        try:
            bills = load_bill_csv(bill_file.read())
        except Exception as exc:
            st.error(f"Bill upload failed: {exc}")
    if st.button("Load sample electricity bill"):
        bills = load_sample_bill()

    st.markdown(f"**<span style='color:{C['magenta']}'>Scenario</span>**", unsafe_allow_html=True)
    waste_reduction_pct = st.slider("Waste reduction (%)", 10, 60, 40, 5)
    profit_increase_pct = st.slider("Profit increase (%)", 10, 60, 35, 5)
    baseline_waste = st.number_input("Baseline waste", 500, 5000, 1200, 50)
    baseline_profit = st.number_input("Baseline profit ($K)", 100, 5000, 1800, 50)

    st.markdown(f"**<span style='color:{C['gold']}'>Plant</span>**", unsafe_allow_html=True)
    energy_output = st.slider("Energy (MW)", 100.0, 200.0, 145.6, 0.1)
    energy_delta_pct = st.slider("Energy Δ (%)", -10.0, 15.0, 5.2, 0.1)
    failure_risk = st.slider("Failure risk (%)", 0, 100, 32, 1)
    units_active = st.slider("Units active", 1, 30, 24)
    units_total = st.slider("Total units", 20, 40, 30)

    failures_count = st.slider("Failures", 0, 20, 8)
    efficiency_pct = st.slider("Efficiency (%)", 50, 100, 88)
    downtime_reduction = st.slider("Downtime cut (%)", 5, 50, 28)

after_waste = baseline_waste * (1 - waste_reduction_pct / 100)
after_profit = baseline_profit * (1 + profit_increase_pct / 100)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">AI-Driven Energy & Failure Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Combined cycle plant · Machine health monitoring · Cost & efficiency reporting</p>',
    unsafe_allow_html=True,
)

analysis = None
with st.spinner("Running demand forecast and machine wastage analysis..."):
    try:
        analysis = run_full_analysis(tariff=tariff, rules=rules, bills=bills)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

if analysis is None:
    st.stop()

kpis = analysis["kpis"]
demand_df = analysis["demand"]
wastage_detail = analysis["wastage_detail"]
wastage_summary = analysis["wastage_summary"]
recommendations = analysis["recommendations"]
bill_summary = analysis["bill_summary"]
reconciliation = analysis["reconciliation"]

avg_failure_risk_pct = float(wastage_detail["failure_risk"].mean() * 100)

try:
    demand_explain = explain_demand()
    anomaly_explain = explain_anomalies()
    shap_error = None
except Exception as exc:
    demand_explain = None
    anomaly_explain = None
    shap_error = str(exc)


tab_overview, tab_impact, tab_trends, tab_health, tab_actions = st.tabs(
    ["📊 Overview", "🎯 Impact", "📈 Trends", "🔧 Health", "✅ Actions"]
)


def animated_before_after(
    title: str, before_val: float, after_val: float, pct_label: str, y_title: str, color_after: str
) -> go.Figure:
    steps = 20
    fig = go.Figure()
    frames = []
    for i in range(1, steps + 1):
        t = i / steps
        grow_after = before_val + (after_val - before_val) * t
        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Bar(
                        x=["Before", "After"],
                        y=[before_val, grow_after],
                        marker=dict(
                            color=[C["before"], color_after],
                            line=dict(width=2, color=[C["border"], C["text"]]),
                        ),
                        text=[f"{before_val:,.0f}", f"{grow_after:,.0f}"],
                        textfont=dict(color=C["text"], size=14, family="Arial Black"),
                        textposition="outside",
                    )
                ],
            )
        )
    fig.add_trace(
        go.Bar(
            x=["Before", "After"],
            y=[before_val, before_val],
            marker=dict(color=[C["before"], color_after], line=dict(width=2, color=C["text"])),
            text=[f"{before_val:,.0f}", "…"],
            textfont=dict(color=C["text"], size=14),
            textposition="outside",
        )
    )
    fig.frames = frames
    fig.add_annotation(
        x=1, y=after_val, text=f"<b>{pct_label}</b>",
        font=dict(size=15, color=color_after),
        showarrow=True, arrowcolor=color_after, bgcolor=C["card"], bordercolor=color_after,
    )
    fig.update_layout(
        **dark_layout(),
        title=dict(text=f"<b>{title}</b>"),
        yaxis_title=y_title,
        height=400,
        showlegend=False,
        updatemenus=[{"type": "buttons", "buttons": PLAY_BTN, "x": 0, "y": 1.15}],
    )
    return fig


def animated_energy_live(hours: int, base_mw: float, risk: float) -> go.Figure:
    rng = np.random.default_rng(42)
    t_all = np.arange(hours)
    wave = base_mw + 8 * np.sin(np.linspace(0, 2 * np.pi, hours)) + rng.normal(0, 0.8, hours)
    fig = go.Figure()
    frames = []
    for i in range(1, hours + 1):
        frames.append(
            go.Frame(
                name=f"h{i}",
                data=[
                    go.Scatter(
                        x=t_all[:i], y=wave[:i],
                        mode="lines+markers",
                        line=dict(color=C["cyan"], width=4, shape="spline"),
                        marker=dict(size=8, color=C["gold"], line=dict(width=2, color=C["text"])),
                        fill="tozeroy",
                        fillcolor="rgba(0,245,255,0.15)",
                        name="Energy",
                    ),
                    go.Scatter(
                        x=t_all[:i],
                        y=np.clip(20 + risk * 0.5 + 8 * np.sin(np.linspace(0, 4, hours))[:i], 5, 95),
                        mode="lines",
                        line=dict(color=C["magenta"], width=2, dash="dot"),
                        name="Risk idx",
                        yaxis="y2",
                    ),
                ],
            )
        )
    fig.add_trace(
        go.Scatter(x=[0], y=[wave[0]], mode="lines", line=dict(color=C["cyan"], width=4), name="Energy")
    )
    fig.add_trace(
        go.Scatter(x=[0], y=[30], mode="lines", line=dict(color=C["magenta"], width=2, dash="dot"), name="Risk", yaxis="y2")
    )
    fig.frames = frames
    fig.update_layout(
        **dark_layout(
            yaxis=dict(title="MW", side="left"),
            yaxis2=dict(title="Risk", overlaying="y", side="right", range=[0, 100]),
        ),
        title=dict(text="<b>24-Hour Energy Output</b>"),
        height=320,
        xaxis_title="Hour",
        hovermode="x unified",
        updatemenus=[{"type": "buttons", "buttons": PLAY_BTN, "x": 0, "y": 1.18}],
    )
    return add_play_slider(fig, hours, "h")


def failure_risk_gauge(risk_pct: float) -> go.Figure:
    bar_color = C["lime"] if risk_pct < 35 else C["gold"] if risk_pct < 65 else C["red"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            number=dict(suffix="%", font=dict(color=C["text"], size=36, family="Arial Black")),
            title=dict(text="<b>Failure Risk</b>", font=dict(color=C["cyan"], size=16)),
            delta=dict(reference=25, increasing=dict(color=C["red"]), decreasing=dict(color=C["lime"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=C["muted"], tickwidth=2),
                bar=dict(color=bar_color, thickness=0.75),
                bgcolor=C["card"],
                borderwidth=2,
                bordercolor=C["border"],
                steps=[
                    {"range": [0, 35], "color": "rgba(57,255,20,0.25)"},
                    {"range": [35, 65], "color": "rgba(255,195,0,0.25)"},
                    {"range": [65, 100], "color": "rgba(255,51,102,0.35)"},
                ],
                threshold=dict(line=dict(color=C["magenta"], width=5), thickness=0.85, value=70),
            ),
        )
    )
    return style_fig(fig, "", height=300)


def animated_performance(days: list[str], values: list[float]) -> go.Figure:
    fig = go.Figure()
    frames = []
    for i in range(1, len(days) + 1):
        frames.append(
            go.Frame(
                name=f"d{i}",
                data=[
                    go.Scatter(
                        x=days[:i], y=values[:i],
                        mode="lines+markers",
                        line=dict(color=C["cyan"], width=4, shape="spline"),
                        marker=dict(size=12, color=C["gold"], symbol="diamond",
                                    line=dict(width=2, color=C["text"])),
                        fill="tozeroy",
                        fillcolor="rgba(0,245,255,0.12)",
                    ),
                    go.Scatter(
                        x=days[:i], y=[85] * i,
                        mode="lines",
                        line=dict(color=C["magenta"], width=2, dash="dash"),
                        name="Target",
                    ),
                ],
            )
        )
    fig.add_trace(go.Scatter(x=[days[0]], y=[values[0]], mode="markers", marker=dict(size=12, color=C["cyan"])))
    fig.frames = frames
    if "Wed" in days:
        wi = days.index("Wed")
        fig.add_annotation(
            x="Wed", y=values[wi], text="<b>Anomaly — Wed</b>",
            font=dict(size=13, color=C["red"]),
            showarrow=True, arrowcolor=C["red"], bgcolor="rgba(255,51,102,0.2)", bordercolor=C["red"],
        )
    fig.update_layout(
        **dark_layout(yaxis=dict(range=[75, 100], title="Performance (%)")),
        title=dict(text="<b>Daily Performance (Mon–Fri)</b>"),
        height=440,
        updatemenus=[{"type": "buttons", "buttons": PLAY_BTN}],
    )
    return add_play_slider(fig, len(days), "d")


def animated_health_bars(components: list[str], health: list[float]) -> go.Figure:
    colors = [C["lime"] if h >= 80 else C["gold"] if h >= 70 else C["red"] for h in health]
    fig = go.Figure()
    frames = []
    for step in range(1, 21):
        t = step / 20
        frames.append(
            go.Frame(
                name=f"s{step}",
                data=[
                    go.Bar(
                        x=components,
                        y=[h * t for h in health],
                        marker=dict(
                            color=colors,
                            line=dict(width=2, color=C["text"]),
                        ),
                        text=[f"{h*t:.0f}%" for h in health],
                        textfont=dict(color=C["text"], size=14, family="Arial Black"),
                        textposition="outside",
                    )
                ],
            )
        )
    fig.add_trace(go.Bar(x=components, y=[0] * len(components), marker_color=colors))
    fig.frames = frames
    fig.add_hline(y=70, line_dash="dash", line_color=C["magenta"], line_width=2,
                  annotation=dict(text="MIN", font=dict(color=C["magenta"])))
    fig.update_layout(
        **dark_layout(yaxis=dict(range=[0, 105], title="Health (%)")),
        title=dict(text="<b>Component Health Status</b>"),
        height=420,
        updatemenus=[{"type": "buttons", "buttons": PLAY_BTN}],
    )
    return fig


# ═══ IMPACT ═══════════════════════════════════════════════════════════════════
with tab_impact:
    st.subheader("Optimization Impact Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Estimated monthly cost", format_inr(kpis["est_monthly_cost_inr"]), f"{tariff:.2f} ₹/kWh")
        st.metric("Projected machine wastage", format_inr(kpis["total_wastage_inr"]), f"{kpis['machines_flagged']} machines flagged")
    with c2:
        st.metric("Potential savings", format_inr(kpis["potential_savings_inr"]), "Based on AI recommendations")
        if bill_summary is not None:
            st.metric("Avg bill / month", format_inr(bill_summary["avg_monthly_inr"]), f"{kpis['wastage_vs_bill_pct']:.1f}% of bill")
        else:
            st.info("Upload bill data to reconcile actual costs with the model forecast.")

    st.plotly_chart(build_wastage_chart(wastage_summary), use_container_width=True)
    with st.expander("Top machine wastage details"):
        st.dataframe(
            wastage_summary[
                ["Product ID", "Type", "total_wastage_inr", "idle_wastage_inr", "anomaly_wastage_inr", "avg_failure_risk"]
            ].rename(
                columns={
                    "Product ID": "Machine",
                    "Type": "Class",
                    "total_wastage_inr": "Total wastage (₹)",
                    "idle_wastage_inr": "Idle wastage (₹)",
                    "anomaly_wastage_inr": "Anomaly wastage (₹)",
                    "avg_failure_risk": "Avg failure risk",
                }
            ),
            use_container_width=True,
        )

# ═══ OVERVIEW ═════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Energy & Failure Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecast demand", f"{kpis['avg_forecast_mw']:.1f} MW", f"{kpis['avg_hourly_cost_inr']:.0f} ₹/h")
    k2.metric("Monthly wastage", format_inr(kpis["total_wastage_inr"]), f"{kpis['machines_flagged']} machines")
    k3.metric("Potential savings", format_inr(kpis["potential_savings_inr"]), "Actionable guidance")
    k4.metric("Avg failure risk", f"{avg_failure_risk_pct:.1f}%", "Machine anomaly exposure")

    gc, cc = st.columns([1, 2])
    with gc:
        st.plotly_chart(failure_risk_gauge(avg_failure_risk_pct), use_container_width=True)
    with cc:
        st.plotly_chart(build_forecast_chart(demand_df), use_container_width=True)

# ═══ TRENDS ═══════════════════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Demand and Risk Trends")
    st.plotly_chart(build_forecast_chart(demand_df), use_container_width=True)
    st.plotly_chart(build_risk_distribution(wastage_detail), use_container_width=True)
    if reconciliation is not None:
        st.markdown("### Model vs actual bill reconciliation")
        st.dataframe(reconciliation, use_container_width=True)

# ═══ HEALTH ═════════════════════════════════════════════════════════════════════
with tab_health:
    st.subheader("Explainability & Risk Drivers")
    if shap_error:
        st.warning(f"SHAP explainability unavailable: {shap_error}")
        st.markdown("Use `pip install shap` and ensure the model is trained to enable feature-level explainability.")
    else:
        explain_tabs = st.tabs(["Forecast driver importance", "Anomaly driver importance"])
        with explain_tabs[0]:
            st.plotly_chart(build_feature_importance_chart(demand_explain["importance"], "Demand forecast feature importance"), use_container_width=True)
        with explain_tabs[1]:
            st.plotly_chart(build_feature_importance_chart(anomaly_explain["importance"], "Anomaly feature importance"), use_container_width=True)

        st.markdown("### Top reason summaries")
        st.markdown("- Forecast drivers: " + "; ".join(top_reasons(demand_explain["importance"], 3)))
        st.markdown("- Anomaly drivers: " + "; ".join(top_reasons(anomaly_explain["importance"], 3)))

# ═══ ACTIONS ════════════════════════════════════════════════════════════════════
with tab_actions:
    st.subheader("Optimization Recommendations")
    pf = st.radio("Priority", ["All", "High", "Medium", "Low"], horizontal=True)
    for rec in recommendations:
        if pf != "All" and rec["priority"] != pf:
            continue
        st.markdown(
            f"""<div class="rec-card" style="border-left-color:{C['gold']};">
            <h4>{rec['category']} <span style="color:{C['gold']};">({rec['priority']})</span></h4>
            <p>{rec['message']}</p><p><strong>Estimated savings: {format_inr(rec['savings_inr'])}</strong></p></div>""",
            unsafe_allow_html=True,
        )
    if reconciliation is not None:
        st.markdown("---")
        st.subheader("Billing reconciliation details")
        st.dataframe(reconciliation, use_container_width=True)

st.markdown("---")
st.markdown(
    f"<p style='color:{C['muted']};font-weight:600;text-align:center;font-size:0.85rem;'>"
    f"Industrial Energy Optimizer · Tariff {tariff:.2f} ₹/kWh · "
    f"Total waste {format_inr(kpis['total_wastage_inr'])} · "
    f"Potential savings {format_inr(kpis['potential_savings_inr'])}</p>",
    unsafe_allow_html=True,
)
