"""AquaSense AI - water-quality anomaly detection and next-step pH forecasting.

Dashboard styled after the AquaGuard AI layout: KPI strip, tabbed workspace,
interactive simulator with presets and WHO/EPA reference auditing.
Run with:  streamlit run app.py
(The previous sidebar version is kept as app_legacy.py.)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aquaguard import (  # noqa: E402
    EXPERIMENTS, META, PRESETS, SENSOR_META, SENSORS,
    Pipeline, audit_reading, baseline_check, load_csv,
)

st.set_page_config(page_title="AquaSense AI | Water Quality Intelligence", page_icon="\U0001F4A7",
                   layout="wide", initial_sidebar_state="collapsed")

CYAN, GREEN, RED, AMBER, VIOLET = "#22d3ee", "#34d399", "#f87171", "#fbbf24", "#8b5cf6"

# ----------------------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#070f24;--card:#0c1a3a;--line:#1a2d5c;--ink:#e6edf7;--muted:#8da2c8;
--cyan:#22d3ee;--green:#34d399;--red:#f87171;--amber:#fbbf24;--violet:#8b5cf6;}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif;}
.stApp{background:radial-gradient(1200px 500px at 15% -10%,#0f2250 0%,rgba(7,15,36,0) 60%),#070f24;color:var(--ink);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1500px;}

.ag-head{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0 1.1rem;border-bottom:1px solid var(--line);margin-bottom:1.4rem;gap:1rem;flex-wrap:wrap}
.ag-brand{display:flex;align-items:center;gap:1rem}
.ag-logo{width:58px;height:58px;border-radius:16px;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.45);display:flex;align-items:center;justify-content:center}
.ag-title{font-size:1.75rem;font-weight:800;letter-spacing:-.02em;line-height:1.1;display:flex;align-items:center;gap:.6rem}
.ag-ver{font-size:.72rem;font-weight:600;color:var(--cyan);border:1px solid rgba(34,211,238,.4);background:rgba(34,211,238,.08);padding:.12rem .5rem;border-radius:7px;font-family:'JetBrains Mono',monospace}
.ag-sub{color:#b4c3e0;font-size:.95rem;font-weight:500;margin-top:.15rem}
.ag-right{display:flex;align-items:center;gap:1rem}
.ag-clock{font-family:'JetBrains Mono',monospace;color:var(--muted);font-size:.95rem}
.ag-pill{display:flex;align-items:center;gap:.5rem;padding:.55rem 1.1rem;border-radius:999px;font-weight:700;font-size:.9rem;color:var(--green);background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.4)}
.ag-dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}

.kpi{background:linear-gradient(160deg,#0d1d42,#0a1632);border:1px solid var(--line);border-radius:20px;padding:1.3rem 1.5rem;min-height:168px;position:relative}
.kpi-label{font-size:.78rem;font-weight:700;letter-spacing:.09em;color:#c3d0ea;text-transform:uppercase}
.kpi-val{font-size:2.05rem;font-weight:800;letter-spacing:-.02em;margin-top:1.15rem;line-height:1.15}
.kpi-sub{color:var(--muted);font-size:.9rem;margin-top:.35rem}
.kpi-ico{position:absolute;right:1.3rem;top:1.2rem;width:54px;height:54px;border-radius:14px;display:flex;align-items:center;justify-content:center}

.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--line);margin-top:1.6rem;flex-wrap:wrap}
.stTabs [data-baseweb="tab"] p{font-size:1.02rem;font-weight:600}
.stTabs [data-baseweb="tab"]{height:3.1rem;padding:0 1.3rem;border-radius:12px 12px 0 0;color:#b4c3e0;font-weight:600;font-size:1rem;background:transparent}
.stTabs [aria-selected="true"]{background:rgba(34,211,238,.1);color:var(--cyan)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--cyan);height:3px}
.stTabs [data-baseweb="tab-border"]{display:none}

.panel-title{font-size:1.45rem;font-weight:800;letter-spacing:-.01em;display:flex;align-items:center;gap:.6rem;margin:.4rem 0 .2rem}
.panel-sub{color:#b4c3e0;font-size:.98rem;margin-bottom:1rem}
.notice{border:1px solid rgba(34,211,238,.35);background:rgba(34,211,238,.06);border-radius:14px;padding:.95rem 1.2rem;color:#c9d6ee;font-size:.93rem;line-height:1.55;margin:.6rem 0 1.2rem}
.notice b{color:#fff}
.warnbox{border:1px solid rgba(251,191,36,.4);background:rgba(251,191,36,.07);border-radius:14px;padding:.95rem 1.2rem;color:#e9dcb4;font-size:.93rem;line-height:1.55;margin:.8rem 0}
.warnbox b{color:#fde68a}
.sec-label{font-size:.78rem;font-weight:700;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin:.6rem 0 .5rem}

div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important;border-color:var(--line)!important;background:rgba(10,22,48,.75)}
.sn-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:.15rem;gap:.5rem}
.sn-name{font-weight:700;font-size:1rem}
.sn-badge{font-family:'JetBrains Mono',monospace;font-size:.74rem;padding:.2rem .55rem;border-radius:8px;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--muted);white-space:nowrap}
.sn-badge.bad{color:var(--red);border-color:rgba(248,113,113,.5);background:rgba(248,113,113,.08)}

div[data-testid="stNumberInput"] input{background:#070f24!important;border:1px solid var(--line)!important;border-radius:10px!important;text-align:center;font-weight:700;font-size:1.02rem}
div[data-testid="stSlider"] [role="slider"]{background:#e6fbff;border:3px solid var(--cyan);box-shadow:0 0 0 4px rgba(34,211,238,.15)}
div[data-testid="stSlider"] div[data-testid="stTickBar"]{display:none}
div[data-testid="stSlider"] [data-testid="stSliderThumbValue"]{display:none}

.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.03);color:#cfdaf0;font-weight:600;padding:.5rem .6rem;width:100%;transition:.15s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--cyan);color:var(--cyan);background:rgba(34,211,238,.06)}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{border-color:var(--cyan);color:var(--cyan);background:rgba(34,211,238,.1)}

.res-main{border-radius:16px;padding:1.2rem 1.4rem;display:flex;gap:1rem;align-items:flex-start}
.res-main.bad{border:1px solid rgba(248,113,113,.5);background:linear-gradient(135deg,rgba(248,113,113,.16),rgba(120,20,40,.18))}
.res-main.good{border:1px solid rgba(52,211,153,.5);background:linear-gradient(135deg,rgba(52,211,153,.14),rgba(10,70,60,.18))}
.res-h{font-size:1.25rem;font-weight:800;letter-spacing:-.01em}
.res-t{color:#c3d0ea;font-size:.93rem;line-height:1.5;margin-top:.3rem}
.mini{border:1px solid var(--line);background:rgba(10,22,48,.8);border-radius:14px;padding:.8rem 1rem;height:100%}
.mini .l{font-size:.72rem;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
.mini .v{font-size:1.45rem;font-weight:800;margin-top:.2rem}
.mini .s{font-size:.8rem;color:var(--muted)}
.aud{display:flex;justify-content:space-between;align-items:center;padding:.5rem .2rem;border-bottom:1px solid rgba(26,45,92,.7);font-size:.93rem}
.aud:last-child{border-bottom:none}
.aud .n{font-weight:600}.aud .r{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:.78rem}
.tag{font-size:.72rem;font-weight:800;padding:.18rem .6rem;border-radius:999px;letter-spacing:.04em}
.tag.ok{color:var(--green);background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.4)}
.tag.no{color:var(--red);background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.45)}

div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(10,22,48,.5)}
@media (max-width:900px){.ag-title{font-size:1.35rem}.kpi-val{font-size:1.7rem}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- helpers
_SVG = ('<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">{p}</svg>')
_PATHS = {
    "drop": '<path d="M12 2.7s7 6.6 7 12a7 7 0 0 1-14 0c0-5.4 7-12 7-12z"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "pulse": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "bolt": '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-6"/>',
    "x": '<circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/>',
}


def icon(name: str, color: str = CYAN, size: int = 26) -> str:
    return _SVG.format(s=size, c=color, p=_PATHS[name])


def kpi(label, value, sub, color, ico, ico_bg):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-val" style="color:{color}">{value}</div><div class="kpi-sub">{sub}</div>'
            f'<div class="kpi-ico" style="background:{ico_bg}">{icon(ico, color)}</div></div>')


def mini(label, value, sub="", color="#e6edf7"):
    return (f'<div class="mini"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=80, b=10), template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,22,48,.6)",
        font=dict(family="Inter, sans-serif", color="#c3d0ea", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(size=14, color="#e6edf7"), x=0.0, y=0.97, yanchor="top"),
    )
    fig.update_xaxes(gridcolor="rgba(141,162,200,.12)", zerolinecolor="rgba(141,162,200,.25)")
    fig.update_yaxes(gridcolor="rgba(141,162,200,.12)", zerolinecolor="rgba(141,162,200,.25)")
    return fig


@st.cache_resource(show_spinner="Loading frozen models\u2026")
def get_pipe() -> Pipeline:
    return Pipeline()


@st.cache_data(show_spinner=False)
def get_baselines():
    return baseline_check(get_pipe())


@st.cache_data(show_spinner=False)
def get_stream() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "test_v1.csv")
    out = get_pipe().run_batch(df, standardised=True)
    out["ph_raw"] = get_pipe().ph_from_z(out["ph"])
    return out


@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "water_quality_timestamped.csv")


pipe = get_pipe()
now = datetime.now(ZoneInfo("Asia/Kolkata"))

# ----------------------------------------------------------------------------- header
st.markdown(
    f"""
<div class="ag-head">
  <div class="ag-brand">
    <div class="ag-logo">{icon('drop', CYAN, 30)}</div>
    <div>
      <div class="ag-title">AquaSense AI <span class="ag-ver">v2.0.0</span></div>
      <div class="ag-sub">Water-Quality Anomaly Detection &amp; Next-Step pH Forecasting</div>
    </div>
  </div>
  <div class="ag-right">
    <div class="ag-clock">{now:%Y-%m-%d %H:%M:%S} IST</div>
    <div class="ag-pill"><span class="ag-dot"></span>MODELS LOADED</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

kpi_slot = st.empty()  # filled once the simulator has produced live values


# ----------------------------------------------------------------------------- simulator state
def _set_values(values: dict) -> None:
    for k in SENSORS:
        m = META[k]
        v = round(float(min(max(values[k], m.lo), m.hi)), 2)
        st.session_state[f"sl_{k}"] = v
        st.session_state[f"nu_{k}"] = v


def on_slider(k: str) -> None:
    st.session_state[f"nu_{k}"] = st.session_state[f"sl_{k}"]
    st.session_state.preset = "Custom"


def on_number(k: str) -> None:
    st.session_state[f"sl_{k}"] = st.session_state[f"nu_{k}"]
    st.session_state.preset = "Custom"


def apply_preset(name: str) -> None:
    if name == "Dataset Median":
        _set_values(pipe.median_reading)
    elif name == "Randomize":
        row = get_raw().dropna(subset=SENSORS).sample(1).iloc[0]
        _set_values({k: float(row[k]) for k in SENSORS})
    else:
        _set_values(PRESETS[name])
    st.session_state.preset = name


if "preset" not in st.session_state:
    apply_preset("Dataset Median")

PRESET_ORDER = ["Dataset Median", "Mountain Spring", "Municipal Tap", "Industrial Runoff",
                "Acid Drainage", "Sensor Failure"]

tab_sim, tab_stream, tab_batch, tab_bench = st.tabs([
    "\u2699\ufe0e  Interactive Simulator",
    "\u223F  Continuous Telemetry Stream",
    "\u25A6  Batch Telemetry Processor (9 Features)",
    "\u2551  Model Benchmark & WHO Standards",
])

NOTICE = (
    "<b>Educational &amp; Demonstration Notice:</b> AquaSense AI is a research prototype trained on a public "
    "physicochemical benchmark dataset with <i>synthetic</i> timestamps and no ground-truth anomaly labels. "
    "It provides algorithmic estimates and is <i>not</i> a substitute for certified laboratory assays. "
    "Drinking-water decisions must be validated by licensed local testing authorities."
)

# =============================================================================== TAB 1
with tab_sim:
    st.markdown(f'<div class="panel-title">{icon("drop", "#e6edf7", 24)} Physicochemical Water Quality Telemetry Input</div>'
                '<div class="panel-sub">Adjust sliders to run the frozen Isolation Forest and Random Forest in real time, '
                'alongside a deterministic 9-axis WHO/EPA reference audit.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="notice">{NOTICE}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Standard presets</div>', unsafe_allow_html=True)
    pc = st.columns(len(PRESET_ORDER) + 1)
    for i, name in enumerate(PRESET_ORDER):
        pc[i].button(name, key=f"pb_{name}", on_click=apply_preset, args=(name,),
                     type="primary" if st.session_state.preset == name else "secondary")
    pc[-1].button("\u21BB Randomize Values", key="pb_rand", on_click=apply_preset, args=("Randomize",),
                  type="primary" if st.session_state.preset == "Randomize" else "secondary")

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        grid = st.columns(2, gap="medium")
        for i, s in enumerate(SENSOR_META):
            cur = st.session_state[f"sl_{s.key}"]
            bad = not ((s.ref_lo is None or cur >= s.ref_lo) and (s.ref_hi is None or cur <= s.ref_hi))
            with grid[i % 2]:
                with st.container(border=True):
                    unit = f" ({s.unit})" if s.unit else ""
                    st.markdown(
                        f'<div class="sn-head"><span class="sn-name">{s.label}{unit}</span>'
                        f'<span class="sn-badge{" bad" if bad else ""}">WHO: {s.ref_text}</span></div>',
                        unsafe_allow_html=True)
                    c1, c2 = st.columns([2.6, 1.3], vertical_alignment="center")
                    fmt = "%.2f" if s.step < 1 else "%.0f"
                    c1.slider(s.label, min_value=s.lo, max_value=s.hi, step=s.step, key=f"sl_{s.key}",
                              label_visibility="collapsed", on_change=on_slider, args=(s.key,))
                    c2.number_input(s.label, min_value=s.lo, max_value=s.hi, step=s.step, format=fmt,
                                    key=f"nu_{s.key}", label_visibility="collapsed",
                                    on_change=on_number, args=(s.key,))

        with st.expander("Temporal context for the next-step forecast", expanded=False):
            t1, t2, t3 = st.columns(3)
            ctx_date = t1.date_input("Observation date", value=now.date(), key="ctx_date")
            ctx_time = t2.time_input("Observation time", value=now.time().replace(second=0, microsecond=0),
                                     key="ctx_time")
            stable = t3.checkbox("Assume stable previous pH", value=True,
                                 help="Sets pH(t-1) and pH(t-2) equal to the current pH.")
            cur_ph = float(st.session_state["sl_ph"])
            if stable:
                prev1 = prev2 = cur_ph
            else:
                p1, p2 = st.columns(2)
                prev1 = p1.number_input("pH at t-1", 0.0, 14.0, cur_ph, 0.01, key="prev1")
                prev2 = p2.number_input("pH at t-2", 0.0, 14.0, cur_ph, 0.01, key="prev2")

    reading = {k: float(st.session_state[f"sl_{k}"]) for k in SENSORS}
    out = pipe.predict_one(reading, datetime.combine(ctx_date, ctx_time), (prev2, prev1))
    aud = audit_reading(reading)
    n_viol = sum(not a["ok"] for a in aud)
    zs = out["z"]
    top = sorted(zs.items(), key=lambda kv: -abs(kv[1]))[:2]
    top_txt = ", ".join(f"{META[k].label} (z = {v:+.1f})" for k, v in top)
    delta = out["next_ph"] - reading["ph"]

    with right:
        with st.container(border=True):
            if out["anomaly"]:
                st.markdown(
                    f'<div class="res-main bad">{icon("x", RED, 34)}<div><div class="res-h">Anomalous Profile Detected (Model Estimate)</div>'
                    f'<div class="res-t">This reading is unusual compared with the training distribution. '
                    f'Largest deviations: {top_txt}.</div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="res-main good">{icon("check", GREEN, 34)}<div><div class="res-h">Within Learned Normal Range (Model Estimate)</div>'
                    f'<div class="res-t">This reading resembles typical training observations. '
                    f'Largest deviations: {top_txt}.</div></div></div>', unsafe_allow_html=True)
            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.markdown(mini("Anomaly score", f'{out["score"]:+.3f}', "&lt; 0 = anomalous",
                             RED if out["anomaly"] else GREEN), unsafe_allow_html=True)
            m2.markdown(mini("Next-step pH", f'{out["next_ph"]:.2f}', f"{delta:+.2f} vs current", CYAN),
                        unsafe_allow_html=True)
            m3.markdown(mini("Latency", f'{out["latency_ms"]:.1f} ms', "both models, 1 row", VIOLET),
                        unsafe_allow_html=True)
            st.markdown(f'<div class="sec-label" style="margin-top:1rem">WHO / EPA reference audit \u00b7 {n_viol} of 9 outside range</div>',
                        unsafe_allow_html=True)
            rows = "".join(
                f'<div class="aud"><div><span class="n">{a["label"]}</span><br>'
                f'<span class="r">{a["value"]:,.2f} {a["unit"]} \u00b7 ref {a["ref"]}</span></div>'
                f'<span class="tag {"ok" if a["ok"] else "no"}">{"PASS" if a["ok"] else "VIOLATION"}</span></div>'
                for a in aud)
            st.markdown(rows, unsafe_allow_html=True)
        st.markdown('<div class="warnbox"><b>Two independent signals.</b> The anomaly detector measures how unusual a reading is '
                    '<i>relative to the training dataset</i>; the audit checks fixed reference limits. Pristine spring water can '
                    'pass every limit and still be flagged, because this dataset\u2019s typical readings are far from pristine.</div>',
                    unsafe_allow_html=True)

    zfig = go.Figure(go.Bar(
        y=[META[k].label for k in SENSORS][::-1], x=[zs[k] for k in SENSORS][::-1], orientation="h",
        marker_color=[RED if abs(zs[k]) > 3 else AMBER if abs(zs[k]) > 2 else CYAN for k in SENSORS][::-1],
        hovertemplate="%{y}: z = %{x:.2f}<extra></extra>"))
    for xv in (-3, 3):
        zfig.add_vline(x=xv, line_dash="dot", line_color="rgba(248,113,113,.6)")
    zfig.update_layout(title="Deviation from the training distribution (standard deviations)")
    st.plotly_chart(style_fig(zfig, 340), key="zfig")

# ------------------------------------------------------------------------------ KPI strip
bl = get_baselines()["table"].set_index("Method")
rf_mae = bl.loc["Tuned Random Forest", "MAE"]
per_mae = bl.loc["Persistence (last value)", "MAE"]
gain = (per_mae - rf_mae) / per_mae * 100
with kpi_slot.container():
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.markdown(kpi("Champion pipeline", "Isolation Forest", "+ Random Forest Regressor (300 Trees)",
                    "#e6edf7", "shield", "rgba(34,211,238,.14)"), unsafe_allow_html=True)
    k2.markdown(kpi("Forecast MAE (test)", f"{rf_mae:.4f}",
                    f"\u2212{gain:.1f}% vs {per_mae:.4f} persistence baseline", CYAN, "pulse",
                    "rgba(52,211,153,.14)"), unsafe_allow_html=True)
    k3.markdown(kpi("Inference latency", f'{out["latency_ms"]:.1f} ms', "Measured live, both models, single row",
                    GREEN, "bolt", "rgba(139,92,246,.2)"), unsafe_allow_html=True)
    k4.markdown(kpi("WHO auditor", f"{n_viol} Violations", "9 deterministic reference guardrails",
                    RED if n_viol else GREEN, "check", "rgba(59,130,246,.2)"), unsafe_allow_html=True)

# =============================================================================== TAB 2
with tab_stream:
    st.markdown(f'<div class="panel-title">{icon("pulse", "#e6edf7", 24)} Continuous Telemetry Stream (Replay)</div>'
                '<div class="panel-sub">Replays the held-out test split (491 observations) through the frozen models. '
                'Timestamps are synthetic: this demonstrates the workflow, not a live sensor feed.</div>',
                unsafe_allow_html=True)
    stream = get_stream()
    n = len(stream)
    if "head" not in st.session_state:
        st.session_state.head = 120
    c1, c2, c3 = st.columns([1, 1, 3], vertical_alignment="bottom")
    window = c1.select_slider("Window (points)", options=[40, 60, 90, 120, 180], value=90)
    if c2.button("\u25B6 Advance 20 steps"):
        st.session_state.head = min(n, st.session_state.head + 20)
    st.session_state.head = max(window, min(n, st.session_state.head))
    head = c3.slider("Replay position", window, n, key="head")
    view = stream.iloc[head - window:head].copy()
    view["ts"] = pd.to_datetime(view["timestamp"])

    f = go.Figure()
    f.add_hrect(y0=6.5, y1=8.5, fillcolor="rgba(52,211,153,.07)", line_width=0)
    f.add_trace(go.Scatter(x=view["ts"], y=view["ph_raw"], name="Observed pH", line=dict(color=CYAN, width=2)))
    f.add_trace(go.Scatter(x=view["ts"].shift(-1), y=view["predicted_next_ph"], name="Forecast (t+1)",
                           line=dict(color=VIOLET, width=2, dash="dot")))
    fl = view[view["anomaly_flag"] == 1]
    f.add_trace(go.Scatter(x=fl["ts"], y=fl["ph_raw"], mode="markers", name="Flagged anomaly",
                           marker=dict(color=RED, size=11, symbol="diamond", line=dict(color="#fff", width=1))))
    f.update_layout(title="pH: observed vs one-step forecast (green band = WHO 6.5\u20138.5)")
    st.plotly_chart(style_fig(f, 380), key="stream_ph")

    g = go.Figure(go.Scatter(x=view["ts"], y=view["anomaly_score"], line=dict(color=AMBER, width=1.8),
                             fill="tozeroy", fillcolor="rgba(251,191,36,.08)", name="Anomaly score"))
    g.add_hline(y=0, line_color=RED, line_dash="dash")
    g.update_layout(title="Isolation Forest decision score (below 0 = flagged)")
    st.plotly_chart(style_fig(g, 260), key="stream_score")

    s1, s2, s3 = st.columns(3)
    s1.markdown(mini("Flagged in window", int(view["anomaly_flag"].sum()), f"of {window} observations", RED),
                unsafe_allow_html=True)
    s2.markdown(mini("Whole test split", f'{stream["anomaly_flag"].mean() * 100:.2f}%',
                     "anomaly rate (matches saved results)", CYAN), unsafe_allow_html=True)
    s3.markdown(mini("Avg WHO violations", f'{view["who_violations"].mean():.1f}', "per observation, of 9", AMBER),
                unsafe_allow_html=True)

# =============================================================================== TAB 3
with tab_batch:
    st.markdown(f'<div class="panel-title">{icon("drop", "#e6edf7", 24)} Batch Telemetry Processor</div>'
                '<div class="panel-sub">Upload a CSV with the nine sensor columns (and optionally a <code>timestamp</code>) to get '
                'anomaly flags, scores, next-step pH and WHO audit counts for every row.</div>', unsafe_allow_html=True)
    b1, b2 = st.columns([2, 1])
    up = b1.file_uploader("Telemetry CSV", type=["csv"], help="Required columns: " + ", ".join(SENSORS))
    units = b2.radio("Input units", ["Raw physical units", "Already standardised (z-score)"], index=0)
    use_sample = b2.checkbox("Use bundled sample (last 200 rows)", value=True)
    sample = get_raw().tail(200)
    b1.download_button("\u2B07 Download CSV template",
                       sample.head(24)[["timestamp"] + SENSORS].to_csv(index=False).encode(),
                       "aquasense_template.csv", "text/csv")

    src = None
    if up is not None:
        src = pd.read_csv(up)
    elif use_sample:
        src = sample[["timestamp"] + SENSORS].reset_index(drop=True)

    if src is None:
        st.info("Upload a CSV or tick the bundled sample to run the pipeline.")
    else:
        missing = [c for c in SENSORS if c not in src.columns]
        res = None
        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}")
        else:
            standardised = units.startswith("Already")
            try:
                res = pipe.run_batch(src, standardised=standardised)
            except Exception as exc:  # surface bad files politely
                st.error(f"Could not process this file: {exc}")
        if res is not None:
            n_fl = int(res["anomaly_flag"].sum())
            q = st.columns(4)
            q[0].markdown(mini("Rows processed", f"{len(res):,}"), unsafe_allow_html=True)
            q[1].markdown(mini("Flagged anomalies", n_fl, f"{n_fl / len(res) * 100:.2f}% of rows", RED),
                          unsafe_allow_html=True)
            q[2].markdown(mini("Mean WHO violations", f'{res["who_violations"].mean():.1f}', "per row, of 9", AMBER),
                          unsafe_allow_html=True)
            q[3].markdown(mini("Forecast rows", int(res["predicted_next_ph"].notna().sum()),
                               "first 2 rows lack lag history", CYAN), unsafe_allow_html=True)
            if not standardised:
                fig = go.Figure(go.Scatter(x=res["timestamp"], y=res["ph"], name="pH", line=dict(color=CYAN, width=1.8)))
                fig.add_trace(go.Scatter(x=res["timestamp"], y=res["predicted_next_ph"].shift(1),
                                         name="Forecast for this step", line=dict(color=VIOLET, width=1.6, dash="dot")))
                fl = res[res["anomaly_flag"] == 1]
                fig.add_trace(go.Scatter(x=fl["timestamp"], y=fl["ph"], mode="markers", name="Flagged",
                                         marker=dict(color=RED, size=9, symbol="diamond")))
                fig.update_layout(title="pH, forecast and flagged rows")
                st.plotly_chart(style_fig(fig, 340), key="batch_fig")
            only = st.toggle("Show flagged rows only", value=False)
            st.dataframe(res[res["anomaly_flag"] == 1] if only else res, hide_index=True)
            st.download_button("\u2B07 Download results CSV", res.to_csv(index=False).encode(),
                               "aquasense_batch_results.csv", "text/csv", type="primary")

# =============================================================================== TAB 4
with tab_bench:
    st.markdown(f'<div class="panel-title">{icon("shield", "#e6edf7", 24)} Model Benchmark &amp; WHO Standards</div>'
                '<div class="panel-sub">Saved validation results from the project plus a re-computed baseline check on the held-out test split.</div>',
                unsafe_allow_html=True)
    b = get_baselines()
    tbl = b["table"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE", x=tbl["Method"], y=tbl["MAE"], marker_color=CYAN,
                         text=tbl["MAE"].round(4), textposition="outside"))
    fig.add_trace(go.Bar(name="RMSE", x=tbl["Method"], y=tbl["RMSE"], marker_color=VIOLET,
                         text=tbl["RMSE"].round(4), textposition="outside"))
    fig.update_layout(barmode="group", yaxis_range=[0, float(tbl["RMSE"].max()) * 1.2],
                      title=f"Next-step pH forecast error on the test split (standardised units, n = {b['n']})")
    st.plotly_chart(style_fig(fig, 360), key="bench_fig")

    mean_mae = tbl.set_index("Method").loc["Train-mean constant", "MAE"]
    st.markdown(
        f'<div class="warnbox"><b>Read this before citing the forecast result.</b> The Random Forest beats the persistence baseline by '
        f'{gain:.1f}%, but improves on a constant "always predict the training mean" model by only '
        f'{(mean_mae - rf_mae) / mean_mae * 100:.1f}% MAE. The correlation between pH(t) and pH(t+1) is just {b["lag1_corr"]:.2f}, '
        f'so this dataset has weak temporal structure and persistence is an unusually weak baseline. '
        f'Errors are in standardised units (1 unit \u2248 {pipe.std[0]:.2f} pH).</div>', unsafe_allow_html=True)

    l, r = st.columns(2, gap="large")
    with l:
        st.markdown('<div class="sec-label">Anomaly detection (final validation)</div>', unsafe_allow_html=True)
        a = load_csv(EXPERIMENTS / "day23_final_validation_results" / "anomaly_performance.csv")
        if a is not None:
            st.dataframe(a.round(4), hide_index=True)
        st.markdown('<div class="sec-label">Chronological validation (expanding window)</div>', unsafe_allow_html=True)
        cv = load_csv(EXPERIMENTS / "day20_chronological_validation.csv")
        if cv is not None:
            cf = go.Figure()
            cf.add_trace(go.Bar(x=cv["Fold"], y=cv["Validation MAE"], name="MAE", marker_color=CYAN))
            cf.add_trace(go.Bar(x=cv["Fold"], y=cv["Validation RMSE"], name="RMSE", marker_color=VIOLET))
            cf.update_layout(barmode="group")
            st.plotly_chart(style_fig(cf, 270), key="cv_fig")
    with r:
        st.markdown('<div class="sec-label">Feature ablation (anomaly-rate shift when a sensor is removed)</div>',
                    unsafe_allow_html=True)
        ab = load_csv(EXPERIMENTS / "day20_feature_ablation_results.csv")
        if ab is not None:
            af = go.Figure(go.Bar(y=ab["Removed Feature"][::-1], x=ab["Absolute Change (%)"][::-1],
                                  orientation="h", marker_color=AMBER))
            af.update_layout(xaxis_title="Absolute change in validation anomaly rate (pp)")
            st.plotly_chart(style_fig(af, 340), key="ab_fig")

    with st.expander("Hyper-parameter search log (day 19)"):
        lg = load_csv(EXPERIMENTS / "day19_experiment_log.csv")
        if lg is not None:
            st.dataframe(lg, hide_index=True)

    st.markdown('<div class="sec-label" style="margin-top:1rem">Reference thresholds used by the auditor</div>',
                unsafe_allow_html=True)
    ref = pd.DataFrame([{"Parameter": s.label, "Unit": s.unit or "pH units", "Band": s.ref_text,
                         "Basis": s.source, "Slider range": f"{s.lo:g} \u2013 {s.hi:g}"} for s in SENSOR_META])
    st.dataframe(ref, hide_index=True)
    st.markdown('<div class="notice"><b>About these limits.</b> They are commonly cited WHO / US-EPA guideline values (several are '
                'aesthetic or operational, not health-based) and are editable in <code>src/aquaguard.py</code>. Confirm against the '
                'standard that applies in your jurisdiction. Many dataset readings (e.g. dissolved solids around 22,000 mg/L) exceed '
                'drinking-water limits by a wide margin, so most rows show several violations.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warnbox"><b>Known limitations.</b> Timestamps were generated synthetically; temperature, dissolved '
                'oxygen and pressure are absent from the dataset; there are no ground-truth anomaly labels, so anomaly rate is '
                'descriptive, not an accuracy score.</div>', unsafe_allow_html=True)

st.caption("AquaSense AI \u00b7 Underwater Water-Quality AI Analytics \u00b7 Academic research prototype. "
           "Validate data quality, preprocessing and sensor calibration before any operational use.")
