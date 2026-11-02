from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import run_inference  # noqa: E402

st.set_page_config(
    page_title="AquaSense | Water Quality AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#e8f3ff; --muted:#92abc2; --panel:#0e2236; --line:rgba(153,197,226,.16); --cyan:#45d6e8; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background: radial-gradient(ellipse at 85% 0%, #123a55 0%, #071725 42%, #06111d 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0c2235,#081522); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:#dcecff; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
    h1 { font-weight:800 !important; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; color:var(--cyan); font-weight:700; }
    .hero { padding:1.5rem 1.7rem; border:1px solid var(--line); border-radius:22px; background:linear-gradient(120deg,rgba(22,66,91,.88),rgba(12,31,49,.8)); margin-bottom:1.25rem; }
    .hero p { color:#a9c2d7; margin-bottom:0; max-width:760px; }
    .metric { border:1px solid var(--line); border-radius:17px; padding:1rem 1.1rem; background:linear-gradient(145deg,rgba(17,44,66,.95),rgba(10,27,43,.92)); min-height:118px; }
    .metric-label { color:#9ab5cb; font-size:.82rem; margin-bottom:.45rem; }
    .metric-value { color:#f0f8ff; font-size:1.75rem; font-weight:800; font-family:'Manrope',sans-serif; line-height:1.2; }
    .metric-note { color:#6f9ab5; font-size:.75rem; margin-top:.35rem; }
    .section-label { color:var(--cyan); font-size:.75rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; margin:1rem 0 .5rem; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .notice { border-left:3px solid #45d6e8; padding:.75rem 1rem; background:rgba(25,76,99,.25); border-radius:0 10px 10px 0; color:#b7ccdc; font-size:.9rem; }
    .stButton button, .stDownloadButton button { border-radius:10px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_PATH = ROOT / "data" / "processed" / "test_v1_engineered.csv"
PRECOMPUTED_PATH = ROOT / "experiments" / "day26_inference_output.csv"

@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)

@st.cache_data(show_spinner=False)
def load_precomputed() -> pd.DataFrame:
    return pd.read_csv(PRECOMPUTED_PATH)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def chart_line(df: pd.DataFrame, x: str, y: str, title: str, color: str = "#45d6e8") -> None:
    fig, ax = plt.subplots(figsize=(10, 3.3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#0b1d2d")
    ax.plot(df[x], df[y], color=color, linewidth=1.8)
    ax.set_title(title, color="#e8f3ff", loc="left", pad=14, fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8da9bf", labelsize=8)
    ax.grid(True, alpha=.15, linestyle="--")
    for spine in ax.spines.values(): spine.set_color("#25445a")
    ax.xaxis.label.set_color("#8da9bf"); ax.yaxis.label.set_color("#8da9bf")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def normalise_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.sort_values("timestamp", na_position="last")
    return out

# ---------- Sidebar ----------
st.sidebar.markdown("# 🌊 AquaSense")
st.sidebar.caption("WATER QUALITY · AI ANALYTICS")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Explore Data", "Anomaly Detection", "pH Forecasting", "Results & Export"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
st.sidebar.success("Isolation Forest · Ready")
st.sidebar.success("Random Forest · Ready")
st.sidebar.caption("Research prototype · Not a certified water-safety system")

# ---------- Header ----------
st.markdown('<div class="eyebrow">UNDERWATER WATER-QUALITY AI ANALYTICS</div>', unsafe_allow_html=True)
st.title(page)
st.markdown(
    '<div class="hero"><div class="eyebrow">MONITOR · DETECT · FORECAST</div>'
    '<h2 style="margin:.35rem 0 .55rem 0">Understand water quality, one observation at a time.</h2>'
    '<p>Explore prepared water-quality observations, flag unusual records with Isolation Forest, and inspect next-step pH estimates from the frozen Random Forest model.</p></div>',
    unsafe_allow_html=True,
)

# ---------- Data source and inference ----------
with st.sidebar.expander("Data source", expanded=False):
    source_mode = st.radio("Choose data", ["Bundled example results", "Run models on CSV"], index=0)
    uploaded = None
    if source_mode == "Run models on CSV":
        uploaded = st.file_uploader("Upload prepared CSV", type=["csv"], help="CSV must include timestamp and the required water-quality feature columns, in the same scaling convention as the training data.")

if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "results_source" not in st.session_state:
    st.session_state.results_source = "Bundled example results"

if source_mode == "Bundled example results":
    try:
        results = load_precomputed()
        st.session_state.results_df = results
        st.session_state.results_source = "Bundled example results"
    except Exception as exc:
        st.error(f"Could not load the bundled example results: {exc}")
        st.stop()
else:
    if uploaded is not None:
        st.caption(f"Selected file: {uploaded.name} · {uploaded.size / 1024:.1f} KB")
        if st.button("Run anomaly detection + pH forecast", type="primary", use_container_width=False):
            with st.spinner("Running the bundled preprocessing and model pipeline…"):
                try:
                    raw_bytes = uploaded.getvalue()
                    with tempfile.TemporaryDirectory() as tmpdir:
                        input_file = Path(tmpdir) / "uploaded_input.csv"
                        output_file = Path(tmpdir) / "inference_output.csv"
                        input_file.write_bytes(raw_bytes)
                        results = run_inference(input_file, output_file)
                    st.session_state.results_df = results
                    st.session_state.results_source = uploaded.name
                    st.success(f"Inference completed for {len(results):,} rows.")
                except Exception as exc:
                    st.error(f"Inference could not run: {exc}")
                    st.info("Check that the CSV has a timestamp column and all required feature columns, with the same preprocessing/scaling convention as the training data.")
    elif st.session_state.results_df is None:
        st.info("Upload a prepared CSV to run inference, or switch to bundled example results.")

results = st.session_state.results_df
if results is None:
    st.stop()
results = normalise_for_display(results)

anomaly_col = "anomaly_flag" in results.columns
forecast_col = "predicted_next_ph" in results.columns
anomaly_count = int(pd.to_numeric(results["anomaly_flag"], errors="coerce").fillna(0).sum()) if anomaly_col else 0
anomaly_rate = anomaly_count / max(len(results), 1) * 100
ph_avg = pd.to_numeric(results.get("ph", pd.Series(dtype=float)), errors="coerce").mean()
forecast_avg = pd.to_numeric(results.get("predicted_next_ph", pd.Series(dtype=float)), errors="coerce").mean()

if page == "Overview":
    st.markdown('<div class="section-label">AT A GLANCE</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]: metric_card("Observations", f"{len(results):,}", "Rows in active dataset")
    with cols[1]: metric_card("Flagged anomalies", f"{anomaly_count:,}", f"{anomaly_rate:.2f}% of observations")
    with cols[2]: metric_card("Mean pH feature", f"{ph_avg:.3f}" if pd.notna(ph_avg) else "N/A", "Dataset scale; check preprocessing")
    with cols[3]: metric_card("Mean predicted next pH", f"{forecast_avg:.3f}" if pd.notna(forecast_avg) else "N/A", "Model estimate, not a safety verdict")

    left, right = st.columns([1.55, 1])
    with left:
        if "timestamp" in results.columns and "ph" in results.columns:
            chart_line(results.tail(180), "timestamp", "ph", "pH FEATURE · RECENT OBSERVATIONS")
        else:
            st.info("The dataset needs timestamp and ph columns to draw the pH trend.")
    with right:
        st.markdown('<div class="section-label">MODEL SNAPSHOT</div>', unsafe_allow_html=True)
        if anomaly_col:
            counts = results["anomaly_flag"].value_counts().reindex([0, 1], fill_value=0)
            st.bar_chart(pd.DataFrame({"Observations": counts.values}, index=["Normal", "Flagged"]), color="#45d6e8")
        st.markdown('<div class="notice">An anomaly flag means a record is unusual relative to the fitted model. It does not automatically mean the water is unsafe.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">LATEST RECORDS</div>', unsafe_allow_html=True)
    st.dataframe(results.tail(8), use_container_width=True, hide_index=True)

elif page == "Explore Data":
    st.markdown('<div class="section-label">DATASET PROFILE</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a: metric_card("Rows", f"{len(results):,}", "Current result dataset")
    with b: metric_card("Columns", f"{results.shape[1]:,}", "Original + engineered + model outputs")
    with c: metric_card("Missing cells", f"{int(results.isna().sum().sum()):,}", "Across all columns")
    st.markdown("#### Preview records")
    st.dataframe(results.head(100), use_container_width=True, hide_index=True)
    st.markdown("#### Missing values by column")
    missing = results.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing cells found in the active results table.")
    else:
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)
    numeric = results.select_dtypes(include="number")
    if not numeric.empty:
        st.markdown("#### Numeric summary")
        st.dataframe(numeric.describe().T.round(4), use_container_width=True)

elif page == "Anomaly Detection":
    st.markdown('<div class="section-label">ISOLATION FOREST OUTPUT</div>', unsafe_allow_html=True)
    if not anomaly_col:
        st.warning("The active dataset has no anomaly_flag output. Run inference or load bundled results.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Flagged records", f"{anomaly_count:,}", "anomaly_flag = 1")
        with c2: metric_card("Flag rate", f"{anomaly_rate:.2f}%", "Share of current dataset")
        score_mean = pd.to_numeric(results.get("anomaly_score", pd.Series(dtype=float)), errors="coerce").mean()
        with c3: metric_card("Mean decision score", f"{score_mean:.4f}" if pd.notna(score_mean) else "N/A", "Model decision_function output")
        plot_df = results.copy()
        if "timestamp" in plot_df.columns and "ph" in plot_df.columns:
            fig, ax = plt.subplots(figsize=(11, 4))
            fig.patch.set_alpha(0); ax.set_facecolor("#0b1d2d")
            normal = plot_df[plot_df["anomaly_flag"] == 0]
            flagged = plot_df[plot_df["anomaly_flag"] == 1]
            ax.plot(plot_df["timestamp"], plot_df["ph"], color="#4aa9c5", alpha=.7, linewidth=1, label="pH feature")
            ax.scatter(normal["timestamp"], normal["ph"], s=13, color="#45d6e8", label="Not flagged")
            ax.scatter(flagged["timestamp"], flagged["ph"], s=30, color="#ff9b71", label="Flagged anomaly", zorder=3)
            ax.set_title("Observations flagged by the model", color="#e8f3ff", loc="left", pad=12)
            ax.tick_params(colors="#8da9bf", labelsize=8); ax.grid(True, alpha=.13)
            for spine in ax.spines.values(): spine.set_color("#25445a")
            ax.legend(facecolor="#10283b", labelcolor="#e8f3ff", frameon=False)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.info("Add timestamp and ph columns to display anomaly positions on a time trend.")
        show_flagged = st.toggle("Show flagged records only", value=True)
        view = results[results["anomaly_flag"] == 1] if show_flagged else results
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.markdown('<div class="notice">Interpretation note: the current dataset has no ground-truth anomaly labels. Flag rate is descriptive and is not an accuracy metric.</div>', unsafe_allow_html=True)

elif page == "pH Forecasting":
    st.markdown('<div class="section-label">RANDOM FOREST · NEXT-STEP ESTIMATE</div>', unsafe_allow_html=True)
    if not forecast_col:
        st.warning("The active dataset has no predicted_next_ph column. Run inference or load bundled results.")
    else:
        vals = pd.to_numeric(results["predicted_next_ph"], errors="coerce").dropna()
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Mean estimate", f"{vals.mean():.3f}" if len(vals) else "N/A", "predicted_next_ph")
        with c2: metric_card("Minimum estimate", f"{vals.min():.3f}" if len(vals) else "N/A", "In current result set")
        with c3: metric_card("Maximum estimate", f"{vals.max():.3f}" if len(vals) else "N/A", "In current result set")
        if "timestamp" in results.columns:
            chart_line(results.tail(180), "timestamp", "predicted_next_ph", "PREDICTED NEXT-STEP pH · RECENT ROWS", "#8b9cff")
        if "ph" in results.columns:
            st.markdown("#### Feature pH vs predicted next-step pH")
            compare_cols = [c for c in ["timestamp", "ph", "predicted_next_ph"] if c in results.columns]
            st.dataframe(results[compare_cols].tail(100), use_container_width=True, hide_index=True)
        st.markdown('<div class="notice">These are model estimates generated from prepared features. They are not direct sensor readings, and a one-step prediction is not a long-horizon forecast.</div>', unsafe_allow_html=True)

elif page == "Results & Export":
    st.markdown('<div class="section-label">RESULTS TABLE</div>', unsafe_allow_html=True)
    st.write(f"Active source: **{st.session_state.results_source}**")
    search = st.text_input("Filter rows by text", placeholder="Type a value to filter visible rows…")
    display_df = results
    if search.strip():
        mask = display_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download full results CSV", data=csv_bytes, file_name="water_quality_ai_results.csv", mime="text/csv", type="primary")
    st.markdown("#### Reference validation metrics")
    st.caption("These are saved experiment results from the project validation, not metrics recomputed on the currently uploaded file.")
    metrics_path = ROOT / "experiments" / "day23_final_validation_results" / "forecast_performance.csv"
    anomaly_metrics_path = ROOT / "experiments" / "day23_final_validation_results" / "anomaly_performance.csv"
    if metrics_path.exists():
        st.write("**pH forecasting**")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
    if anomaly_metrics_path.exists():
        st.write("**Anomaly detection**")
        st.dataframe(pd.read_csv(anomaly_metrics_path), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("AquaSense · Underwater Water-Quality AI Analytics · Academic/research prototype. Validate data quality, preprocessing, and sensor calibration before operational use.")
