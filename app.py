import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, leaves_list

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    layout="wide",
    page_title="Analysis Dashboard",
    page_icon="📊"
)

# ============================
# GLOBAL STYLES — LIGHT THEME
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* Base — light */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #f4f6fa;
    color: #1a2540;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 100%; }

/* ── Header Banner ── */
.dash-header {
    background: linear-gradient(135deg, #1a3a6e 0%, #1e4d8c 60%, #1565c0 100%);
    border: 1px solid #1a5cb5;
    border-radius: 12px;
    padding: 28px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.04em;
    margin: 0;
}
.dash-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #90caf9;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #e8edf5;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #c5d0e0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #4a5e80;
    background: transparent;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1565c0 !important;
    border: 1px solid #c5d0e0 !important;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #d0daea;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-card:hover { border-color: #1565c0; }
.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1565c0, #42a5f5);
    opacity: 0.7;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5a6e8a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a2540;
}
.metric-sub {
    font-size: 0.7rem;
    color: #5a6e8a;
    margin-top: 4px;
}

/* ── Change Cards ── */
.change-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 18px 0;
}
.change-card {
    background: #ffffff;
    border: 1px solid #d0daea;
    border-radius: 10px;
    padding: 22px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.change-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #1565c0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
    border-bottom: 1px solid #d0daea;
    padding-bottom: 10px;
}
.change-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #eef1f7;
}
.change-row:last-child { border-bottom: none; }
.change-key { font-size: 0.72rem; color: #4a5e80; }
.change-val { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; }
.pos { color: #1a7a4a; }
.neg { color: #c0392b; }
.neu { color: #1a2540; }

/* ── Insight Block ── */
.insight-block {
    background: #f0f5ff;
    border: 1px solid #c5d0e0;
    border-left: 4px solid #1565c0;
    border-radius: 10px;
    padding: 24px 28px;
    margin-top: 18px;
}
.insight-heading {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #1565c0;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.insight-text {
    font-size: 0.82rem;
    color: #2a3a55;
    line-height: 1.7;
}
.insight-text b { color: #1a2540; }

/* ── Section Labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #1565c0;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0 0 10px 0;
    border-bottom: 1px solid #d0daea;
    margin-bottom: 14px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #eef1f7;
    border-right: 1px solid #d0daea;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Selectbox / Slider ── */
.stSelectbox > div > div, .stSlider > div {
    background: #ffffff !important;
    border-color: #c5d0e0 !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stSlider label {
    font-size: 0.7rem !important;
    color: #4a5e80 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #d0daea;
    border-radius: 8px;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Axis scale panel ── */
.axis-panel {
    background: #ffffff;
    border: 1px solid #d0daea;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.axis-panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    color: #1565c0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid #e0e8f5;
    padding-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("""
<div class="dash-header">
    <div class="dash-title">TREND ANALYSIS DASHBOARD</div>
    <div class="dash-subtitle">Dynamic Variable Comparison &nbsp;·&nbsp; Trend Analysis &nbsp;·&nbsp; Correlation Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ============================
# SIDEBAR UPLOAD
# ============================
with st.sidebar:
    st.markdown('<div class="section-label">DATA SOURCE</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.markdown('<div class="section-label">ABOUT</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem;color:#4a5e80;line-height:1.6;">Upload an Excel file with the first column as Date and subsequent columns as numeric variables.</p>', unsafe_allow_html=True)

# ============================
# MAIN LOGIC
# ============================
if file:
    df = pd.read_excel(file, header=0)
    df = df.dropna(how="all")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=df.columns[1:], how="all")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    all_cols = list(df.columns)
    def labeled(col):
        idx = all_cols.index(col)
        return f"{idx}-{col}"

    numeric_labeled = [labeled(c) for c in numeric_cols]
    label_to_col = {labeled(c): c for c in numeric_cols}

    # ============================
    # TABS
    # ============================
    tab1, tab2, tab3 = st.tabs(["⚙️  Preprocess", "📊  Correlation", "📈  Analysis"])

    # ── TAB 1: PREPROCESS ──
    with tab1:
        st.markdown('<div class="section-label">DATA CLEANING & OUTLIER TREATMENT</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            lower = st.slider("Lower Percentile Cutoff", 0, 50, 5)
        with col2:
            upper = st.slider("Upper Percentile Cutoff", 50, 100, 95)

        df_clean = df.copy()

        if st.button("Apply Outlier Treatment"):
            for col in numeric_cols:
                series = df_clean[col].dropna()
                if len(series) > 0:
                    low = np.percentile(series, lower)
                    high = np.percentile(series, upper)
                    df_clean = df_clean[
                        (df_clean[col].isna()) |
                        ((df_clean[col] >= low) & (df_clean[col] <= high))
                    ]
            st.success(f"✅ Outlier treatment applied — rows retained: {len(df_clean)}")

        st.markdown('<div class="section-label" style="margin-top:18px;">CLEANED DATA PREVIEW</div>', unsafe_allow_html=True)
        st.dataframe(df_clean, use_container_width=True, hide_index=True)

    # ── TAB 2: CORRELATION ──
    with tab2:
        st.markdown('<div class="section-label">CORRELATION INTELLIGENCE</div>', unsafe_allow_html=True)

        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for correlation analysis.")
            st.stop()

        corr = df[numeric_cols].corr()

        st.markdown('<div class="section-label" style="margin-top:6px;">FULL CORRELATION MATRIX</div>', unsafe_allow_html=True)
        fig_full = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont={"size": 11, "color": "#1a2540"}
        ))
        fig_full.update_layout(
            height=850, margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor="#f4f6fa", plot_bgcolor="#f4f6fa",
            font=dict(color="#1a2540"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#1a2540")),
            yaxis=dict(tickfont=dict(size=10, color="#1a2540"))
        )
        st.plotly_chart(fig_full, use_container_width=True, config={"scrollZoom": True})

        st.markdown('<div class="section-label" style="margin-top:18px;">FOCUSED CORRELATION — TOP VARIABLES</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            target_lbl = st.selectbox("Select Target Variable", numeric_labeled)
            target_col = label_to_col[target_lbl]
        with c2:
            top_n = st.slider("Top N Variables", 5, 25, 15)

        corr_target = corr[target_col].abs().sort_values(ascending=False).head(top_n).index
        corr_filtered = corr.loc[corr_target, corr_target]

        fig_top = go.Figure(data=go.Heatmap(
            z=corr_filtered.values, x=corr_filtered.columns, y=corr_filtered.columns,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=np.round(corr_filtered.values, 2),
            texttemplate="%{text}", textfont={"size": 13, "color": "#1a2540"}
        ))
        fig_top.update_layout(
            height=640, margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor="#f4f6fa", plot_bgcolor="#f4f6fa",
            font=dict(color="#1a2540"),
            xaxis=dict(tickangle=-45, tickfont=dict(color="#1a2540")),
            yaxis=dict(tickfont=dict(color="#1a2540"))
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:18px;">CLUSTERED CORRELATION — HIERARCHICAL GROUPING</div>', unsafe_allow_html=True)
        try:
            linked = linkage(corr, method='ward')
            order = leaves_list(linked)
            corr_clustered = corr.iloc[order, order]

            fig_cluster = go.Figure(data=go.Heatmap(
                z=corr_clustered.values, x=corr_clustered.columns, y=corr_clustered.columns,
                colorscale="RdYlGn", zmin=-1, zmax=1,
                text=np.round(corr_clustered.values, 2),
                texttemplate="%{text}", textfont={"size": 11, "color": "#1a2540"}
            ))
            fig_cluster.update_layout(
                height=850, margin=dict(l=40, r=40, t=30, b=40),
                paper_bgcolor="#f4f6fa", plot_bgcolor="#f4f6fa",
                font=dict(color="#1a2540"),
                xaxis=dict(tickangle=-45, tickfont=dict(color="#1a2540")),
                yaxis=dict(tickfont=dict(color="#1a2540"))
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.success("✅ Variables automatically grouped by similarity")
        except Exception:
            st.warning("Clustering unavailable. Ensure scipy is installed.")

    # ── TAB 3: ANALYSIS ──
    with tab3:
        st.markdown('<div class="section-label">DYNAMIC VARIABLE ANALYSIS</div>', unsafe_allow_html=True)

        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns.")
            st.stop()

        date_list = sorted(df[date_col].dt.date.unique())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            start_date = st.selectbox("Start Date", date_list, key="sd")
        with c2:
            end_date = st.selectbox("End Date", date_list, index=len(date_list)-1, key="ed")
        with c3:
            primary_lbl = st.selectbox("Primary Variable", numeric_labeled, key="pv")
            primary = label_to_col[primary_lbl]
        with c4:
            secondary_lbl = st.selectbox("Secondary Variable", numeric_labeled, index=1, key="sv")
            secondary = label_to_col[secondary_lbl]

        if start_date > end_date:
            st.error("⚠ Start Date cannot be after End Date.")
            st.stop()

        df_f = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]

        if len(df_f) < 2:
            st.warning("Not enough data in selected date range.")
            st.stop()

        x = df_f[primary]
        y = df_f[secondary]

        # ── Regression ──
        slope, intercept, r_val, p, std_err = linregress(x.dropna(), y.dropna())
        r2 = r_val**2
        n = len(x)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
        corr_val = x.corr(y)

        def relation_strength(c):
            if abs(c) < 0.3: return "Weak"
            elif abs(c) < 0.7: return "Moderate"
            else: return "Strong"

        strength = relation_strength(corr_val)
        relation = "Positive" if corr_val > 0 else "Negative"

        def pct_change_series(series):
            s = series.dropna()
            if len(s) < 2:
                return None, None, None, None
            start_val = s.iloc[0]
            end_val   = s.iloc[-1]
            if start_val == 0:
                return start_val, end_val, None, None
            chg = ((end_val - start_val) / abs(start_val)) * 100
            period_pct = series.pct_change() * 100
            avg_period_chg = period_pct.mean()
            return start_val, end_val, chg, avg_period_chg

        px_start, px_end, px_chg, px_avg = pct_change_series(x)
        py_start, py_end, py_chg, py_avg = pct_change_series(y)

        def fmt_chg(v):
            if v is None: return "N/A", "neu"
            sign = "+" if v >= 0 else ""
            cls  = "pos" if v >= 0 else "neg"
            return f"{sign}{v:.2f}%", cls

        px_chg_str, px_cls   = fmt_chg(px_chg)
        px_avg_str, px_avg_cls = fmt_chg(px_avg)
        py_chg_str, py_cls   = fmt_chg(py_chg)
        py_avg_str, py_avg_cls = fmt_chg(py_avg)

        # ── TOP METRICS ROW ──
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Correlation Coefficient</div>
                <div class="metric-value">{corr_val:.3f}</div>
                <div class="metric-sub">{strength} {relation}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{r2:.4f}</div>
                <div class="metric-sub">Adjusted R² {adj_r2:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Regression Slope</div>
                <div class="metric-value">{slope:.4f}</div>
                <div class="metric-sub">Std Error {std_err:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Data Points</div>
                <div class="metric-value">{n}</div>
                <div class="metric-sub">{start_date} → {end_date}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── % CHANGE CARDS ──
        st.markdown('<div class="section-label" style="margin-top:8px;">PERIOD CHANGE ANALYTICS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="change-grid">
            <div class="change-card">
                <div class="change-card-title">📌 {primary} — Period Performance</div>
                <div class="change-row">
                    <span class="change-key">Opening Value</span>
                    <span class="change-val neu">{px_start:.4g}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Closing Value</span>
                    <span class="change-val neu">{px_end:.4g}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Total % Change</span>
                    <span class="change-val {px_cls}">{px_chg_str}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Avg Period-on-Period Change</span>
                    <span class="change-val {px_avg_cls}">{px_avg_str}</span>
                </div>
            </div>
            <div class="change-card">
                <div class="change-card-title">📌 {secondary} — Period Performance</div>
                <div class="change-row">
                    <span class="change-key">Opening Value</span>
                    <span class="change-val neu">{py_start:.4g}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Closing Value</span>
                    <span class="change-val neu">{py_end:.4g}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Total % Change</span>
                    <span class="change-val {py_cls}">{py_chg_str}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Avg Period-on-Period Change</span>
                    <span class="change-val {py_avg_cls}">{py_avg_str}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── AXIS SCALE CUSTOMIZATION ──
        st.markdown('<div class="section-label" style="margin-top:8px;">CHART AXIS SCALE CUSTOMIZATION</div>', unsafe_allow_html=True)

        x_min_data = float(x.min()) if not x.isna().all() else 0.0
        x_max_data = float(x.max()) if not x.isna().all() else 100.0
        y_min_data = float(y.min()) if not y.isna().all() else 0.0
        y_max_data = float(y.max()) if not y.isna().all() else 100.0

        x_padding = (x_max_data - x_min_data) * 0.1 if x_max_data != x_min_data else abs(x_max_data) * 0.1 or 10
        y_padding = (y_max_data - y_min_data) * 0.1 if y_max_data != y_min_data else abs(y_max_data) * 0.1 or 10

        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        with ac1:
            y1_min = st.number_input(
                f"Y-Left Min ({primary[:12]}...)" if len(primary) > 12 else f"Y-Left Min ({primary})",
                value=round(x_min_data - x_padding, 4), format="%.4f", key="y1min"
            )
        with ac2:
            y1_max = st.number_input(
                f"Y-Left Max ({primary[:12]}...)" if len(primary) > 12 else f"Y-Left Max ({primary})",
                value=round(x_max_data + x_padding, 4), format="%.4f", key="y1max"
            )
        with ac3:
            y2_min = st.number_input(
                f"Y-Right Min ({secondary[:12]}...)" if len(secondary) > 12 else f"Y-Right Min ({secondary})",
                value=round(y_min_data - y_padding, 4), format="%.4f", key="y2min"
            )
        with ac4:
            y2_max = st.number_input(
                f"Y-Right Max ({secondary[:12]}...)" if len(secondary) > 12 else f"Y-Right Max ({secondary})",
                value=round(y_max_data + y_padding, 4), format="%.4f", key="y2max"
            )
        with ac5:
            auto_scale = st.checkbox("Auto Scale", value=True, key="autoscale")

        # ── CHART ──
        st.markdown('<div class="section-label" style="margin-top:8px;">TIME SERIES & TREND</div>', unsafe_allow_html=True)

        PRIMARY_COLOR   = "#1a6fba"   # blue  – primary axis
        SECONDARY_COLOR = "#c0392b"   # red   – secondary axis

        fig = go.Figure()

        # Primary smooth line — NO fill
        fig.add_trace(go.Scatter(
            x=df_f[date_col], y=x,
            name=primary,
            mode="lines",
            line=dict(width=2.5, color=PRIMARY_COLOR, shape="spline", smoothing=1.3),
        ))

        # Primary exponential trendline — solid, smooth, same blue, full range with 500 dense points, hidden from legend
        try:
            df_trend_p = df_f[[date_col, primary]].dropna()
            df_trend_p = df_trend_p[df_trend_p[primary] > 0].reset_index(drop=True)
            t_fit = np.arange(len(df_trend_p))
            y_log = np.log(df_trend_p[primary].values)
            coef  = np.polyfit(t_fit, y_log, 1)

            # Generate 500 evenly-spaced points across full date range → no gaps
            date_start_ns = df_trend_p[date_col].iloc[0].value
            date_end_ns   = df_trend_p[date_col].iloc[-1].value
            dense_dates   = pd.to_datetime(np.linspace(date_start_ns, date_end_ns, 500))
            t_dense       = np.linspace(0, len(df_trend_p) - 1, 500)
            trend_dense   = np.exp(coef[1]) * np.exp(coef[0] * t_dense)

            fig.add_trace(go.Scatter(
                x=dense_dates, y=trend_dense,
                name=f"Trend ({primary})",
                mode="lines",
                showlegend=False,
                line=dict(width=2, color=PRIMARY_COLOR, shape="spline", smoothing=1.3)
            ))
        except Exception:
            pass

        # Secondary smooth line — NO fill, NO trendline
        fig.add_trace(go.Scatter(
            x=df_f[date_col], y=y,
            name=secondary,
            yaxis="y2",
            mode="lines",
            line=dict(width=2.5, color=SECONDARY_COLOR, shape="spline", smoothing=1.3),
        ))

        # Build axis range config
        if auto_scale:
            yaxis_range  = None
            yaxis2_range = None
        else:
            yaxis_range  = [y1_min, y1_max]
            yaxis2_range = [y2_min, y2_max]

        fig.update_layout(
            height=520,
            title=dict(
                text=f"<b>{primary}</b>  vs  <b>{secondary}</b>",
                font=dict(family="Syne, sans-serif", size=15, color="#1a2540"),
                x=0.5, xanchor="center"
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="DM Mono, monospace", color="#1a2540", size=11),
            legend=dict(
                bgcolor="#f8f9fc", bordercolor="#d0daea", borderwidth=1,
                orientation="h", x=0.5, xanchor="center", y=-0.22,
                font=dict(size=12, color="#1a2540")
            ),
            yaxis=dict(
                title=dict(text=f"<b>{primary}</b>", font=dict(color=PRIMARY_COLOR, size=12)),
                tickfont=dict(color=PRIMARY_COLOR, size=11),
                gridcolor="#e8edf5",
                gridwidth=1,
                griddash="dash",
                zeroline=True,
                zerolinecolor="#c5d0e0",
                showline=True,
                linecolor="#c5d0e0",
                range=yaxis_range
            ),
            yaxis2=dict(
                title=dict(text=f"<b>{secondary}</b>", font=dict(color=SECONDARY_COLOR, size=12)),
                tickfont=dict(color=SECONDARY_COLOR, size=11),
                overlaying="y", side="right",
                gridcolor="#e8edf5",
                zeroline=False,
                showline=True,
                linecolor="#c5d0e0",
                range=yaxis2_range
            ),
            xaxis=dict(
                gridcolor="#e8edf5",
                gridwidth=1,
                griddash="dash",
                zeroline=False,
                showline=True,
                linecolor="#c5d0e0",
                tickangle=-45,
                tickfont=dict(color="#2a3a55", size=10)
            ),
            margin=dict(l=70, r=80, t=60, b=90)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── ANALYTICS PANEL ──
        st.markdown('<div class="section-label" style="margin-top:4px;">VARIABLE STATISTICS & REGRESSION SUMMARY</div>', unsafe_allow_html=True)

        x_min_s   = f"{x.min():.7g}"
        x_max_s   = f"{x.max():.7g}"
        x_mean_s  = f"{x.mean():.7g}"
        x_std_s   = f"{x.std():.7g}"

        y_min_s   = f"{y.min():.7g}"
        y_max_s   = f"{y.max():.7g}"
        y_mean_s  = f"{y.mean():.7g}"
        y_std_s   = f"{y.std():.7g}"

        corr_impact   = "Positive" if corr_val > 0 else "Negative"
        corr_val_s    = f"{corr_val:.7f}"
        corr_val_col  = "#c0392b" if corr_val < 0 else "#1a7a4a"

        slope_color   = "#1a7a4a" if slope    >= 0   else "#c0392b"
        r2_color      = "#1a7a4a" if r2       >= 0.7 else ("#d97706" if r2 >= 0.4 else "#c0392b")

        slope_s   = f"{slope:.3f}"
        r2_s      = f"{r2:.3f}"
        adj_r2_s  = f"{adj_r2:.3f}"
        stderr_s  = f"{std_err:.4f}"
        rel_s     = f"{strength} | {relation}"

        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
                f'<div style="background:#e8f0fe;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#1565c0;letter-spacing:0.1em;text-transform:uppercase;">{primary}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#f4f7fc;">'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Min</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Max</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Average</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Standard Deviation</th>'
                f'</tr></thead>'
                f'<tbody><tr>'
                f'<td style="padding:10px 14px;color:#1a2540;">{x_min_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{x_max_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{x_mean_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{x_std_s}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
                f'<div style="background:#fce8e6;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#c0392b;letter-spacing:0.1em;text-transform:uppercase;">{secondary}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#fdf4f4;">'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Min</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Max</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Average</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Standard Deviation</th>'
                f'</tr></thead>'
                f'<tbody><tr>'
                f'<td style="padding:10px 14px;color:#1a2540;">{y_min_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{y_max_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{y_mean_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{y_std_s}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
                f'<div style="background:#fffbe6;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#b45309;letter-spacing:0.1em;text-transform:uppercase;">Correlation — Primary vs Secondary</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#fdfaf0;">'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Value</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Impact</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Level</th>'
                f'</tr></thead>'
                f'<tbody><tr>'
                f'<td style="padding:10px 14px;color:{corr_val_col};font-weight:700;">{corr_val_s}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{corr_impact}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{strength}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

        with right_col:
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;padding:0;box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
                f'<div style="background:#fffbe6;padding:10px 16px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#b45309;letter-spacing:0.1em;text-transform:uppercase;">Regression Summary</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #eef1f7;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5e80;letter-spacing:0.05em;">SLOPE</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{slope_color};">{slope_s}</span>'
                f'</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #eef1f7;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5e80;letter-spacing:0.05em;">R²</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_color};">{r2_s}</span>'
                f'</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5e80;letter-spacing:0.05em;">ADJ. R²</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_color};">{adj_r2_s}</span>'
                f'</div>'

                f'</div>',
                unsafe_allow_html=True
            )

        # ── KEY INSIGHTS ──
        st.markdown('<div class="section-label" style="margin-top:10px;">KEY INSIGHTS</div>', unsafe_allow_html=True)

        trend_dir_p = "increased" if (px_chg or 0) >= 0 else "decreased"
        trend_dir_s = "increased" if (py_chg or 0) >= 0 else "decreased"

        st.markdown(f"""
        <div class="insight-block">
            <div class="insight-heading">Relationship</div>
            <div class="insight-text">
                There is a <b>{strength} {relation}</b> relationship between <b>{primary}</b> and <b>{secondary}</b>,
                with a correlation coefficient of <b>{corr_val:.3f}</b> over the selected period
                (<b>{start_date}</b> to <b>{end_date}</b>).
            </div>
            <div class="insight-heading" style="margin-top:18px;">Period Change Summary</div>
            <div class="insight-text">
                <b>{primary}</b> {trend_dir_p} by <b>{px_chg_str}</b> overall during this period,
                moving from <b>{px_start:.4g}</b> to <b>{px_end:.4g}</b>,
                with an average period-on-period change of <b>{px_avg_str}</b>.<br><br>
                <b>{secondary}</b> {trend_dir_s} by <b>{py_chg_str}</b> overall during this period,
                moving from <b>{py_start:.4g}</b> to <b>{py_end:.4g}</b>,
                with an average period-on-period change of <b>{py_avg_str}</b>.
            </div>
            <div class="insight-heading" style="margin-top:18px;">Regression Interpretation</div>
            <div class="insight-text">
                The regression slope of <b>{slope:.4f}</b> indicates that for every unit increase in <b>{primary}</b>,
                <b>{secondary}</b> changes by approximately <b>{slope:.4f}</b> units.
                The model explains <b>{r2*100:.1f}%</b> of variance (R² = <b>{r2:.4f}</b>,
                Adjusted R² = <b>{adj_r2:.4f}</b>).
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 80px 40px;
        background: #ffffff;
        border: 1px dashed #c5d0e0;
        border-radius: 12px;
        margin-top: 40px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    ">
        <div style="font-size:3rem;margin-bottom:16px;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#1565c0;font-weight:700;letter-spacing:0.06em;">
            NO DATA LOADED
        </div>
        <div style="font-size:0.78rem;color:#4a5e80;margin-top:10px;line-height:1.7;">
            Upload an Excel (.xlsx) file using the sidebar panel.<br>
            First column should be a Date column. Remaining columns should be numeric variables.
        </div>
    </div>
    """, unsafe_allow_html=True)
