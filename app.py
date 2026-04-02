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
# GLOBAL STYLES
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0f14;
    color: #d4dbe8;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 100%; }

/* ── Header Banner ── */
.dash-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0a3055 100%);
    border: 1px solid #1e3a5f;
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
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #e2eaf4;
    letter-spacing: 0.04em;
    margin: 0;
}
.dash-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #111520;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #6b80a0;
    background: transparent;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1a2e4a !important;
    color: #38bdf8 !important;
    border: 1px solid #234b72 !important;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0;
}
.metric-card {
    background: #111520;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    opacity: 0.5;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a6080;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2eaf4;
}
.metric-sub {
    font-size: 0.7rem;
    color: #4a6080;
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
    background: #111520;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 22px 24px;
}
.change-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 10px;
}
.change-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #161d2c;
}
.change-row:last-child { border-bottom: none; }
.change-key { font-size: 0.72rem; color: #6b80a0; }
.change-val { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; }
.pos { color: #34d399; }
.neg { color: #f87171; }
.neu { color: #e2eaf4; }

/* ── Insight Block ── */
.insight-block {
    background: linear-gradient(135deg, #0d1b2a, #0f2035);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 24px 28px;
    margin-top: 18px;
}
.insight-heading {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.insight-text {
    font-size: 0.82rem;
    color: #a8bdd0;
    line-height: 1.7;
}
.insight-text b { color: #e2eaf4; }

/* ── Section Labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0 0 10px 0;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 14px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0d13;
    border-right: 1px solid #1a2535;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Selectbox / Slider ── */
.stSelectbox > div > div, .stSlider > div {
    background: #111520 !important;
    border-color: #1e2d45 !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stSlider label {
    font-size: 0.7rem !important;
    color: #4a6080 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #1e2d45;
    border-radius: 8px;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
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
    st.markdown('<p style="font-size:0.72rem;color:#4a6080;line-height:1.6;">Upload an Excel file with the first column as Date and subsequent columns as numeric variables.</p>', unsafe_allow_html=True)

# ============================
# MAIN LOGIC
# ============================
if file:
    # ── Load ──
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

    # ── Build numbered labels for dropdowns ──
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

    # ── TAB 1: PREPROCESS ──────────────────────────────────
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

    # ── TAB 2: CORRELATION ─────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">CORRELATION INTELLIGENCE</div>', unsafe_allow_html=True)

        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for correlation analysis.")
            st.stop()

        corr = df[numeric_cols].corr()

        # Full heatmap
        st.markdown('<div class="section-label" style="margin-top:6px;">FULL CORRELATION MATRIX</div>', unsafe_allow_html=True)
        fig_full = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont={"size": 11}
        ))
        fig_full.update_layout(
            height=850, margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
            font=dict(color="#d4dbe8"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig_full, use_container_width=True, config={"scrollZoom": True})

        # Top correlated
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
            texttemplate="%{text}", textfont={"size": 13}
        ))
        fig_top.update_layout(
            height=640, margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
            font=dict(color="#d4dbe8"),
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig_top, use_container_width=True)

        # Clustered
        st.markdown('<div class="section-label" style="margin-top:18px;">CLUSTERED CORRELATION — HIERARCHICAL GROUPING</div>', unsafe_allow_html=True)
        try:
            linked = linkage(corr, method='ward')
            order = leaves_list(linked)
            corr_clustered = corr.iloc[order, order]

            fig_cluster = go.Figure(data=go.Heatmap(
                z=corr_clustered.values, x=corr_clustered.columns, y=corr_clustered.columns,
                colorscale="RdYlGn", zmin=-1, zmax=1,
                text=np.round(corr_clustered.values, 2),
                texttemplate="%{text}", textfont={"size": 11}
            ))
            fig_cluster.update_layout(
                height=850, margin=dict(l=40, r=40, t=30, b=40),
                paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                font=dict(color="#d4dbe8"),
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.success("✅ Variables automatically grouped by similarity")
        except Exception:
            st.warning("Clustering unavailable. Ensure scipy is installed.")

    # ── TAB 3: ANALYSIS ────────────────────────────────────
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

        # ── % Change Calculations ──
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
                <div class="change-row">
                    <span class="change-key">Min / Max</span>
                    <span class="change-val neu">{x.min():.4g} / {x.max():.4g}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Std Deviation</span>
                    <span class="change-val neu">{x.std():.4g}</span>
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
                <div class="change-row">
                    <span class="change-key">Min / Max</span>
                    <span class="change-val neu">{y.min():.4g} / {y.max():.4g}</span>
                </div>
                <div class="change-row">
                    <span class="change-key">Std Deviation</span>
                    <span class="change-val neu">{y.std():.4g}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CHART — white background, shaded fills, colored axes like Image 2 ──
        st.markdown('<div class="section-label" style="margin-top:8px;">TIME SERIES & TREND</div>', unsafe_allow_html=True)

        PRIMARY_COLOR   = "#1a6fba"   # blue  – primary axis
        SECONDARY_COLOR = "#c0392b"   # red   – secondary axis
        TREND_COLOR     = "#e6b800"   # gold  – trendline

        fig = go.Figure()

        # Primary fill area
        fig.add_trace(go.Scatter(
            x=df_f[date_col], y=x,
            name=primary,
            mode="lines",
            line=dict(width=2.5, color=PRIMARY_COLOR),
            fill="tozeroy",
            fillcolor="rgba(26,111,186,0.10)"
        ))

        # Secondary fill area
        fig.add_trace(go.Scatter(
            x=df_f[date_col], y=y,
            name=secondary,
            yaxis="y2",
            mode="lines",
            line=dict(width=2.5, color=SECONDARY_COLOR),
            fill="tozeroy",
            fillcolor="rgba(192,57,43,0.10)"
        ))

        # Exponential trendline on primary (dashed gold, like Image 2)
        try:
            df_trend = df_f[[date_col, primary]].dropna()
            df_trend = df_trend[df_trend[primary] > 0]
            t = np.arange(len(df_trend))
            y_log = np.log(df_trend[primary])
            coef  = np.polyfit(t, y_log, 1)
            trend = np.exp(coef[1]) * np.exp(coef[0] * t)
            fig.add_trace(go.Scatter(
                x=df_trend[date_col], y=trend,
                name=f"Trend ({primary})",
                mode="lines",
                line=dict(dash="dash", width=2, color=TREND_COLOR)
            ))
        except Exception:
            pass

        fig.update_layout(
            height=500,
            title=dict(
                text=f"<b>{primary}</b>  vs  <b>{secondary}</b>",
                font=dict(family="Arial, sans-serif", size=15, color="#1a1a2e"),
                x=0.5, xanchor="center"
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Arial, sans-serif", color="#333333", size=11),
            legend=dict(
                bgcolor="#f8f8f8", bordercolor="#cccccc", borderwidth=1,
                orientation="h", x=0.5, xanchor="center", y=-0.18,
                font=dict(size=12, color="#333333")
            ),
            yaxis=dict(
                title=dict(text=f"<b>{primary}</b>", font=dict(color=PRIMARY_COLOR, size=12)),
                tickfont=dict(color=PRIMARY_COLOR),
                gridcolor="#e8e8e8",
                gridwidth=1,
                griddash="dash",
                zeroline=True,
                zerolinecolor="#cccccc",
                showline=True,
                linecolor="#cccccc"
            ),
            yaxis2=dict(
                title=dict(text=f"<b>{secondary}</b>", font=dict(color=SECONDARY_COLOR, size=12)),
                tickfont=dict(color=SECONDARY_COLOR),
                overlaying="y", side="right",
                gridcolor="#e8e8e8",
                zeroline=False,
                showline=True,
                linecolor="#cccccc"
            ),
            xaxis=dict(
                gridcolor="#e8e8e8",
                gridwidth=1,
                griddash="dash",
                zeroline=False,
                showline=True,
                linecolor="#cccccc",
                tickangle=-45
            ),
            margin=dict(l=60, r=70, t=55, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── ANALYTICS PANEL: LEFT stats tables | RIGHT regression KPIs ──
        st.markdown('<div class="section-label" style="margin-top:4px;">VARIABLE STATISTICS & REGRESSION SUMMARY</div>', unsafe_allow_html=True)

        # Pre-compute ALL dynamic values before building HTML strings
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
        corr_val_col  = "#f87171" if corr_val < 0 else "#34d399"

        slope_color   = "#34d399" if slope    >= 0   else "#f87171"
        r2_color      = "#34d399" if r2       >= 0.7 else ("#fbbf24" if r2 >= 0.4 else "#f87171")
        rel_color     = "#34d399" if corr_val >= 0   else "#f87171"

        slope_s   = f"{slope:.3f}"
        r2_s      = f"{r2:.3f}"
        adj_r2_s  = f"{adj_r2:.3f}"
        stderr_s  = f"{std_err:.4f}"
        rel_s     = f"{strength} | {relation}"

        left_col, right_col = st.columns([2, 1])

        with left_col:
            # ── Primary stats table ──
            st.markdown(
                f'<div style="background:#111520;border:1px solid #1e2d45;border-radius:8px;overflow:hidden;margin-bottom:12px;">'
                f'<div style="background:#1a2e4a;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;">{primary}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#161d2c;">'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Min</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Max</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Average</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Standard Deviation</th>'
                f'</tr></thead>'
                f'<tbody><tr>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{x_min_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{x_max_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{x_mean_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{x_std_s}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

            # ── Secondary stats table ──
            st.markdown(
                f'<div style="background:#111520;border:1px solid #1e2d45;border-radius:8px;overflow:hidden;margin-bottom:12px;">'
                f'<div style="background:#2a1a3a;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#f472b6;letter-spacing:0.1em;text-transform:uppercase;">{secondary}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#161d2c;">'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Min</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Max</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Average</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Standard Deviation</th>'
                f'</tr></thead>'
                f'<tbody><tr>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{y_min_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{y_max_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{y_mean_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{y_std_s}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

            # ── Correlation table ──
            st.markdown(
                f'<div style="background:#111520;border:1px solid #1e2d45;border-radius:8px;overflow:hidden;">'
                f'<div style="background:#162030;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#fbbf24;letter-spacing:0.1em;text-transform:uppercase;">Correlation — Primary vs Secondary</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#161d2c;">'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Value</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Impact</th>'
                f'<th style="padding:8px 14px;color:#4a6080;font-weight:500;text-align:left;border-bottom:1px solid #1e2d45;">Level</th>'
                f'</tr></thead>'
                f'<tbody><tr>'
                f'<td style="padding:10px 14px;color:{corr_val_col};font-weight:700;">{corr_val_s}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{corr_impact}</td>'
                f'<td style="padding:10px 14px;color:#e2eaf4;">{strength}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

        with right_col:
            # ── Regression KPI panel ──
            st.markdown(
                f'<div style="background:#111520;border:1px solid #1e2d45;border-radius:8px;padding:0;">'
                f'<div style="background:#162030;padding:10px 16px;font-family:Syne,sans-serif;font-size:0.72rem;'
                f'font-weight:700;color:#fbbf24;letter-spacing:0.1em;text-transform:uppercase;">Regression Summary</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #161d2c;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#6b80a0;letter-spacing:0.05em;">SLOPE</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{slope_color};">{slope_s}</span>'
                f'</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #161d2c;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#6b80a0;letter-spacing:0.05em;">R\u00b2</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_color};">{r2_s}</span>'
                f'</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #161d2c;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#6b80a0;letter-spacing:0.05em;">ADJ. R\u00b2</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_color};">{adj_r2_s}</span>'
                f'</div>'

                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #161d2c;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#6b80a0;letter-spacing:0.05em;">STD ERROR</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e2eaf4;">{stderr_s}</span>'
                f'</div>'

                f'<div style="padding:13px 18px;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#6b80a0;letter-spacing:0.05em;display:block;margin-bottom:8px;">RELATIONSHIP</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:{rel_color};">{rel_s}</span>'
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
        background: #111520;
        border: 1px dashed #1e2d45;
        border-radius: 12px;
        margin-top: 40px;
    ">
        <div style="font-size:3rem;margin-bottom:16px;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#38bdf8;font-weight:700;letter-spacing:0.06em;">
            NO DATA LOADED
        </div>
        <div style="font-size:0.78rem;color:#4a6080;margin-top:10px;line-height:1.7;">
            Upload an Excel (.xlsx) file using the sidebar panel.<br>
            First column should be a Date column. Remaining columns should be numeric variables.
        </div>
    </div>
    """, unsafe_allow_html=True)
