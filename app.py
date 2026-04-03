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

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #f4f6fa;
    color: #1a2540;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 100%; }

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

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0;
}
.metric-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 14px 0;
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

[data-testid="stSidebar"] {
    background: #eef1f7;
    border-right: 1px solid #d0daea;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

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
.stDataFrame {
    border: 1px solid #d0daea;
    border-radius: 8px;
}
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

.info-box {
    background: #e8f4fd;
    border: 1px solid #90caf9;
    border-left: 4px solid #1565c0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #1a2540;
    line-height: 1.6;
    margin-bottom: 14px;
}
.tl-info-bar {
    background: #f0f5ff;
    border: 1px solid #c5d0e0;
    border-left: 4px solid #1565c0;
    border-radius: 8px;
    padding: 12px 20px;
    display: flex;
    gap: 40px;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
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
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown('<div class="section-label">DATA SOURCE</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.markdown('<div class="section-label">ABOUT</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem;color:#4a5e80;line-height:1.6;">Upload an Excel file with the first column as Date and subsequent columns as numeric variables.</p>', unsafe_allow_html=True)

# ============================
# MAIN
# ============================
if file:
    # Raw load
    df_raw = pd.read_excel(file, header=0)
    df_raw = df_raw.dropna(how="all")

    # Working copy
    df = df_raw.copy()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols     = list(df.columns)

    def labeled(col):
        return f"{all_cols.index(col)}-{col}"

    numeric_labeled = [labeled(c) for c in numeric_cols]
    label_to_col    = {labeled(c): c for c in numeric_cols}

    # ============================
    # PREPROCESS COMPUTATION (outside tabs — df_clean shared across all tabs)
    # ============================
    text_strategy_val = st.session_state.get("text_strat", "Convert to 0")
    null_strategy_val = st.session_state.get("null_strat", "Keep nulls as-is")
    sigma_n_val       = int(st.session_state.get("sigma_n", 3))
    out_cols_val      = st.session_state.get("out_cols", [])
    apply_outlier_val = bool(st.session_state.get("apply_out", False))

    df_clean = df.copy()

    for col in numeric_cols:
        if col not in df_raw.columns:
            continue
        text_mask = df_raw[col].notna() & df_clean[col].isna()
        if not text_mask.any():
            continue
        if text_strategy_val == "Convert to 0":
            df_clean.loc[text_mask, col] = 0.0
        elif text_strategy_val == "Convert to column Mean":
            m = df_clean[col].mean()
            df_clean.loc[text_mask, col] = m if not np.isnan(m) else 0.0
        elif text_strategy_val == "Drop rows containing text values":
            df_clean = df_clean[~text_mask]

    if null_strategy_val == "Drop rows with any null":
        df_clean = df_clean.dropna(subset=numeric_cols)
    elif null_strategy_val == "Fill nulls with column Mean":
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif null_strategy_val == "Fill nulls with column Median":
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif null_strategy_val == "Fill nulls with 0":
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(0.0)

    outlier_report = {}
    if apply_outlier_val:
        target_cols_ov = out_cols_val if out_cols_val else numeric_cols
        for col in target_cols_ov:
            s = df_clean[col].dropna()
            if len(s) < 3:
                continue
            mu, sv  = s.mean(), s.std()
            lo, hi  = mu - sigma_n_val * sv, mu + sigma_n_val * sv
            out_mask = (df_clean[col] < lo) | (df_clean[col] > hi)
            outlier_report[col] = {
                "removed": int(out_mask.sum()),
                "lower": round(lo, 4), "upper": round(hi, 4),
                "mean": round(mu, 4),  "std":   round(sv, 4)
            }
            df_clean = df_clean[~out_mask | df_clean[col].isna()]

    # Labels derived from df_clean (used in Correlation + Analysis tabs)
    clean_numeric_cols    = df_clean.select_dtypes(include=np.number).columns.tolist()
    clean_all_cols        = list(df_clean.columns)
    def clean_labeled(col):
        return f"{clean_all_cols.index(col)}-{col}"
    clean_numeric_labeled = [clean_labeled(c) for c in clean_numeric_cols]
    clean_label_to_col    = {clean_labeled(c): c for c in clean_numeric_cols}

    # ============================
    # TABS
    # ============================
    tab0, tab1, tab2, tab3 = st.tabs(["🗂️  Raw Data", "⚙️  Preprocess", "📊  Correlation", "📈  Analysis"])

    # ══════════════════════════
    # TAB 0 — RAW DATA
    # ══════════════════════════
    with tab0:
        st.markdown('<div class="section-label">RAW UPLOADED DATA</div>', unsafe_allow_html=True)

        null_count   = int(df_raw.isnull().sum().sum())
        non_num_count = 0
        for col in df_raw.columns[1:]:
            non_num_count += max(0,
                int(pd.to_numeric(df_raw[col], errors='coerce').isna().sum())
                - int(df_raw[col].isna().sum())
            )

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{len(df_raw)}</div>
                <div class="metric-sub">Excluding header</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{len(df_raw.columns)}</div>
                <div class="metric-sub">Including date column</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Null / Empty Cells</div>
                <div class="metric-value">{null_count}</div>
                <div class="metric-sub">Across all columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Non-Numeric Cells</div>
                <div class="metric-value">{non_num_count}</div>
                <div class="metric-sub">In numeric columns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:8px;">DATA PREVIEW</div>', unsafe_allow_html=True)
        st.dataframe(df_raw, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label" style="margin-top:18px;">NULL & DATA QUALITY PER COLUMN</div>', unsafe_allow_html=True)
        quality_rows = []
        for col in df_raw.columns:
            nc  = int(df_raw[col].isnull().sum())
            nnc = 0 if col == date_col else max(0,
                int(pd.to_numeric(df_raw[col], errors='coerce').isna().sum()) - int(df_raw[col].isna().sum()))
            quality_rows.append({
                "Column": col,
                "Dtype": "Date" if col == date_col else str(df_raw[col].dtype),
                "Null Count": nc,
                "Non-Numeric Count": nnc,
                "Fill Rate %": f"{100*(1 - nc/max(len(df_raw),1)):.1f}%"
            })
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════
    # TAB 1 — PREPROCESS
    # ══════════════════════════
    with tab1:
        st.markdown('<div class="section-label">STEP 1 — NON-NUMERIC / TEXT VALUE TREATMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Cells containing text or non-numeric values (e.g. "N/A", "—", "bad data") in numeric columns are handled here.</div>', unsafe_allow_html=True)

        st.radio(
            "Non-Numeric / Text Cell Strategy",
            ["Convert to 0", "Convert to column Mean", "Drop rows containing text values", "Keep as NaN (treat as null)"],
            horizontal=True, key="text_strat"
        )

        st.markdown('<div class="section-label" style="margin-top:18px;">STEP 2 — NULL VALUE TREATMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Choose how to handle null / empty cells in numeric columns.</div>', unsafe_allow_html=True)

        st.radio(
            "Null Value Strategy",
            ["Drop rows with any null", "Fill nulls with column Mean", "Fill nulls with column Median", "Fill nulls with 0", "Keep nulls as-is"],
            horizontal=True, key="null_strat"
        )

        st.markdown('<div class="section-label" style="margin-top:18px;">STEP 3 — OUTLIER TREATMENT (3-SIGMA RULE)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            '<b>3-Sigma (3σ) Rule</b>: Values outside <b>Mean ± N × Std Dev</b> are flagged as outliers. '
            '<b>1σ</b> removes ~32% of data (aggressive) · <b>2σ</b> removes ~5% (moderate) · <b>3σ</b> removes ~0.3% (conservative).'
            '</div>',
            unsafe_allow_html=True
        )

        oc1, oc2 = st.columns([1, 3])
        with oc1:
            st.radio(
                "Sigma Level (N)",
                [1, 2, 3],
                index=2,
                format_func=lambda x: f"{x}σ  ({'Aggressive' if x==1 else 'Moderate' if x==2 else 'Conservative'})",
                key="sigma_n"
            )
        with oc2:
            st.multiselect(
                "Apply to Columns (leave empty = all numeric)",
                numeric_cols, key="out_cols"
            )
        st.checkbox("Apply Outlier Treatment (3-Sigma)", value=False, key="apply_out")

        # ── Display results using pre-computed df_clean ──
        removed_total  = len(df) - len(df_clean)
        null_remaining = int(df_clean[numeric_cols].isnull().sum().sum())

        st.markdown(f"""
        <div class="metric-grid-3">
            <div class="metric-card">
                <div class="metric-label">Rows After Cleaning</div>
                <div class="metric-value">{len(df_clean)}</div>
                <div class="metric-sub">Started with {len(df)} rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rows Removed</div>
                <div class="metric-value">{removed_total}</div>
                <div class="metric-sub">Outliers + dropped nulls</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Remaining Nulls</div>
                <div class="metric-value">{null_remaining}</div>
                <div class="metric-sub">In numeric columns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if apply_outlier_val and outlier_report:
            st.markdown('<div class="section-label" style="margin-top:8px;">3-SIGMA OUTLIER REPORT</div>', unsafe_allow_html=True)
            rrows = []
            for col, info in outlier_report.items():
                rrows.append({
                    "Column": col,
                    "Mean": info["mean"], "Std Dev": info["std"],
                    f"Lower (Mean−{sigma_n_val}σ)": info["lower"],
                    f"Upper (Mean+{sigma_n_val}σ)": info["upper"],
                    "Outliers Removed": info["removed"]
                })
            st.dataframe(pd.DataFrame(rrows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label" style="margin-top:18px;">CLEANED DATA PREVIEW (used by Correlation & Analysis tabs)</div>', unsafe_allow_html=True)
        st.dataframe(df_clean, use_container_width=True, hide_index=True)

    # ══════════════════════════
    # TAB 2 — CORRELATION
    # ══════════════════════════
    with tab2:
        st.markdown('<div class="section-label">CORRELATION INTELLIGENCE</div>', unsafe_allow_html=True)

        if len(clean_numeric_cols) < 2:
            st.warning("Not enough numeric columns in cleaned data. Adjust Preprocess settings.")
            st.stop()

        corr = df_clean[clean_numeric_cols].corr()

        # Full matrix
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

        # ── Single-row bar chart ──
        st.markdown('<div class="section-label" style="margin-top:18px;">TARGET CORRELATION — SINGLE ROW BAR CHART</div>', unsafe_allow_html=True)
        bc1, bc2 = st.columns([3, 1])
        with bc1:
            target_lbl = st.selectbox("Select Target Variable", clean_numeric_labeled, key="corr_target")
            target_col = clean_label_to_col[target_lbl]
        with bc2:
            top_n = st.slider("Top N Variables", 5, len(clean_numeric_cols), min(15, len(clean_numeric_cols)), key="top_n_corr")

        corr_series = corr[target_col].drop(labels=[target_col])
        corr_top    = corr_series.abs().sort_values(ascending=False).head(top_n)
        corr_vals   = corr_series[corr_top.index]

        bar_colors = ["#1565c0" if v >= 0 else "#c0392b" for v in corr_vals.values]

        fig_bar = go.Figure(go.Bar(
            x=corr_vals.index.tolist(),
            y=corr_vals.values,
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in corr_vals.values],
            textposition="outside",
            textfont=dict(size=11, color="#1a2540"),
            width=0.6
        ))
        fig_bar.add_hline(y=0,    line_color="#888888", line_width=1)
        fig_bar.add_hline(y=0.7,  line_dash="dash", line_color="#1a7a4a", line_width=1,
                          annotation_text="Strong +0.7", annotation_font_color="#1a7a4a", annotation_position="right")
        fig_bar.add_hline(y=-0.7, line_dash="dash", line_color="#c0392b", line_width=1,
                          annotation_text="Strong −0.7", annotation_font_color="#c0392b", annotation_position="right")
        fig_bar.add_hline(y=0.3,  line_dash="dot", line_color="#b8860b", line_width=1)
        fig_bar.add_hline(y=-0.3, line_dash="dot", line_color="#b8860b", line_width=1)

        fig_bar.update_layout(
            height=420,
            title=dict(
                text=f"Correlation with <b>{target_col}</b>",
                font=dict(family="Syne, sans-serif", size=14, color="#1a2540"),
                x=0.5, xanchor="center"
            ),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(family="DM Mono, monospace", color="#1a2540", size=11),
            yaxis=dict(
                title="Correlation Coefficient",
                range=[-1.2, 1.2],
                gridcolor="#e8edf5", zeroline=False,
                tickfont=dict(color="#1a2540")
            ),
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(color="#1a2540", size=10),
                gridcolor="#e8edf5"
            ),
            margin=dict(l=60, r=80, t=60, b=130),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        

    # ══════════════════════════
    # TAB 3 — ANALYSIS
    # ══════════════════════════
    with tab3:
        st.markdown('<div class="section-label">DYNAMIC VARIABLE ANALYSIS</div>', unsafe_allow_html=True)

        if len(clean_numeric_cols) < 2:
            st.warning("Not enough numeric columns in cleaned data. Adjust Preprocess settings.")
            st.stop()

        date_list = sorted(df_clean[date_col].dt.date.unique())

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            start_date = st.selectbox("Start Date", date_list, key="sd")
        with r1c2:
            end_date = st.selectbox("End Date", date_list, index=len(date_list)-1, key="ed")
        with r1c3:
            primary_lbl = st.selectbox("Primary Variable", clean_numeric_labeled, key="pv")
            primary     = clean_label_to_col[primary_lbl]
        with r1c4:
            secondary_lbl = st.selectbox("Secondary Variable", clean_numeric_labeled, index=1, key="sv")
            secondary     = clean_label_to_col[secondary_lbl]

        # Trendline controls
        tl1, tl2 = st.columns([2, 2])
        with tl1:
            trendline_type = st.selectbox(
                "Trendline Type (for Primary Variable)",
                ["Exponential", "Linear", "Logarithmic", "Power"],
                key="tl_type"
            )
        with tl2:
            show_equation = st.checkbox("Show Equation & R² on Chart", value=True, key="show_eq")

        if start_date > end_date:
            st.error("⚠ Start Date cannot be after End Date.")
            st.stop()

        df_f = df_clean[(df_clean[date_col].dt.date >= start_date) & (df_clean[date_col].dt.date <= end_date)]

        if len(df_f) < 2:
            st.warning("Not enough data in selected date range.")
            st.stop()

        x = df_f[primary]
        y = df_f[secondary]

        # Regression (x vs y for summary stats)
        slope, intercept, r_val, p, std_err = linregress(x.dropna(), y.dropna())
        r2     = r_val**2
        n      = len(x)
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
            if len(s) < 2: return None, None, None, None
            sv, ev = s.iloc[0], s.iloc[-1]
            if sv == 0: return sv, ev, None, None
            chg = ((ev - sv) / abs(sv)) * 100
            return sv, ev, chg, (series.pct_change() * 100).mean()

        px_start, px_end, px_chg, px_avg = pct_change_series(x)
        py_start, py_end, py_chg, py_avg = pct_change_series(y)

        def fmt_chg(v):
            if v is None: return "N/A", "neu"
            return (f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"), ("pos" if v >= 0 else "neg")

        px_chg_str, px_cls     = fmt_chg(px_chg)
        px_avg_str, px_avg_cls = fmt_chg(px_avg)
        py_chg_str, py_cls     = fmt_chg(py_chg)
        py_avg_str, py_avg_cls = fmt_chg(py_avg)

        # Metrics
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

        # Period change cards
        st.markdown('<div class="section-label" style="margin-top:8px;">PERIOD CHANGE ANALYTICS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="change-grid">
            <div class="change-card">
                <div class="change-card-title">📌 {primary} — Period Performance</div>
                <div class="change-row"><span class="change-key">Opening Value</span><span class="change-val neu">{px_start:.4g}</span></div>
                <div class="change-row"><span class="change-key">Closing Value</span><span class="change-val neu">{px_end:.4g}</span></div>
                <div class="change-row"><span class="change-key">Total % Change</span><span class="change-val {px_cls}">{px_chg_str}</span></div>
            </div>
            <div class="change-card">
                <div class="change-card-title">📌 {secondary} — Period Performance</div>
                <div class="change-row"><span class="change-key">Opening Value</span><span class="change-val neu">{py_start:.4g}</span></div>
                <div class="change-row"><span class="change-key">Closing Value</span><span class="change-val neu">{py_end:.4g}</span></div>
                <div class="change-row"><span class="change-key">Total % Change</span><span class="change-val {py_cls}">{py_chg_str}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Axis scale
        st.markdown('<div class="section-label" style="margin-top:8px;">CHART AXIS SCALE CUSTOMIZATION</div>', unsafe_allow_html=True)

        x_min_d = float(x.min()) if not x.isna().all() else 0.0
        x_max_d = float(x.max()) if not x.isna().all() else 100.0
        y_min_d = float(y.min()) if not y.isna().all() else 0.0
        y_max_d = float(y.max()) if not y.isna().all() else 100.0
        xp = (x_max_d - x_min_d) * 0.1 or abs(x_max_d) * 0.1 or 10
        yp = (y_max_d - y_min_d) * 0.1 or abs(y_max_d) * 0.1 or 10

        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        with ac1:
            lbl = f"Y-Left Min ({primary[:10]}...)" if len(primary)>10 else f"Y-Left Min ({primary})"
            y1_min = st.number_input(lbl, value=round(x_min_d - xp, 4), format="%.4f", key="y1min")
        with ac2:
            lbl = f"Y-Left Max ({primary[:10]}...)" if len(primary)>10 else f"Y-Left Max ({primary})"
            y1_max = st.number_input(lbl, value=round(x_max_d + xp, 4), format="%.4f", key="y1max")
        with ac3:
            lbl = f"Y-Right Min ({secondary[:10]}...)" if len(secondary)>10 else f"Y-Right Min ({secondary})"
            y2_min = st.number_input(lbl, value=round(y_min_d - yp, 4), format="%.4f", key="y2min")
        with ac4:
            lbl = f"Y-Right Max ({secondary[:10]}...)" if len(secondary)>10 else f"Y-Right Max ({secondary})"
            y2_max = st.number_input(lbl, value=round(y_max_d + yp, 4), format="%.4f", key="y2max")
        with ac5:
            auto_scale = st.checkbox("Auto Scale", value=True, key="autoscale")

        yaxis_range  = None if auto_scale else [y1_min, y1_max]
        yaxis2_range = None if auto_scale else [y2_min, y2_max]

        # ── CHART ──
        st.markdown('<div class="section-label" style="margin-top:8px;">TIME SERIES & TREND</div>', unsafe_allow_html=True)

        PRIMARY_COLOR   = "#1a6fba"
        SECONDARY_COLOR = "#c0392b"
        N_DENSE         = 500

        fig = go.Figure()

        # Primary smooth line
        fig.add_trace(go.Scatter(
            x=df_f[date_col], y=x,
            name=primary, mode="lines",
            line=dict(width=2.5, color=PRIMARY_COLOR, shape="spline", smoothing=1.3)
        ))

        # ── TRENDLINE ──
        trend_equation = ""
        trend_r2       = None
        trend_ok       = False

        try:
            df_tp  = df_f[[date_col, primary]].dropna().reset_index(drop=True)
            yv     = df_tp[primary].values
            t_fit  = np.arange(len(df_tp), dtype=float)

            date_start_ns = df_tp[date_col].iloc[0].value
            date_end_ns   = df_tp[date_col].iloc[-1].value
            dense_dates   = pd.to_datetime(np.linspace(date_start_ns, date_end_ns, N_DENSE))
            t_dense       = np.linspace(0, len(df_tp) - 1, N_DENSE)

            if trendline_type == "Exponential":
                mask  = yv > 0
                coef  = np.polyfit(t_fit[mask], np.log(yv[mask]), 1)
                a, b  = np.exp(coef[1]), coef[0]
                y_pred_fit   = a * np.exp(b * t_fit)
                trend_dense_y = a * np.exp(b * t_dense)
                sign_b = "+" if b >= 0 else "−"
                trend_equation = f"y = {a:.4g} · e^({b:+.4g}x)"

            elif trendline_type == "Linear":
                coef  = np.polyfit(t_fit, yv, 1)
                a, b  = coef[0], coef[1]
                y_pred_fit    = a * t_fit + b
                trend_dense_y = a * t_dense + b
                trend_equation = f"y = {a:.4g}x {'+ ' if b>=0 else '− '}{abs(b):.4g}"

            elif trendline_type == "Logarithmic":
                t_log = t_fit + 1
                coef  = np.polyfit(np.log(t_log), yv, 1)
                a, b  = coef[0], coef[1]
                y_pred_fit    = a * np.log(t_fit + 1) + b
                trend_dense_y = a * np.log(t_dense + 1) + b
                trend_equation = f"y = {a:.4g} · ln(x) {'+ ' if b>=0 else '− '}{abs(b):.4g}"

            elif trendline_type == "Power":
                mask  = yv > 0
                coef  = np.polyfit(np.log(t_fit[mask] + 1), np.log(yv[mask]), 1)
                b, a  = coef[0], np.exp(coef[1])
                y_pred_fit    = a * (t_fit + 1) ** b
                trend_dense_y = a * (t_dense + 1) ** b
                trend_equation = f"y = {a:.4g} · x^{b:.4g}"

            # R² for trendline
            ss_res   = np.sum((yv - y_pred_fit) ** 2)
            ss_tot   = np.sum((yv - np.mean(yv)) ** 2)
            trend_r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            trend_ok = True

        except Exception:
            trend_ok = False

        if trend_ok:
            fig.add_trace(go.Scatter(
                x=dense_dates, y=trend_dense_y,
                name=f"Trend ({primary})",
                mode="lines", showlegend=False,
                line=dict(width=2, color=PRIMARY_COLOR, shape="spline", smoothing=1.3)
            ))
            if show_equation and trend_equation:
                ann_x  = dense_dates[int(N_DENSE * 0.62)]
                ann_y  = float(np.nanpercentile(x.dropna(), 40))
                eq_text = f"{trend_equation}<br>R² = {trend_r2:.4f}"
                fig.add_annotation(
                    x=ann_x, y=ann_y,
                    text=eq_text,
                    showarrow=False,
                    font=dict(family="DM Mono, monospace", size=12, color="#1a2540"),
                    bgcolor="rgba(255,255,255,0.90)",
                    bordercolor="#1565c0",
                    borderwidth=1,
                    borderpad=8,
                    align="left"
                )

        # Secondary smooth line
        fig.add_trace(go.Scatter(
            x=df_f[date_col], y=y,
            name=secondary, yaxis="y2", mode="lines",
            line=dict(width=2.5, color=SECONDARY_COLOR, shape="spline", smoothing=1.3)
        ))

        fig.update_layout(
            height=520,
            title=dict(
                text=f"<b>{primary}</b>  vs  <b>{secondary}</b>  —  {trendline_type} Trendline",
                font=dict(family="Syne, sans-serif", size=15, color="#1a2540"),
                x=0.5, xanchor="center"
            ),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(family="DM Mono, monospace", color="#1a2540", size=11),
            legend=dict(
                bgcolor="#f8f9fc", bordercolor="#d0daea", borderwidth=1,
                orientation="h", x=0.5, xanchor="center", y=-0.22,
                font=dict(size=12, color="#1a2540")
            ),
            yaxis=dict(
                title=dict(text=f"<b>{primary}</b>", font=dict(color=PRIMARY_COLOR, size=12)),
                tickfont=dict(color=PRIMARY_COLOR, size=11),
                gridcolor="#e8edf5", gridwidth=1, griddash="dash",
                zeroline=True, zerolinecolor="#c5d0e0",
                showline=True, linecolor="#c5d0e0",
                range=yaxis_range
            ),
            yaxis2=dict(
                title=dict(text=f"<b>{secondary}</b>", font=dict(color=SECONDARY_COLOR, size=12)),
                tickfont=dict(color=SECONDARY_COLOR, size=11),
                overlaying="y", side="right",
                gridcolor="#e8edf5", zeroline=False,
                showline=True, linecolor="#c5d0e0",
                range=yaxis2_range
            ),
            xaxis=dict(
                gridcolor="#e8edf5", gridwidth=1, griddash="dash",
                zeroline=False, showline=True, linecolor="#c5d0e0",
                tickangle=-45, tickfont=dict(color="#2a3a55", size=10)
            ),
            margin=dict(l=70, r=80, t=60, b=90)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Trendline info banner
        if trend_ok and trend_equation:
            tl_r2_col = "#1a7a4a" if trend_r2 >= 0.7 else ("#d97706" if trend_r2 >= 0.4 else "#c0392b")
            st.markdown(
                f'<div class="tl-info-bar">'
                f'<div><div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5e80;text-transform:uppercase;letter-spacing:0.08em;">Trendline Type</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#1565c0;">{trendline_type}</div></div>'
                f'<div><div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5e80;text-transform:uppercase;letter-spacing:0.08em;">Equation</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:0.95rem;font-weight:600;color:#1a2540;">{trend_equation}</div></div>'
                f'<div><div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5e80;text-transform:uppercase;letter-spacing:0.08em;">Trendline R²</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{tl_r2_col};">{trend_r2:.4f}</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Stats & regression
        st.markdown('<div class="section-label" style="margin-top:4px;">VARIABLE STATISTICS & REGRESSION SUMMARY</div>', unsafe_allow_html=True)

        def stat_table(col_name, s, header_bg, header_color):
            return (
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
                f'<div style="background:{header_bg};padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;font-weight:700;color:{header_color};letter-spacing:0.1em;text-transform:uppercase;">{col_name}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#f4f7fc;">'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Min</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Max</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Average</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Std Dev</th>'
                f'</tr></thead><tbody><tr>'
                f'<td style="padding:10px 14px;color:#1a2540;">{s.min():.7g}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{s.max():.7g}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{s.mean():.7g}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{s.std():.7g}</td>'
                f'</tr></tbody></table></div>'
            )

        corr_val_col = "#c0392b" if corr_val < 0 else "#1a7a4a"
        slope_color  = "#1a7a4a" if slope >= 0  else "#c0392b"
        r2_color     = "#1a7a4a" if r2    >= 0.7 else ("#d97706" if r2 >= 0.4 else "#c0392b")

        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.markdown(stat_table(primary, x, "#e8f0fe", "#1565c0"), unsafe_allow_html=True)
            st.markdown(stat_table(secondary, y, "#fce8e6", "#c0392b"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
                f'<div style="background:#fffbe6;padding:8px 14px;font-family:Syne,sans-serif;font-size:0.72rem;font-weight:700;color:#b45309;letter-spacing:0.1em;text-transform:uppercase;">Correlation — Primary vs Secondary</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:0.78rem;">'
                f'<thead><tr style="background:#fdfaf0;">'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Value</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Impact</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Level</th>'
                f'</tr></thead><tbody><tr>'
                f'<td style="padding:10px 14px;color:{corr_val_col};font-weight:700;">{corr_val:.7f}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{"Positive" if corr_val > 0 else "Negative"}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{strength}</td>'
                f'</tr></tbody></table></div>',
                unsafe_allow_html=True
            )

        with right_col:
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #d0daea;border-radius:8px;padding:0;box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
                f'<div style="background:#fffbe6;padding:10px 16px;font-family:Syne,sans-serif;font-size:0.72rem;font-weight:700;color:#b45309;letter-spacing:0.1em;text-transform:uppercase;">Regression Summary</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #eef1f7;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5e80;">SLOPE</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{slope_color};">{slope:.3f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #eef1f7;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5e80;">R²</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_color};">{r2:.3f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;">'
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5e80;">ADJ. R²</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_color};">{adj_r2:.3f}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Key insights
        st.markdown('<div class="section-label" style="margin-top:10px;">KEY INSIGHTS</div>', unsafe_allow_html=True)
        tl_text = (f"Trendline ({trendline_type}): <b>{trend_equation}</b> | R² = <b>{trend_r2:.4f}</b>"
                   if trend_ok and trend_equation else "Trendline could not be fitted.")

        st.markdown(f"""
        <div class="insight-block">
            <div class="insight-heading">Relationship</div>
            <div class="insight-text">
                <b>{strength} {relation}</b> relationship between <b>{primary}</b> and <b>{secondary}</b>
                (r = <b>{corr_val:.3f}</b>) from <b>{start_date}</b> to <b>{end_date}</b>.
            </div>
            <div class="insight-heading" style="margin-top:16px;">Trendline Summary</div>
            <div class="insight-text">{tl_text}</div>
            <div class="insight-heading" style="margin-top:16px;">Period Change</div>
            <div class="insight-text">
                <b>{primary}</b> {"increased" if (px_chg or 0)>=0 else "decreased"} by <b>{px_chg_str}</b>
                ({px_start:.4g} → {px_end:.4g}), avg change <b>{px_avg_str}</b>.<br><br>
                <b>{secondary}</b> {"increased" if (py_chg or 0)>=0 else "decreased"} by <b>{py_chg_str}</b>
                ({py_start:.4g} → {py_end:.4g}), avg change <b>{py_avg_str}</b>.
            </div>
            <div class="insight-heading" style="margin-top:16px;">Regression Interpretation</div>
            <div class="insight-text">
                Slope <b>{slope:.4f}</b> — per unit increase in <b>{primary}</b>,
                <b>{secondary}</b> changes by ~<b>{slope:.4f}</b> units.
                Model explains <b>{r2*100:.1f}%</b> of variance
                (R² = <b>{r2:.4f}</b>, Adj. R² = <b>{adj_r2:.4f}</b>).
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;background:#ffffff;border:1px dashed #c5d0e0;
        border-radius:12px;margin-top:40px;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
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
