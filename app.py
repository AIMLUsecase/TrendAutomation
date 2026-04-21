import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, leaves_list
import base64, io, json

st.set_page_config(layout="wide", page_title="Analysis Dashboard", page_icon="📊")

# ─────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Mono',monospace;background:#f4f6fa;color:#1a2540;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem;max-width:100%;}
.dash-header{background:linear-gradient(135deg,#1a3a6e,#1e4d8c 60%,#1565c0);border:1px solid #1a5cb5;border-radius:12px;padding:28px 40px;margin-bottom:28px;position:relative;overflow:hidden;}
.dash-header::before{content:"";position:absolute;top:-40px;right:-40px;width:220px;height:220px;background:radial-gradient(circle,rgba(255,255,255,.10),transparent 70%);border-radius:50%;}
.dash-title{font-family:'Syne',sans-serif;font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:.04em;margin:0;}
.dash-subtitle{font-family:'DM Mono',monospace;font-size:.75rem;color:#90caf9;letter-spacing:.12em;text-transform:uppercase;margin-top:6px;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#e8edf5;border-radius:10px;padding:4px;border:1px solid #c5d0e0;}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:600;letter-spacing:.06em;color:#4a5e80;background:transparent;border-radius:8px;padding:10px 24px;border:none;}
.stTabs [aria-selected="true"]{background:#fff!important;color:#1565c0!important;border:1px solid #c5d0e0!important;}
.metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:18px 0;}
.metric-grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0;}
.metric-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0;}
.metric-card{background:#fff;border:1px solid #d0daea;border-radius:10px;padding:18px 20px;position:relative;overflow:hidden;transition:border-color .2s;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.metric-card:hover{border-color:#1565c0;}
.metric-card::after{content:"";position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1565c0,#42a5f5);opacity:.7;}
.metric-label{font-family:'DM Mono',monospace;font-size:.65rem;color:#5a6e8a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;}
.metric-value{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:#1a2540;}
.metric-sub{font-size:.7rem;color:#5a6e8a;margin-top:4px;}
.change-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0;}
.change-card{background:#fff;border:1px solid #d0daea;border-radius:10px;padding:22px 24px;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.change-card-title{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:600;color:#1565c0;letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px;border-bottom:1px solid #d0daea;padding-bottom:10px;}
.change-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #eef1f7;}
.change-row:last-child{border-bottom:none;}
.change-key{font-size:.72rem;color:#4a5e80;}
.change-val{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;}
.pos{color:#1a7a4a;}.neg{color:#c0392b;}.neu{color:#1a2540;}
.insight-block{background:#f0f5ff;border:1px solid #c5d0e0;border-left:4px solid #1565c0;border-radius:10px;padding:24px 28px;margin-top:18px;}
.insight-heading{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;color:#1565c0;letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px;}
.insight-text{font-size:.82rem;color:#2a3a55;line-height:1.7;}
.insight-text b{color:#1a2540;}
.section-label{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;color:#1565c0;letter-spacing:.14em;text-transform:uppercase;padding:0 0 10px;border-bottom:1px solid #d0daea;margin-bottom:14px;}
[data-testid="stSidebar"]{background:#eef1f7;border-right:1px solid #d0daea;}
[data-testid="stSidebar"] .block-container{padding:1.5rem 1.2rem;}
.stSelectbox>div>div,.stSlider>div{background:#fff!important;border-color:#c5d0e0!important;border-radius:8px!important;}
.stSelectbox label,.stSlider label{font-size:.7rem!important;color:#4a5e80!important;letter-spacing:.08em!important;text-transform:uppercase!important;}
.stDataFrame{border:1px solid #d0daea;border-radius:8px;}
[data-testid="stFileUploaderDropzoneInstructions"] button{background:linear-gradient(135deg,#1565c0,#0d47a1);color:#fff;border:none;border-radius:8px;font-family:'Syne',sans-serif;font-weight:600;letter-spacing:.08em;padding:10px 24px;}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1);color:#fff;border:none;border-radius:8px;font-family:'Syne',sans-serif;font-size:.78rem;font-weight:600;letter-spacing:.08em;padding:10px 24px;}
.stButton>button:hover{opacity:.85;}
.info-box{background:#e8f4fd;border:1px solid #90caf9;border-left:4px solid #1565c0;border-radius:8px;padding:12px 16px;font-size:.78rem;color:#1a2540;line-height:1.6;margin-bottom:14px;}
.tl-info-bar{background:#f0f5ff;border:1px solid #c5d0e0;border-left:4px solid #1565c0;border-radius:8px;padding:12px 20px;display:flex;gap:40px;align-items:center;margin-bottom:10px;flex-wrap:wrap;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-header">
  <div class="dash-title">TREND ANALYSIS DASHBOARD</div>
  <div class="dash-subtitle">Dynamic Variable Comparison &nbsp;·&nbsp; Trend Analysis &nbsp;·&nbsp; Correlation Intelligence</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-label">DATA SOURCE</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload Master Data Files", type=["xlsx"], label_visibility="visible")
    st.markdown("---")
    st.markdown('<div class="section-label">ABOUT</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.72rem;color:#4a5e80;line-height:1.6;">Upload an Excel file — first column: Date, remaining: numeric variables.</p>', unsafe_allow_html=True)

# ── Rename "Browse files" → "Upload Files" via components iframe ──
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    function rename() {
        var doc = window.parent.document;
        var found = false;
        doc.querySelectorAll('button').forEach(function(btn) {
            var spans = btn.querySelectorAll('span');
            spans.forEach(function(sp) {
                if (sp.innerText && sp.innerText.trim().toLowerCase() === 'browse files') {
                    sp.innerText = 'Upload Files';
                    found = true;
                }
            });
            if (!found && btn.innerText && btn.innerText.trim().toLowerCase() === 'browse files') {
                btn.innerText = 'Upload Files';
                found = true;
            }
        });
        if (!found) setTimeout(rename, 200);
    }
    setTimeout(rename, 300);
    // keep watching for rerenders
    setInterval(rename, 2000);
})();
</script>
""", height=0)

# ─────────────────────────────────────────────────────────
# HELPER: build a trendline
# ─────────────────────────────────────────────────────────
def compute_trendline(dates, values, tl_type, n_dense=500):
    """Returns (dense_dates, dense_y, equation_str, r2_val, ok_bool)"""
    try:
        df_tp = pd.DataFrame({"d": dates, "v": values}).dropna().reset_index(drop=True)
        if len(df_tp) < 3:
            return None, None, "", None, False
        yv    = df_tp["v"].values.astype(float)
        t_fit = np.arange(len(df_tp), dtype=float)
        ds    = df_tp["d"]
        date_start_ns = ds.iloc[0].value
        date_end_ns   = ds.iloc[-1].value
        dense_dates   = pd.to_datetime(np.linspace(date_start_ns, date_end_ns, n_dense))
        t_dense       = np.linspace(0, len(df_tp)-1, n_dense)

        if tl_type == "Exponential":
            mask  = yv > 0
            if mask.sum() < 2: raise ValueError
            coef  = np.polyfit(t_fit[mask], np.log(yv[mask]), 1)
            a, b  = np.exp(coef[1]), coef[0]
            y_pred_fit = a * np.exp(b * t_fit)
            dense_y    = a * np.exp(b * t_dense)
            eq = f"y = {a:.4g}·e^({b:+.4g}x)"

        elif tl_type == "Linear":
            coef  = np.polyfit(t_fit, yv, 1)
            a, b  = coef[0], coef[1]
            y_pred_fit = a*t_fit + b
            dense_y    = a*t_dense + b
            eq = f"y = {a:.4g}x {'+ ' if b>=0 else '− '}{abs(b):.4g}"

        elif tl_type == "Logarithmic":
            coef  = np.polyfit(np.log(t_fit+1), yv, 1)
            a, b  = coef[0], coef[1]
            y_pred_fit = a*np.log(t_fit+1) + b
            dense_y    = a*np.log(t_dense+1) + b
            eq = f"y = {a:.4g}·ln(x) {'+ ' if b>=0 else '− '}{abs(b):.4g}"

        elif tl_type == "Power":
            mask  = yv > 0
            if mask.sum() < 2: raise ValueError
            coef  = np.polyfit(np.log(t_fit[mask]+1), np.log(yv[mask]), 1)
            b, a  = coef[0], np.exp(coef[1])
            y_pred_fit = a*(t_fit+1)**b
            dense_y    = a*(t_dense+1)**b
            eq = f"y = {a:.4g}·x^{b:.4g}"
        else:
            raise ValueError

        ss_res = np.sum((yv - y_pred_fit)**2)
        ss_tot = np.sum((yv - np.mean(yv))**2)
        r2_val = float(1 - ss_res/ss_tot) if ss_tot != 0 else 0.0
        return dense_dates, dense_y, eq, r2_val, True
    except Exception:
        return None, None, "", None, False


# ─────────────────────────────────────────────────────────
# HELPER: build chart figure
# ─────────────────────────────────────────────────────────
CHART_MODES = {
    "Line":                    ("lines",        "spline",  1.3),
    "Scatter":                 ("markers",      "linear",  0),
    "Scatter + Line":          ("lines+markers","spline",  1.3),
    "Straight Line (XY)":      ("lines",        "linear",  0),
    "Scatter Smooth Line (XY)":("lines+markers","spline",  1.3),
}

PRIMARY_COLOR   = "#1a6fba"
SECONDARY_COLOR = "#c0392b"
PALETTE = ["#1a6fba","#c0392b","#1a7a4a","#d97706","#7c3aed","#0e7490","#be185d","#92400e","#374151","#065f46"]

def make_chart(df_plot, date_col, primary, secondaries,
               tl_type, show_eq,
               chart_mode_primary, chart_modes_sec,
               y_range=None, y2_range=None,
               height=520):
    mode_p, shape_p, smooth_p = CHART_MODES.get(chart_mode_primary, ("lines","spline",1.3))
    fig = go.Figure()

    x_vals = df_plot[date_col]
    yp     = df_plot[primary]

    fig.add_trace(go.Scatter(
        x=x_vals, y=yp, name=primary, mode=mode_p,
        line=dict(width=2.5, color=PRIMARY_COLOR, shape=shape_p, smoothing=smooth_p),
        marker=dict(size=5, color=PRIMARY_COLOR)
    ))

    # Trendline on primary
    dd, dy, eq, r2v, ok = compute_trendline(x_vals, yp, tl_type)
    if ok:
        fig.add_trace(go.Scatter(
            x=dd, y=dy, mode="lines", showlegend=False,
            line=dict(width=1.8, color=PRIMARY_COLOR, shape="spline", smoothing=1.3)
        ))
        if show_eq and eq:
            ann_x = dd[int(len(dd)*0.62)]
            ann_y = float(np.nanpercentile(yp.dropna(), 40))
            fig.add_annotation(
                x=ann_x, y=ann_y, text=f"{eq}<br>R² = {r2v:.4f}",
                showarrow=False,
                font=dict(family="DM Mono,monospace", size=11, color="#1a2540"),
                bgcolor="rgba(255,255,255,.90)", bordercolor="#1565c0",
                borderwidth=1, borderpad=7, align="left"
            )

    # Secondaries
    for i, sec in enumerate(secondaries):
        color = PALETTE[(i+1) % len(PALETTE)]
        cm    = chart_modes_sec[i] if i < len(chart_modes_sec) else "Line"
        mode_s, shape_s, smooth_s = CHART_MODES.get(cm, ("lines","spline",1.3))
        fig.add_trace(go.Scatter(
            x=x_vals, y=df_plot[sec], name=sec,
            yaxis="y2" if i == 0 else "y2",
            mode=mode_s,
            line=dict(width=2.2, color=color, shape=shape_s, smoothing=smooth_s),
            marker=dict(size=5, color=color)
        ))

    sec_title = secondaries[0] if secondaries else ""
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="DM Mono,monospace", color="#1a2540", size=11),
        legend=dict(bgcolor="#f8f9fc", bordercolor="#d0daea", borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.22,
                    font=dict(size=11, color="#1a2540")),
        xaxis=dict(title="Date", gridcolor="#e8edf5", gridwidth=1, griddash="dash",
                   zeroline=False, showline=True, linecolor="#c5d0e0",
                   tickangle=-45, tickfont=dict(color="#2a3a55", size=10)),
        yaxis=dict(
            title=dict(text=f"<b>{primary}</b>", font=dict(color=PRIMARY_COLOR, size=12)),
            tickfont=dict(color=PRIMARY_COLOR, size=11),
            gridcolor="#e8edf5", gridwidth=1, griddash="dash",
            zeroline=True, zerolinecolor="#c5d0e0",
            showline=True, linecolor="#c5d0e0",
            range=y_range
        ),
        yaxis2=dict(
            title=dict(text=f"<b>{sec_title}</b>", font=dict(color=SECONDARY_COLOR, size=12)),
            tickfont=dict(color=SECONDARY_COLOR, size=11),
            overlaying="y", side="right",
            gridcolor="#e8edf5", zeroline=False,
            showline=True, linecolor="#c5d0e0",
            range=y2_range
        ),
        margin=dict(l=70, r=80, t=60, b=90)
    )
    return fig, eq, r2v, ok


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if file:
    df_raw = pd.read_excel(file, header=0).dropna(how="all")
    df = df_raw.copy()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── Reset state on new file ──
    file_id = file.name + str(file.size)
    if st.session_state.get("_loaded_file_id") != file_id:
        st.session_state.update({
            "_loaded_file_id":       file_id,
            "df2_filtered":          None,
            "df3_filtered_out":      None,
            "df4_clean":             None,
            "preprocess_applied":    False,
            "outlier_report":        {},
            "run_preprocess":        False,
            "active_dataset_choice": "DF1",
        })

    # ── Fill strategy options ──
    FILL_OPTIONS = [
        "1. Linear Interpolation  — fills gaps by drawing a straight line between known values",
        "2. Time-Series Interpolation  — like linear but uses actual dates/time as the axis",
        "3. Forward Fill (ffill)  — copies the last known value forward into each gap",
        "4. Backward Fill (bfill)  — copies the next known value backward into each gap",
        "5. Conditional / Grouped Interpolation  — fills using local window mean (prev 7 + next 7)",
        "6. Spline / Polynomial Interpolation  — fits a smooth curve through known points to fill gaps",
        "7. Column Mean  — replaces every blank with the column's average (excluding blanks)",
    ]

    FILL_DESCRIPTIONS = {
        FILL_OPTIONS[0]: "Draws a straight line between the two nearest known values on either side of a gap. Simple and accurate for smoothly changing data.",
        FILL_OPTIONS[1]: "Same as linear but uses the actual timestamps. Best for unevenly spaced time-series where gaps of different durations should be treated proportionally.",
        FILL_OPTIONS[2]: "Fills each blank with the last valid value before it. Good when data changes slowly and the most recent reading is still valid.",
        FILL_OPTIONS[3]: "Fills each blank with the next valid value after it. Useful when you know future data and want to back-propagate it.",
        FILL_OPTIONS[4]: "Fills each blank using the average of up to 7 real values before it and up to 7 after it. Balances local context from both sides.",
        FILL_OPTIONS[5]: "Fits a smooth curve (cubic spline) through all known data points and reads off estimated values at the blank positions. Best for smooth, continuous measurements.",
        FILL_OPTIONS[6]: "Replaces every blank with the overall column average. Simple baseline — use when no pattern is expected.",
    }

    def apply_fill_col(series, strategy, date_index=None):
        """Apply selected fill strategy to a series. Returns filled series."""
        s = series.copy().astype(float)
        col_mean = float(np.nanmean(s)) if not np.all(np.isnan(s)) else np.nan

        if strategy == FILL_OPTIONS[0]:
            # 1. Linear interpolation
            s = s.interpolate(method="linear", limit_direction="both")
            s = s.fillna(col_mean)

        elif strategy == FILL_OPTIONS[1]:
            # 2. Time-series interpolation (uses index as time axis)
            if date_index is not None:
                try:
                    tmp = pd.Series(s.values, index=date_index)
                    tmp = tmp.interpolate(method="time", limit_direction="both")
                    s   = pd.Series(tmp.values, index=s.index)
                except Exception:
                    s = s.interpolate(method="linear", limit_direction="both")
            else:
                s = s.interpolate(method="linear", limit_direction="both")
            s = s.fillna(col_mean)

        elif strategy == FILL_OPTIONS[2]:
            # 3. Forward fill
            s = s.ffill()
            s = s.bfill()           # bfill handles leading NaNs
            s = s.fillna(col_mean)

        elif strategy == FILL_OPTIONS[3]:
            # 4. Backward fill
            s = s.bfill()
            s = s.ffill()           # ffill handles trailing NaNs
            s = s.fillna(col_mean)

        elif strategy == FILL_OPTIONS[4]:
            # 5. Conditional/grouped: local window mean (prev7 + next7 actual values)
            arr = s.values.copy()
            for i in range(len(arr)):
                if not np.isnan(arr[i]):
                    continue
                prev7 = arr[:i][~np.isnan(arr[:i])][-7:]
                next7 = arr[i+1:][~np.isnan(arr[i+1:])][:7]
                local = np.concatenate([prev7, next7])
                arr[i] = np.mean(local) if len(local) > 0 else col_mean
            s = pd.Series(arr, index=s.index)

        elif strategy == FILL_OPTIONS[5]:
            # 6. Spline interpolation (cubic)
            try:
                from scipy.interpolate import CubicSpline
                valid = s.dropna()
                if len(valid) >= 4:
                    cs   = CubicSpline(valid.index.tolist(), valid.values)
                    mask = s.isna()
                    s[mask] = cs(s.index[mask].tolist())
                else:
                    s = s.interpolate(method="linear", limit_direction="both")
            except Exception:
                s = s.interpolate(method="linear", limit_direction="both")
            s = s.fillna(col_mean)

        elif strategy == FILL_OPTIONS[6]:
            # 7. Column mean
            s = s.fillna(col_mean)

        return s

    # ── Raw stats for display ──
    orig_rows    = len(df)
    orig_cols    = len(df.columns)
    orig_nulls   = int(df[numeric_cols].isnull().sum().sum())
    orig_non_num = 0
    orig_zeros   = 0
    for col in df.columns[1:]:
        coerced = pd.to_numeric(df_raw[col], errors='coerce')
        orig_non_num += max(0, int(coerced.isna().sum()) - int(df_raw[col].isna().sum()))
        orig_zeros   += int((df[col] == 0).sum()) if col in numeric_cols else 0
    orig_valid = orig_rows * orig_cols - orig_nulls - orig_non_num

    # ── Outlier counts for sigma cards (hybrid) ──
    outlier_counts_by_sigma = {}
    for sig in [1, 2, 3]:
        total_out = 0
        for col in numeric_cols:
            s = df[col].replace(0, np.nan).dropna()
            if len(s) < 3: continue
            mu, sv = s.mean(), s.std()
            cv = sv / abs(mu) if mu != 0 else float('inf')
            if len(s) >= 10 and (sv == 0 or cv <= 1.5):
                lo, hi = mu - sig * sv, mu + sig * sv
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = (s.max()*0.01, s.max()*10) if iqr == 0 else (q1-3*iqr, q3+3*iqr)
            total_out += int((df[col].notna() & (df[col] != 0) & ((df[col]<lo)|(df[col]>hi))).sum())
        outlier_counts_by_sigma[sig] = total_out

    # ── Build the 4 dataframes when Apply is clicked ──
    if st.session_state.get("run_preprocess"):
        _sigma      = int(st.session_state.get("sigma_n", 3))
        _apply_out  = bool(st.session_state.get("apply_out", False))
        _fill_strat = st.session_state.get("fill_strat", FILL_OPTIONS[3])

        # DF1 — Raw (already df, just store reference key)
        # DF2 — Filtered: zeros + nulls + non-numeric blanked, outliers KEPT
        _df2 = df.copy()
        for col in numeric_cols:
            _df2.loc[_df2[col] == 0, col] = np.nan
        # (non-numeric already NaN from pd.to_numeric coerce)

        # DF3 — Filtered + outliers: same as DF2 but also blank outliers
        _df3 = _df2.copy()
        _outlier_report = {}
        if _apply_out:
            for col in numeric_cols:
                s_col = _df3[col]
                s = s_col.dropna()
                if len(s) < 3: continue
                mu, sv = s.mean(), s.std()
                cv = sv / abs(mu) if mu != 0 else float('inf')
                if len(s) >= 10 and (sv == 0 or cv <= 1.5):
                    lo, hi = mu - _sigma * sv, mu + _sigma * sv
                    method = f"σ-rule ({_sigma}σ)"
                else:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = (s.max()*0.01, s.max()*10) if iqr == 0 else (q1-3*iqr, q3+3*iqr)
                    method = "IQR practical rule"
                out_mask = s_col.notna() & ((s_col < lo) | (s_col > hi))
                if out_mask.any():
                    _outlier_report[col] = {
                        "blanked": int(out_mask.sum()), "lower": round(lo,4),
                        "upper": round(hi,4), "mean": round(mu,4),
                        "std": round(sv,4), "method": method,
                    }
                    _df3.loc[out_mask, col] = np.nan

        # DF4 — Cleaned: fill all blanked cells in DF3
        _df4 = _df3.copy()
        _date_idx = pd.to_datetime(_df3[date_col]) if date_col in _df3.columns else None
        for col in numeric_cols:
            _df4[col] = apply_fill_col(_df4[col], _fill_strat, date_index=_date_idx)

        st.session_state["df2_filtered"]       = _df2.copy()
        st.session_state["df3_filtered_out"]   = _df3.copy()
        st.session_state["df4_clean"]          = _df4.copy()
        st.session_state["outlier_report"]     = _outlier_report
        st.session_state["preprocess_applied"] = True
        st.session_state["run_preprocess"]     = False

    # ── Load all 4 dataframes directly from session state ──
    preprocess_applied = st.session_state.get("preprocess_applied", False)
    outlier_report     = st.session_state.get("outlier_report", {})
    df1 = df.copy()                                          # DF1 always = raw df
    df2 = st.session_state.get("df2_filtered")              # DF2: zeros/null/non-num blanked
    df3 = st.session_state.get("df3_filtered_out")          # DF3: DF2 + outliers blanked
    df4 = st.session_state.get("df4_clean")                 # DF4: DF3 filled

    # ── Resolve df_active based on session state choice ──
    _adc = st.session_state.get("active_dataset_choice", "DF1")
    if   _adc == "DF2" and df2 is not None: df_active = df2.copy()
    elif _adc == "DF3" and df3 is not None: df_active = df3.copy()
    elif _adc == "DF4" and df4 is not None: df_active = df4.copy()
    else:                                   df_active = df1.copy()

    # ── Build labeled columns from df_active ──
    def make_labeled(source_df):
        num_cols = source_df.select_dtypes(include=np.number).columns.tolist()
        all_cols = list(source_df.columns)
        labeled  = [f"{all_cols.index(c)}-{c}" for c in num_cols]
        lbl2col  = {f"{all_cols.index(c)}-{c}": c for c in num_cols}
        return num_cols, labeled, lbl2col

    active_numeric_cols, active_numeric_labeled, active_label_to_col = make_labeled(df_active)
    # keep clean_numeric_cols for guard checks
    _ref = df4 if df4 is not None else df_active
    clean_numeric_cols, _, _ = make_labeled(_ref)

    # ── Badge map ──
    _badge_map = {
        "DF1": ("#5a6e8a", "DF1 — RAW DATA"),
        "DF2": ("#1a7a4a", "DF2 — FILTERED DATA"),
        "DF3": ("#7c3aed", "DF3 — FILTERED + OUTLIERS"),
        "DF4": ("#1565c0", "DF4 — CLEANED DATA"),
    }
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🗂️  Raw Data", "⚙️  Preprocess", "📊  Correlation", "📈  Analysis", "📑  Report"
    ])

    # ══════════════════════════════
    # TAB 0 — RAW DATA
    # ══════════════════════════════
    with tab0:
        st.markdown('<div class="section-label">RAW UPLOADED DATA</div>', unsafe_allow_html=True)
        total_rows = len(df_raw)
        total_cols = len(df_raw.columns)
        null_count = int(df_raw.isnull().sum().sum())
        non_num_count = 0
        for col in df_raw.columns[1:]:
            non_num_count += max(0,
                int(pd.to_numeric(df_raw[col], errors='coerce').isna().sum())
                - int(df_raw[col].isna().sum()))
        valid_cells = total_rows * total_cols - null_count - non_num_count

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Total Rows</div>
                <div class="metric-value">{total_rows}</div><div class="metric-sub">Excl. header</div></div>
            <div class="metric-card"><div class="metric-label">Total Columns</div>
                <div class="metric-value">{total_cols}</div><div class="metric-sub">Incl. date col</div></div>
            <div class="metric-card"><div class="metric-label">Null / Empty Cells</div>
                <div class="metric-value">{null_count}</div><div class="metric-sub">Across all cols</div></div>
            <div class="metric-card"><div class="metric-label">Non-Numeric Cells</div>
                <div class="metric-value">{non_num_count}</div><div class="metric-sub">In numeric cols</div></div>
            <div class="metric-card"><div class="metric-label">Valid Data Cells</div>
                <div class="metric-value">{valid_cells}</div>
                <div class="metric-sub">Rows×Cols − Null − Non-Num</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:8px;">DATA PREVIEW</div>', unsafe_allow_html=True)
        st.dataframe(df_raw, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label" style="margin-top:18px;">NULL & DATA QUALITY PER COLUMN</div>', unsafe_allow_html=True)
        qrows = []
        for col in df_raw.columns:
            nc  = int(df_raw[col].isnull().sum())
            nnc = 0 if col == date_col else max(0,
                int(pd.to_numeric(df_raw[col], errors='coerce').isna().sum()) - nc)
            qrows.append({"Column": col,
                          "Dtype": "Date" if col == date_col else str(df_raw[col].dtype),
                          "Null Count": nc, "Non-Numeric Count": nnc,
                          "Fill Rate %": f"{100*(1 - nc/max(total_rows,1)):.1f}%"})
        st.dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)

    # ══════════════════════════════
    # TAB 1 — PREPROCESS
    # ══════════════════════════════
    with tab1:

        # ════════════════════════════════════════
        # ROW 1: BEFORE card + AFTER card side by side
        # ════════════════════════════════════════
        col_before, col_after = st.columns(2)

        # ── BEFORE card ──
        with col_before:
            st.markdown('<div class="section-label">📋 BEFORE PREPROCESSING</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #d0daea;border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
                    <div><div class="metric-label">TOTAL ROWS</div>
                        <div class="metric-value">{orig_rows}</div>
                        <div class="metric-sub">Raw uploaded rows</div></div>
                    <div><div class="metric-label">TOTAL COLUMNS</div>
                        <div class="metric-value">{orig_cols}</div>
                        <div class="metric-sub">Incl. date column</div></div>
                    <div><div class="metric-label">NULL / EMPTY CELLS</div>
                        <div class="metric-value" style="color:#c0392b;">{orig_nulls}</div>
                        <div class="metric-sub">Across all columns</div></div>
                    <div><div class="metric-label">NON-NUMERIC CELLS</div>
                        <div class="metric-value" style="color:#c0392b;">{orig_non_num}</div>
                        <div class="metric-sub">In numeric columns</div></div>
                    <div><div class="metric-label">ZERO VALUE CELLS</div>
                        <div class="metric-value" style="color:#d97706;">{orig_zeros}</div>
                        <div class="metric-sub">Will be blanked</div></div>
                    <div><div class="metric-label">VALID DATA CELLS</div>
                        <div class="metric-value">{orig_valid}</div>
                        <div class="metric-sub">Rows × Cols − Null − Non-Num</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── AFTER card ──
        with col_after:
            st.markdown('<div class="section-label">✅ AFTER PREPROCESSING</div>', unsafe_allow_html=True)
            if preprocess_applied:
                _cnc       = df4.select_dtypes(include=np.number).columns.tolist()
                after_nulls       = int(df4[_cnc].isnull().sum().sum()) if _cnc else 0
                after_rows        = len(df4)
                after_cols        = len(df4.columns)
                tot_out_blanked   = sum(v["blanked"] for v in outlier_report.values())
                tot_blanked       = orig_nulls + orig_non_num + orig_zeros + tot_out_blanked
                cells_filled      = tot_blanked - after_nulls
                after_valid       = after_rows * after_cols - after_nulls
                st.markdown(f"""
                <div style="background:#fff;border:2px solid #1a7a4a;border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
                        <div><div class="metric-label">TOTAL ROWS</div>
                            <div class="metric-value">{after_rows}</div>
                            <div class="metric-sub">After preprocessing</div></div>
                        <div><div class="metric-label">TOTAL BLANKED</div>
                            <div class="metric-value" style="color:#c0392b;">{tot_blanked}</div>
                            <div class="metric-sub">Null+non-num+zero+outlier</div></div>
                        <div><div class="metric-label">CELLS FILLED</div>
                            <div class="metric-value" style="color:#1a7a4a;">{cells_filled}</div>
                            <div class="metric-sub">By fill strategy</div></div>
                        <div><div class="metric-label">REMAINING NULLS</div>
                            <div class="metric-value">{after_nulls}</div>
                            <div class="metric-sub">After fill</div></div>
                        <div><div class="metric-label">OUTLIER CELLS BLANKED</div>
                            <div class="metric-value" style="color:#c0392b;">{tot_out_blanked}</div>
                            <div class="metric-sub">By sigma rule</div></div>
                        <div><div class="metric-label">VALID DATA CELLS</div>
                            <div class="metric-value" style="color:#1565c0;">{after_valid}</div>
                            <div class="metric-sub">After preprocessing</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                if outlier_report:
                    with st.expander("📊 View Outlier Detail Report"):
                        _s  = int(st.session_state.get("sigma_n", 3))
                        rr  = [{"Column": c,
                                "Method": i.get("method","σ-rule"),
                                "Mean": i["mean"], "Std Dev": i["std"],
                                "Lower Bound": i["lower"],
                                "Upper Bound": i["upper"],
                                "Cells Blanked": i["blanked"]}
                               for c, i in outlier_report.items()]
                        st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)
            else:
                st.markdown("""
                <div style="background:#f8fafc;border:1px dashed #c5d0e0;border-radius:10px;
                    padding:52px 20px;text-align:center;color:#5a6e8a;font-size:.82rem;">
                    ⚙️ Configure settings below and click<br><b>Apply Preprocessing</b> to see results here.
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='margin:26px 0;border:none;border-top:2px solid #d0daea;'>", unsafe_allow_html=True)

        # ════════════════════════════════════════
        # SIGMA SELECTION + OUTLIER COUNT CARDS
        # ════════════════════════════════════════
        st.markdown('<div class="section-label">STEP 1 — OUTLIER DETECTION & BLANKING</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            '<b>Hybrid outlier detection:</b> For each column the system automatically selects the best method:<br>'
            '• <b>Sigma rule</b> — used when data is continuous and well-distributed (CV ≤ 1.5, ≥10 values)<br>'
            '• <b>IQR practical rule</b> — used for sparse, bimodal, or ON/OFF columns (e.g. flow = 0 or 300–500) '
            'where sigma becomes misleadingly large and misses real anomalies like 0.652 in a 300–500 range<br>'
            'Zero values are always blanked first, then outlier detection runs on the remaining values.'
            '</div>',
            unsafe_allow_html=True)

        sig_col, chk_col = st.columns([3, 1])
        with sig_col:
            st.radio("Sigma Level", [1, 2, 3], index=2,
                     format_func=lambda v: f"{v}σ  ({'Aggressive' if v==1 else 'Moderate' if v==2 else 'Conservative'})",
                     key="sigma_n", horizontal=True)
        with chk_col:
            st.checkbox("Enable Outlier Blanking", value=False, key="apply_out")

        cur_sigma = int(st.session_state.get("sigma_n", 3))
        st.markdown(f"""
        <div class="metric-grid-3" style="margin-top:12px;">
            <div class="metric-card" style="border-color:{'#1565c0' if cur_sigma==1 else '#d0daea'};">
                <div class="metric-label">OUTLIERS AT 1σ</div>
                <div class="metric-value" style="color:{'#c0392b' if cur_sigma==1 else '#1a2540'};">{outlier_counts_by_sigma[1]}</div>
                <div class="metric-sub">{'◀ Currently selected' if cur_sigma==1 else 'Aggressive — removes most'}</div>
            </div>
            <div class="metric-card" style="border-color:{'#1565c0' if cur_sigma==2 else '#d0daea'};">
                <div class="metric-label">OUTLIERS AT 2σ</div>
                <div class="metric-value" style="color:{'#c0392b' if cur_sigma==2 else '#1a2540'};">{outlier_counts_by_sigma[2]}</div>
                <div class="metric-sub">{'◀ Currently selected' if cur_sigma==2 else 'Moderate removal'}</div>
            </div>
            <div class="metric-card" style="border-color:{'#1565c0' if cur_sigma==3 else '#d0daea'};">
                <div class="metric-label">OUTLIERS AT 3σ</div>
                <div class="metric-value" style="color:{'#c0392b' if cur_sigma==3 else '#1a2540'};">{outlier_counts_by_sigma[3]}</div>
                <div class="metric-sub">{'◀ Currently selected' if cur_sigma==3 else 'Conservative — removes least'}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ════════════════════════════════════════
        # DATA QUALITY CHECK: columns must have ≥50% data
        # ════════════════════════════════════════
        _fill_rate = {col: 1 - df[col].isna().sum() / max(len(df), 1) for col in numeric_cols}
        _low_cols  = [c for c, r in _fill_rate.items() if r < 0.5]
        _can_preprocess = len(_low_cols) == 0

        if _low_cols:
            st.markdown(
                f'<div class="info-box" style="border-left-color:#c0392b;background:#fef2f2;">'
                f'<b>⚠️ Data Quality Warning:</b> {len(_low_cols)} column(s) have less than 50% valid data. '
                f'Preprocessing is disabled until these columns are either removed or have sufficient data.<br>'
                f'<b>Columns below 50% fill rate:</b> '
                f'{", ".join(f"{c} ({_fill_rate[c]*100:.0f}%)" for c in _low_cols[:10])}'
                + ("..." if len(_low_cols) > 10 else "") +
                f'</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box" style="border-left-color:#1a7a4a;background:#f0fdf4;">'
                '✅ All columns have ≥50% valid data. Preprocessing is enabled.'
                '</div>', unsafe_allow_html=True)

        # Show per-column fill rate table
        with st.expander("📊 View column fill rate details"):
            _fr_rows = [{"Column": c, "Valid Rows": int(_fill_rate[c]*len(df)),
                         "Total Rows": len(df), "Fill Rate": f"{_fill_rate[c]*100:.1f}%",
                         "Status": "✅ OK" if _fill_rate[c] >= 0.5 else "❌ Below 50%"}
                        for c in numeric_cols]
            st.dataframe(pd.DataFrame(_fr_rows), use_container_width=True, hide_index=True)

        st.markdown("<hr style='margin:18px 0;border:none;border-top:1px solid #d0daea;'>", unsafe_allow_html=True)

        # ════════════════════════════════════════
        # STEP 2 — FILL STRATEGY (7 options)
        # ════════════════════════════════════════
        st.markdown('<div class="section-label" style="margin-top:4px;">STEP 2 — FILL STRATEGY  <span style="font-size:.65rem;color:#5a6e8a;font-weight:400;">(applied to all blanked cells: zeros · nulls · non-numeric · outliers)</span></div>', unsafe_allow_html=True)

        _cur_fill_idx = FILL_OPTIONS.index(st.session_state.get("fill_strat", FILL_OPTIONS[0])) \
                        if st.session_state.get("fill_strat") in FILL_OPTIONS else 0
        _sel_fill = st.radio(
            "Fill Strategy",
            FILL_OPTIONS,
            index=_cur_fill_idx,
            key="fill_strat",
            label_visibility="collapsed"
        )
        # Show plain-English explanation of selected method
        _desc = FILL_DESCRIPTIONS.get(_sel_fill, "")
        if _desc:
            st.markdown(
                f'<div style="background:#f0f5ff;border-left:3px solid #1565c0;border-radius:6px;'
                f'padding:10px 16px;font-size:.78rem;color:#2a3a55;margin-top:6px;line-height:1.6;">'
                f'📌 <b>How this works:</b> {_desc}</div>',
                unsafe_allow_html=True)

        # ════════════════════════════════════════
        # APPLY BUTTON (disabled if <50% data in any column)
        # ════════════════════════════════════════
        st.markdown("<div style='margin-top:24px;'>", unsafe_allow_html=True)
        if _can_preprocess:
            if st.button("⚙️  Apply Preprocessing", use_container_width=True, key="btn_preprocess"):
                st.session_state["run_preprocess"] = True
                st.rerun()
        else:
            st.markdown(
                '<div style="opacity:0.5;cursor:not-allowed;">'
                '<div style="background:#9ca3af;color:#fff;border-radius:8px;padding:10px 24px;'
                'font-family:Syne,sans-serif;font-size:.78rem;font-weight:600;letter-spacing:.08em;'
                'text-align:center;">⚙️  Apply Preprocessing (Disabled — fix column data first)</div>'
                '</div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if preprocess_applied:
            st.success("✅ Preprocessing applied — scroll up to see the After card results.")

        st.markdown("<hr style='margin:26px 0;border:none;border-top:2px solid #d0daea;'>", unsafe_allow_html=True)

        # ════════════════════════════════════════
        # DATASET CHOICE (3 options) + COLOURED TABLE
        # ════════════════════════════════════════
        st.markdown('<div class="section-label">CHOOSE ACTIVE DATASET  —  used by Correlation, Analysis & Report tabs</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Select which version of the data all other tabs will use.</div>', unsafe_allow_html=True)

        DATASET_OPTIONS = [
            "📋  DF1 — RAW DATA  (original uploaded data, no changes)",
            "🟢  DF2 — FILTERED DATA  (zeros + nulls + non-numeric blanked, outliers kept)",
            "🟣  DF3 — FILTERED + OUTLIERS  (zeros + nulls + non-numeric + outliers blanked)",
            "🔵  DF4 — CLEANED DATA  (all blanked cells filled per selected strategy)",
        ]
        _opt_map  = {DATASET_OPTIONS[0]:"DF1", DATASET_OPTIONS[1]:"DF2",
                     DATASET_OPTIONS[2]:"DF3", DATASET_OPTIONS[3]:"DF4"}
        _key_list = ["DF1","DF2","DF3","DF4"]
        _cur_adc  = st.session_state.get("active_dataset_choice","DF1")
        _cur_idx  = _key_list.index(_cur_adc) if _cur_adc in _key_list else 0

        sel_opt    = st.radio("Active Dataset", DATASET_OPTIONS, index=_cur_idx, key="dataset_choice_radio")
        new_choice = _opt_map[sel_opt]
        if new_choice in ("DF2","DF3","DF4") and not preprocess_applied:
            st.warning("⚠️ Available only after clicking Apply Preprocessing. Showing DF1 (Raw) instead.")
            new_choice = "DF1"

        if new_choice != st.session_state.get("active_dataset_choice","DF1"):
            st.session_state["active_dataset_choice"] = new_choice
            st.rerun()

        new_choice = st.session_state.get("active_dataset_choice","DF1")
        _bc, _bt   = _badge_map.get(new_choice, ("#5a6e8a","DF1 — RAW DATA"))
        _preview_label = {
            "DF1": "DF1 — RAW DATA PREVIEW",
            "DF2": "DF2 — FILTERED DATA PREVIEW  (🟢 green = blanked zero/null/non-numeric  |  🟣 purple = outlier cells kept)",
            "DF3": "DF3 — FILTERED+OUTLIERS PREVIEW  (🟢 green = zero/null/non-num  |  🟣 purple = outlier-blanked)",
            "DF4": "DF4 — CLEANED DATA PREVIEW  (🔵 blue = cells that were filled)",
        }.get(new_choice, "DATA PREVIEW")

        st.markdown(
            f'<div style="background:{_bc};color:#fff;font-family:Syne,sans-serif;font-size:.72rem;'
            f'font-weight:700;letter-spacing:.12em;padding:8px 20px;border-radius:6px;'
            f'display:inline-block;margin:12px 0 6px;">ACTIVE: {_bt}</div>',
            unsafe_allow_html=True)
        st.markdown(f'<div class="section-label" style="margin-top:4px;">{_preview_label}</div>', unsafe_allow_html=True)

        def _render_styled(source_df, style_fn):
            # Replace NaN/None with empty string BEFORE styling
            clean_df = source_df.copy().fillna("")

            styled = clean_df.style.apply(style_fn, axis=None)

            st.dataframe(styled, use_container_width=True, hide_index=True)

        if new_choice == "DF1":
            st.dataframe(df_active.style.format(na_rep=""), use_container_width=True, hide_index=True)

        elif new_choice == "DF2" and df2 is not None:
            _sigma_disp   = int(st.session_state.get("sigma_n", 3))
            _outlier_mask = pd.DataFrame(False, index=df2.index, columns=numeric_cols)
            for _c in numeric_cols:
                _s = df2[_c].dropna()
                if len(_s) < 3: continue
                _mu, _sv = _s.mean(), _s.std()
                _cv = _sv / abs(_mu) if _mu != 0 else float("inf")
                if len(_s) >= 10 and (_sv == 0 or _cv <= 1.5):
                    _lo, _hi = _mu - _sigma_disp * _sv, _mu + _sigma_disp * _sv
                else:
                    _q1, _q3 = _s.quantile(0.25), _s.quantile(0.75)
                    _iqr = _q3 - _q1
                    _lo, _hi = (_s.max()*0.01, _s.max()*10) if _iqr==0 else (_q1-3*_iqr, _q3+3*_iqr)
                _outlier_mask[_c] = df2[_c].notna() & ((df2[_c] < _lo) | (df2[_c] > _hi))
            _df2_nan = df2[numeric_cols].isna()
            def _style_df2(df_s):
                styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
                for c in df_s.columns:
                    if c in _df2_nan.columns:
                        for idx in df_s.index:
                            if _df2_nan.at[idx, c]:
                                styles.at[idx, c] = "background-color:#d4edda;color:#155724;"
                            elif _outlier_mask.at[idx, c]:
                                styles.at[idx, c] = "background-color:#e9d8fd;color:#44337a;"
                return styles
            _render_styled(df2, _style_df2)

        elif new_choice == "DF3" and df3 is not None:
            _df2_nan = df2[numeric_cols].isna() if df2 is not None else                        pd.DataFrame(False, index=df3.index, columns=numeric_cols)
            _df3_nan = df3[numeric_cols].isna()
            def _style_df3(df_s):
                styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
                for c in df_s.columns:
                    if c in _df3_nan.columns:
                        for idx in df_s.index:
                            is_df3 = _df3_nan.at[idx, c]
                            is_df2 = _df2_nan.at[idx, c] if c in _df2_nan.columns else False
                            if is_df3 and not is_df2:
                                styles.at[idx, c] = "background-color:#e9d8fd;color:#44337a;"
                            elif is_df3 and is_df2:
                                styles.at[idx, c] = "background-color:#d4edda;color:#155724;"
                return styles
            _render_styled(df3, _style_df3)

        elif new_choice == "DF4" and df4 is not None:
            _was_blank = df3[numeric_cols].isna() if df3 is not None else                          pd.DataFrame(False, index=df4.index, columns=numeric_cols)
            def _style_df4(df_s):
                styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
                for c in df_s.columns:
                    if c in _was_blank.columns:
                        styles[c] = _was_blank[c].map(
                            lambda x: "background-color:#cce5ff;color:#004085;" if x else "")
                return styles
            _render_styled(df4, _style_df4)

        else:
            st.dataframe(df_active.style.format(na_rep=""), use_container_width=True, hide_index=True)

    # ══════════════════════════════
    # TAB 2 — CORRELATION
    # ══════════════════════════════
    with tab2:
        st.markdown('<div class="section-label">CORRELATION INTELLIGENCE</div>', unsafe_allow_html=True)
        if len(active_numeric_cols) < 2:
            st.warning("Not enough numeric columns in the active dataset.")
            st.stop()

        _adc2 = st.session_state.get("active_dataset_choice", "RAW")
        _bc, _bt = _badge_map.get(_adc2, ("#5a6e8a","RAW DATA"))
        st.markdown(
            f'<div style="background:{_bc};color:#fff;font-family:Syne,sans-serif;font-size:.7rem;'
            f'font-weight:700;letter-spacing:.12em;padding:6px 14px;border-radius:6px;display:inline-block;margin-bottom:12px;">'
            f'USING: {_bt}</div>', unsafe_allow_html=True)

        corr = df_active[active_numeric_cols].corr()

        st.markdown('<div class="section-label" style="margin-top:6px;">FULL CORRELATION MATRIX</div>', unsafe_allow_html=True)
        fig_full = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont={"size": 11, "color": "#1a2540"}
        ))
        fig_full.update_layout(height=850, margin=dict(l=40,r=40,t=30,b=40),
            paper_bgcolor="#f4f6fa", plot_bgcolor="#f4f6fa", font=dict(color="#1a2540"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#1a2540")),
            yaxis=dict(tickfont=dict(size=10, color="#1a2540")))
        st.plotly_chart(fig_full, use_container_width=True, config={"scrollZoom": True})

        st.markdown('<div class="section-label" style="margin-top:18px;">TARGET CORRELATION — BAR CHART</div>', unsafe_allow_html=True)
        bc1, bc2 = st.columns([3, 1])
        with bc1:
            target_lbl = st.selectbox("Select Target Variable", active_numeric_labeled, key="corr_target")
            target_col = active_label_to_col[target_lbl]
        with bc2:
            top_n = st.slider("Top N", 5, len(active_numeric_cols), min(15, len(active_numeric_cols)), key="top_n_corr")

        corr_series = corr[target_col].drop(labels=[target_col])
        corr_top    = corr_series.abs().sort_values(ascending=False).head(top_n)
        corr_vals   = corr_series[corr_top.index]
        bar_colors  = ["#1565c0" if v >= 0 else "#c0392b" for v in corr_vals.values]

        fig_bar = go.Figure(go.Bar(
            x=corr_vals.index.tolist(), y=corr_vals.values,
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in corr_vals.values],
            textposition="outside", textfont=dict(size=11, color="#1a2540"), width=0.6))
        fig_bar.add_hline(y=0, line_color="#888", line_width=1)
        for yv, lbl, clr in [(0.7,"Strong +0.7","#1a7a4a"),(-0.7,"Strong −0.7","#c0392b")]:
            fig_bar.add_hline(y=yv, line_dash="dash", line_color=clr, line_width=1,
                              annotation_text=lbl, annotation_font_color=clr, annotation_position="right")
        for yv in [0.3, -0.3]:
            fig_bar.add_hline(y=yv, line_dash="dot", line_color="#b8860b", line_width=1)
        fig_bar.update_layout(
            height=420,
            title=dict(text=f"Correlation with <b>{target_col}</b>",
                       font=dict(family="Syne,sans-serif",size=14,color="#1a2540"), x=0.5, xanchor="center"),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(family="DM Mono,monospace", color="#1a2540", size=11),
            yaxis=dict(title="Correlation Coefficient", range=[-1.2,1.2],
                       gridcolor="#e8edf5", zeroline=False, tickfont=dict(color="#1a2540")),
            xaxis=dict(tickangle=-45, tickfont=dict(color="#1a2540", size=10), gridcolor="#e8edf5"),
            margin=dict(l=60,r=80,t=60,b=130), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

 
    # ══════════════════════════════
    # TAB 3 — ANALYSIS
    # ══════════════════════════════
    with tab3:
        st.markdown('<div class="section-label">DYNAMIC VARIABLE ANALYSIS</div>', unsafe_allow_html=True)

        # ── DEBUG: confirm active dataset ──
        _adc_dbg = st.session_state.get("active_dataset_choice", "DF1")
        _bc_dbg, _bt_dbg = _badge_map.get(_adc_dbg, ("#5a6e8a","DF1 — RAW DATA"))
        st.markdown(
            f'<div style="background:{_bc_dbg};color:#fff;font-family:Syne,sans-serif;font-size:.7rem;'
            f'font-weight:700;letter-spacing:.12em;padding:6px 14px;border-radius:6px;display:inline-block;margin-bottom:8px;">'
            f'USING: {_bt_dbg}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Debug — active dataset info"):
            st.write(f"active_dataset_choice = **{_adc_dbg}**")
            st.write(f"df_active shape = {df_active.shape}")
            st.write(f"df1 nulls in numeric = {int(df1[numeric_cols].isnull().sum().sum())}")
            if df2 is not None: st.write(f"df2 nulls in numeric = {int(df2[numeric_cols].isnull().sum().sum())}")
            if df3 is not None: st.write(f"df3 nulls in numeric = {int(df3[numeric_cols].isnull().sum().sum())}")
            if df4 is not None: st.write(f"df4 nulls in numeric = {int(df4[numeric_cols].isnull().sum().sum())}")
            st.write(f"df_active nulls in numeric = {int(df_active[numeric_cols].isnull().sum().sum())}")
            st.write(f"preprocess_applied = {preprocess_applied}")
            st.write("**Null counts per column in df_active (top 20 by null count):**")
            _null_series = df_active[numeric_cols].isnull().sum().sort_values(ascending=False).head(20)
            st.dataframe(_null_series.reset_index().rename(columns={"index":"Column", 0:"Nulls"}), hide_index=True)

        date_list = sorted(df_active[date_col].dt.date.unique())

        r1, r2, r3, r4 = st.columns(4)
        with r1: start_date = st.selectbox("Start Date", date_list, key="sd")
        with r2: end_date   = st.selectbox("End Date", date_list, index=len(date_list)-1, key="ed")
        with r3:
            prim_lbl = st.selectbox("Primary Variable", active_numeric_labeled, key="pv")
            primary  = active_label_to_col[prim_lbl]
        with r4:
            sec_lbl   = st.selectbox("Secondary Variable", active_numeric_labeled, index=min(1, len(active_numeric_labeled)-1), key="sv")
            secondary = active_label_to_col[sec_lbl]

        # Chart type + trendline row
        ct1, ct2, ct3, ct4 = st.columns(4)
        with ct1:
            chart_mode_primary = st.selectbox(
                "Primary Chart Type",
                list(CHART_MODES.keys()), key="chart_mode_p")
        with ct2:
            chart_mode_secondary = st.selectbox(
                "Secondary Chart Type",
                list(CHART_MODES.keys()), index=2, key="chart_mode_s")
        with ct3:
            trendline_type = st.selectbox(
                "Trendline Type (Primary)",
                ["Exponential","Linear","Logarithmic","Power"], key="tl_type")
        with ct4:
            show_equation = st.checkbox("Show Equation & R²", value=True, key="show_eq")

        if start_date > end_date:
            st.error("Start Date cannot be after End Date.")
            st.stop()

        df_live=df_active
        st.write(df_live)
        df_f = df_live[(df_live[date_col].dt.date >= start_date) & (df_live[date_col].dt.date <= end_date)]
        if len(df_live) < 2:
            st.warning("Not enough data in selected range.")
            st.stop()

        df_live[date_col] = pd.to_datetime(df_live[date_col], errors='coerce')

        df_f = df_live[
        (df_live[date_col] >= pd.to_datetime(start_date)) &
        (df_live[date_col] <= pd.to_datetime(end_date))
        ]

        df_f = df_f.dropna(subset=[date_col, primary, secondary])

        st.write(f"df_active shape = {df_f.shape}")
        x = df_f[primary]
        y = df_f[secondary]

        # Regression (x vs y)
        valid_mask = x.notna() & y.notna()
        xv, yv_reg = x[valid_mask].values, y[valid_mask].values
        if len(xv) >= 2:
            slope, intercept, r_val, p, std_err = linregress(xv, yv_reg)
            r2_stat = r_val**2; n = len(xv)
            adj_r2  = 1 - (1 - r2_stat)*(n-1)/(n-2) if n > 2 else r2_stat
        else:
            slope = intercept = r_val = std_err = 0; r2_stat = adj_r2 = 0; n = 0
        corr_val = x.corr(y)
        strength = ("Weak" if abs(corr_val or 0) < 0.3 else
                    "Moderate" if abs(corr_val or 0) < 0.7 else "Strong")
        relation = "Positive" if (corr_val or 0) > 0 else "Negative"

        def pct_ch(s):
            s2 = s.dropna()
            if len(s2) < 2: return None,None,None,None
            sv, ev = s2.iloc[0], s2.iloc[-1]
            if sv == 0: return sv,ev,None,None
            return sv, ev, ((ev-sv)/abs(sv))*100, (s2.pct_change()*100).mean()

        px_s, px_e, px_c, px_a = pct_ch(x)
        py_s, py_e, py_c, py_a = pct_ch(y)

        def fc(v):
            if v is None: return "N/A","neu"
            return (f"+{v:.2f}%" if v>=0 else f"{v:.2f}%"), ("pos" if v>=0 else "neg")

        px_cs, px_cl = fc(px_c); px_as, px_al = fc(px_a)
        py_cs, py_cl = fc(py_c); py_as, py_al = fc(py_a)

        # Metrics
        st.markdown(f"""
        <div class="metric-grid-4">
            <div class="metric-card"><div class="metric-label">Correlation</div>
                <div class="metric-value">{corr_val:.3f}</div><div class="metric-sub">{strength} {relation}</div></div>
            <div class="metric-card"><div class="metric-label">R²</div>
                <div class="metric-value">{r2_stat:.4f}</div><div class="metric-sub">Adj R² {adj_r2:.4f}</div></div>
            <div class="metric-card"><div class="metric-label">Slope</div>
                <div class="metric-value">{slope:.4f}</div><div class="metric-sub">Std Err {std_err:.4f}</div></div>
            <div class="metric-card"><div class="metric-label">Data Points</div>
                <div class="metric-value">{n}</div><div class="metric-sub">{start_date} → {end_date}</div></div>
        </div>""", unsafe_allow_html=True)

        # Period cards
        st.markdown('<div class="section-label" style="margin-top:8px;">PERIOD CHANGE ANALYTICS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="change-grid">
            <div class="change-card">
                <div class="change-card-title">📌 {primary}</div>
                <div class="change-row"><span class="change-key">Opening</span><span class="change-val neu">{px_s:.4g}</span></div>
                <div class="change-row"><span class="change-key">Closing</span><span class="change-val neu">{px_e:.4g}</span></div>
                <div class="change-row"><span class="change-key">Total % Change</span><span class="change-val {px_cl}">{px_cs}</span></div>
            </div>
            <div class="change-card">
                <div class="change-card-title">📌 {secondary}</div>
                <div class="change-row"><span class="change-key">Opening</span><span class="change-val neu">{py_s:.4g}</span></div>
                <div class="change-row"><span class="change-key">Closing</span><span class="change-val neu">{py_e:.4g}</span></div>
                <div class="change-row"><span class="change-key">Total % Change</span><span class="change-val {py_cl}">{py_cs}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Axis scale
        st.markdown('<div class="section-label" style="margin-top:8px;">CHART AXIS SCALE CUSTOMIZATION</div>', unsafe_allow_html=True)
        xmn = float(x.min()) if not x.isna().all() else 0.0
        xmx = float(x.max()) if not x.isna().all() else 100.0
        ymn = float(y.min()) if not y.isna().all() else 0.0
        ymx = float(y.max()) if not y.isna().all() else 100.0
        xp  = (xmx-xmn)*0.1 or abs(xmx)*0.1 or 10
        yp  = (ymx-ymn)*0.1 or abs(ymx)*0.1 or 10

        ac1,ac2,ac3,ac4,ac5 = st.columns(5)
        with ac1:
            lbl = f"Y-Left Min ({primary[:10]}...)" if len(primary)>10 else f"Y-Left Min ({primary})"
            y1_min = st.number_input(lbl, value=round(xmn-xp,4), format="%.4f", key="y1min")
        with ac2:
            lbl = f"Y-Left Max ({primary[:10]}...)" if len(primary)>10 else f"Y-Left Max ({primary})"
            y1_max = st.number_input(lbl, value=round(xmx+xp,4), format="%.4f", key="y1max")
        with ac3:
            lbl = f"Y-Right Min ({secondary[:10]}...)" if len(secondary)>10 else f"Y-Right Min ({secondary})"
            y2_min = st.number_input(lbl, value=round(ymn-yp,4), format="%.4f", key="y2min")
        with ac4:
            lbl = f"Y-Right Max ({secondary[:10]}...)" if len(secondary)>10 else f"Y-Right Max ({secondary})"
            y2_max = st.number_input(lbl, value=round(ymx+yp,4), format="%.4f", key="y2max")
        with ac5:
            auto_scale = st.checkbox("Auto Scale", value=True, key="autoscale")

        y_range  = None if auto_scale else [y1_min, y1_max]
        y2_range = None if auto_scale else [y2_min, y2_max]

        st.markdown('<div class="section-label" style="margin-top:8px;">TIME SERIES & TREND</div>', unsafe_allow_html=True)

        fig_main, eq_str, r2_tl, tl_ok = make_chart(
            df_f, date_col, primary, [secondary],
            trendline_type, show_equation,
            chart_mode_primary, [chart_mode_secondary],
            y_range=y_range, y2_range=y2_range
        )
        fig_main.update_layout(
            title=dict(text=f"<b>{primary}</b>  vs  <b>{secondary}</b>  — {trendline_type} Trendline",
                       font=dict(family="Syne,sans-serif",size=15,color="#1a2540"), x=0.5, xanchor="center"))
        st.plotly_chart(fig_main, use_container_width=True)

        # Trendline banner
        if tl_ok and eq_str:
            tl_r2_col = "#1a7a4a" if r2_tl >= 0.7 else ("#d97706" if r2_tl >= 0.4 else "#c0392b")
            st.markdown(
                f'<div class="tl-info-bar">'
                f'<div><div style="font-family:DM Mono,monospace;font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Trendline</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#1565c0;">{trendline_type}</div></div>'
                f'<div><div style="font-family:DM Mono,monospace;font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Equation</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:.92rem;font-weight:600;color:#1a2540;">{eq_str}</div></div>'
                f'<div><div style="font-family:DM Mono,monospace;font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Trendline R²</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{tl_r2_col};">{r2_tl:.4f}</div></div>'
                f'</div>', unsafe_allow_html=True)

        # Stats tables
        st.markdown('<div class="section-label" style="margin-top:4px;">VARIABLE STATISTICS & REGRESSION SUMMARY</div>', unsafe_allow_html=True)

        def stat_table(cname, s, hbg, hcol):
            return (f'<div style="background:#fff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
                    f'<div style="background:{hbg};padding:8px 14px;font-family:Syne,sans-serif;font-size:.72rem;font-weight:700;color:{hcol};letter-spacing:.1em;text-transform:uppercase;">{cname}</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:.78rem;">'
                    f'<thead><tr style="background:#f4f7fc;">'
                    f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Min</th>'
                    f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Max</th>'
                    f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Mean</th>'
                    f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Std Dev</th>'
                    f'</tr></thead><tbody><tr>'
                    f'<td style="padding:10px 14px;color:#1a2540;">{s.min():.7g}</td>'
                    f'<td style="padding:10px 14px;color:#1a2540;">{s.max():.7g}</td>'
                    f'<td style="padding:10px 14px;color:#1a2540;">{s.mean():.7g}</td>'
                    f'<td style="padding:10px 14px;color:#1a2540;">{s.std():.7g}</td>'
                    f'</tr></tbody></table></div>')

        cv_col  = "#c0392b" if (corr_val or 0) < 0 else "#1a7a4a"
        sl_col  = "#1a7a4a" if slope >= 0 else "#c0392b"
        r2_col  = "#1a7a4a" if r2_stat >= 0.7 else ("#d97706" if r2_stat >= 0.4 else "#c0392b")

        lc, rc = st.columns([2, 1])
        with lc:
            st.markdown(stat_table(primary, x, "#e8f0fe", "#1565c0"), unsafe_allow_html=True)
            st.markdown(stat_table(secondary, y, "#fce8e6", "#c0392b"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#fff;border:1px solid #d0daea;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
                f'<div style="background:#fffbe6;padding:8px 14px;font-family:Syne,sans-serif;font-size:.72rem;font-weight:700;color:#b45309;letter-spacing:.1em;text-transform:uppercase;">Correlation — {primary} vs {secondary}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-family:DM Mono,monospace;font-size:.78rem;">'
                f'<thead><tr style="background:#fdfaf0;">'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Value</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Direction</th>'
                f'<th style="padding:8px 14px;color:#4a5e80;font-weight:500;text-align:left;border-bottom:1px solid #d0daea;">Strength</th>'
                f'</tr></thead><tbody><tr>'
                f'<td style="padding:10px 14px;color:{cv_col};font-weight:700;">{corr_val:.7f}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{relation}</td>'
                f'<td style="padding:10px 14px;color:#1a2540;">{strength}</td>'
                f'</tr></tbody></table></div>', unsafe_allow_html=True)
        with rc:
            st.markdown(
                f'<div style="background:#fff;border:1px solid #d0daea;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.05);">'
                f'<div style="background:#fffbe6;padding:10px 16px;font-family:Syne,sans-serif;font-size:.72rem;font-weight:700;color:#b45309;letter-spacing:.1em;text-transform:uppercase;">Regression Summary</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #eef1f7;">'
                f'<span style="font-family:DM Mono,monospace;font-size:.75rem;color:#4a5e80;">SLOPE</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{sl_col};">{slope:.3f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid #eef1f7;">'
                f'<span style="font-family:DM Mono,monospace;font-size:.75rem;color:#4a5e80;">R²</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_col};">{r2_stat:.3f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:13px 18px;">'
                f'<span style="font-family:DM Mono,monospace;font-size:.75rem;color:#4a5e80;">ADJ. R²</span>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{r2_col};">{adj_r2:.3f}</span></div>'
                f'</div>', unsafe_allow_html=True)

        tl_txt = (f"Trendline ({trendline_type}): <b>{eq_str}</b> | R² = <b>{r2_tl:.4f}</b>"
                  if tl_ok and eq_str else "Trendline could not be fitted.")
        st.markdown(f"""
        <div class="insight-block">
            <div class="insight-heading">Relationship</div>
            <div class="insight-text"><b>{strength} {relation}</b> relationship (r = <b>{corr_val:.3f}</b>)
            from <b>{start_date}</b> to <b>{end_date}</b>.</div>
            <div class="insight-heading" style="margin-top:16px;">Trendline</div>
            <div class="insight-text">{tl_txt}</div>
            <div class="insight-heading" style="margin-top:16px;">Period Change</div>
            <div class="insight-text">
                <b>{primary}</b>: <b>{px_cs}</b> ({px_s:.4g} → {px_e:.4g}), avg <b>{px_as}</b><br><br>
                <b>{secondary}</b>: <b>{py_cs}</b> ({py_s:.4g} → {py_e:.4g}), avg <b>{py_as}</b>
            </div>
            <div class="insight-heading" style="margin-top:16px;">Regression</div>
            <div class="insight-text">Slope <b>{slope:.4f}</b> — per unit of <b>{primary}</b>,
            <b>{secondary}</b> changes by ~<b>{slope:.4f}</b> units.
            Model explains <b>{r2_stat*100:.1f}%</b> of variance.</div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════
    # TAB 4 — REPORT
    # ══════════════════════════════
    with tab4:
        import json as _rjson

        st.markdown('<div class="section-label">MULTI-SERIES REPORT BUILDER</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Select one primary and multiple secondary variables. Configure each chart individually. Click <b>Generate</b> to preview, then <b>Download HTML Report</b> to save — open in browser and Print → Save as PDF.</div>', unsafe_allow_html=True)

        if len(active_numeric_cols) < 2:
            st.warning("Not enough numeric columns in the active dataset.")
        else:
            _adc4 = st.session_state.get("active_dataset_choice", "RAW")
            _bc4, _bt4 = _badge_map.get(_adc4, ("#5a6e8a","RAW DATA"))
            st.markdown(
                f'<div style="background:{_bc4};color:#fff;font-family:Syne,sans-serif;font-size:.7rem;'
                f'font-weight:700;letter-spacing:.12em;padding:6px 14px;border-radius:6px;display:inline-block;margin-bottom:12px;">'
                f'USING: {_bt4}</div>', unsafe_allow_html=True)

            date_list_r = sorted(df_active[date_col].dt.date.unique())

            rp1, rp2 = st.columns(2)
            with rp1: r_start = st.selectbox("Start Date", date_list_r, key="r_sd")
            with rp2: r_end   = st.selectbox("End Date",   date_list_r, index=len(date_list_r)-1, key="r_ed")

            if r_start > r_end:
                st.error("Start Date cannot be after End Date.")
            else:
                rpp1, rpp2 = st.columns([1, 2])
                with rpp1:
                    r_prim_lbl = st.selectbox("Primary Variable", active_numeric_labeled, key="r_pv")
                    r_primary  = active_label_to_col[r_prim_lbl]
                with rpp2:
                    r_sec_lbls    = st.multiselect("Secondary Variables (one chart each)", active_numeric_labeled, key="r_svs")
                    r_secondaries = [active_label_to_col[l] for l in r_sec_lbls]

                if not r_secondaries:
                    st.info("Select at least one secondary variable to configure charts.")
                else:
                    st.markdown('<div class="section-label" style="margin-top:18px;">PER-CHART CONFIGURATION</div>', unsafe_allow_html=True)

                    chart_configs = []
                    for i, sec in enumerate(r_secondaries):
                        with st.expander(f"📊 Chart {i+1}: {r_primary}  vs  {sec}", expanded=(i == 0)):
                            cc1, cc2, cc3, cc4 = st.columns(4)
                            with cc1: cm_p  = st.selectbox("Primary Chart Type",   list(CHART_MODES.keys()),                           key=f"r_cmp_{i}")
                            with cc2: cm_s  = st.selectbox("Secondary Chart Type", list(CHART_MODES.keys()), index=2,                  key=f"r_cms_{i}")
                            with cc3: tl_t  = st.selectbox("Trendline Type",       ["Exponential","Linear","Logarithmic","Power"],      key=f"r_tlt_{i}")
                            with cc4: sh_eq = st.checkbox("Show Equation",  value=True,  key=f"r_sheq_{i}")

                            sc1, sc2 = st.columns(2)
                            with sc1: r_auto     = st.checkbox("Auto Scale", value=True, key=f"r_auto_{i}")
                            with sc2: chart_title = st.text_input("Chart Title", value=f"{r_primary} vs {sec}", key=f"r_title_{i}")

                            if r_auto:
                                yr1, yr2 = None, None
                            else:
                                df_tmp = df_active[(df_active[date_col].dt.date >= r_start) & (df_active[date_col].dt.date <= r_end)]
                                def _pad(mn, mx): p = (mx-mn)*0.1 or abs(mx)*0.1 or 10; return round(mn-p,4), round(mx+p,4)
                                x1a,x1b = _pad(float(df_tmp[r_primary].min()), float(df_tmp[r_primary].max()))
                                x2a,x2b = _pad(float(df_tmp[sec].min()),       float(df_tmp[sec].max()))
                                a1,a2,a3,a4 = st.columns(4)
                                with a1: yr1a = st.number_input("Y-Left Min",  value=x1a, format="%.4f", key=f"r_y1mn_{i}")
                                with a2: yr1b = st.number_input("Y-Left Max",  value=x1b, format="%.4f", key=f"r_y1mx_{i}")
                                with a3: yr2a = st.number_input("Y-Right Min", value=x2a, format="%.4f", key=f"r_y2mn_{i}")
                                with a4: yr2b = st.number_input("Y-Right Max", value=x2b, format="%.4f", key=f"r_y2mx_{i}")
                                yr1, yr2 = [yr1a, yr1b], [yr2a, yr2b]

                            chart_configs.append({"secondary": sec, "cm_p": cm_p, "cm_s": cm_s,
                                                   "tl_type": tl_t, "show_eq": sh_eq,
                                                   "title": chart_title, "y_range": yr1, "y2_range": yr2})

                    st.markdown("---")
                    generate = st.button(f"🔄  Generate {len(chart_configs)} Chart{'s' if len(chart_configs)!=1 else ''}",
                                         key="gen_report", use_container_width=True)
                    if generate:
                        st.session_state["report_html"] = None

                    if generate or st.session_state.get("report_html"):
                        df_r = df_active[(df_active[date_col].dt.date >= r_start) & (df_active[date_col].dt.date <= r_end)].copy()

                        if len(df_r) < 2:
                            st.warning("Not enough data in selected range.")
                        else:
                            if generate:
                                st.markdown('<div class="section-label" style="margin-top:16px;">GENERATED CHARTS</div>', unsafe_allow_html=True)

                                # ── x dates as ISO strings — NEVER use fig.to_json() ──
                                x_dates_iso = [d.strftime("%Y-%m-%d") for d in df_r[date_col]]

                                report_charts = []   # payload for HTML
                                for i, cfg in enumerate(chart_configs):
                                    sec = cfg["secondary"]

                                    # Build Plotly figure for on-screen preview
                                    fig_r, eq_r, r2_r, ok_r = make_chart(
                                        df_r, date_col, r_primary, [sec],
                                        cfg["tl_type"], cfg["show_eq"],
                                        cfg["cm_p"], [cfg["cm_s"]],
                                        y_range=cfg["y_range"], y2_range=cfg["y2_range"], height=480)
                                    fig_r.update_layout(title=dict(
                                        text=f"<b>{cfg['title']}</b>",
                                        font=dict(family="Syne,sans-serif",size=14,color="#1a2540"),
                                        x=0.5, xanchor="center"))

                                    # On-screen preview
                                    st.markdown(
                                        f'<div style="background:#fff;border:1px solid #d0daea;border-radius:12px;'
                                        f'padding:14px 18px 2px;margin-bottom:4px;box-shadow:0 2px 8px rgba(0,0,0,.06);">'
                                        f'<div style="font-family:Syne,sans-serif;font-size:.72rem;font-weight:700;'
                                        f'color:#1565c0;letter-spacing:.1em;text-transform:uppercase;">Chart {i+1}: {cfg["title"]}</div>'
                                        f'</div>', unsafe_allow_html=True)
                                    st.plotly_chart(fig_r, use_container_width=True, key=f"rc_{i}")

                                    if ok_r and eq_r:
                                        tl_col = "#1a7a4a" if r2_r >= 0.7 else ("#d97706" if r2_r >= 0.4 else "#c0392b")
                                        st.markdown(
                                            f'<div style="background:#f0f5ff;border-left:3px solid #1565c0;border-radius:6px;'
                                            f'padding:8px 16px;display:flex;gap:30px;margin-bottom:18px;flex-wrap:wrap;">'
                                            f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Trendline</span>'
                                            f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#1565c0;">{cfg["tl_type"]}</div></div>'
                                            f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Equation</span>'
                                            f'<div style="font-family:DM Mono,monospace;font-weight:600;color:#1a2540;">{eq_r}</div></div>'
                                            f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">R²</span>'
                                            f'<div style="font-family:Syne,sans-serif;font-weight:700;color:{tl_col};">{r2_r:.4f}</div></div>'
                                            f'</div>', unsafe_allow_html=True)

                                    # ── Build per-chart trace dicts manually from DataFrame ──
                                    mode_p, shape_p, sm_p = CHART_MODES.get(cfg["cm_p"], ("lines","spline",1.3))
                                    mode_s, shape_s, sm_s = CHART_MODES.get(cfg["cm_s"], ("lines","spline",1.3))

                                    prim_y = df_r[r_primary].tolist()
                                    sec_y  = df_r[sec].tolist()

                                    html_traces = [
                                        {"type":"scatter","mode":mode_p,"name":r_primary,
                                         "x":x_dates_iso,"y":prim_y,
                                         "line":{"color":"#1a6fba","width":2.5,"shape":"spline"},
                                         "marker":{"size":5,"color":"#1a6fba"}},
                                        {"type":"scatter","mode":mode_s,"name":sec,
                                         "x":x_dates_iso,"y":sec_y,"yaxis":"y2",
                                         "line":{"color":"#c0392b","width":2.5,"shape":"spline"},
                                         "marker":{"size":5,"color":"#c0392b"}},
                                    ]

                                    # Add trendline trace if computed
                                    if ok_r:
                                        dd, dy, _, _, _ = compute_trendline(df_r[date_col], df_r[r_primary], cfg["tl_type"])
                                        if dd is not None:
                                            tl_x = [d.strftime("%Y-%m-%d") for d in dd]
                                            html_traces.append({
                                                "type":"scatter","mode":"lines","name":"Trendline",
                                                "x":tl_x,"y":dy.tolist(),"showlegend":False,
                                                "line":{"color":"#1a6fba","width":1.8,"dash":"solid","shape":"spline"}
                                            })

                                    eq_badge = f"{eq_r}   R² = {r2_r:.4f}" if ok_r and eq_r else ""
                                    report_charts.append({
                                        "div_id":        f"rc_{i}",
                                        "chart_title":   cfg["title"],
                                        "y_left_title":  r_primary,
                                        "y_right_title": sec,
                                        "eq_badge":      eq_badge,
                                        "traces":        html_traces,
                                        "y_range":       cfg["y_range"],
                                        "y2_range":      cfg["y2_range"],
                                    })

                                # ── Build HTML ──
                                divs_html = ""
                                for c in report_charts:
                                    eq_html = (f'<p style="font-family:monospace;font-size:13px;color:#1565c0;'
                                               f'margin-top:8px;padding:5px 12px;background:#f0f5ff;'
                                               f'border-left:3px solid #1565c0;display:inline-block;border-radius:4px;">'
                                               f'{c["eq_badge"]}</p>') if c["eq_badge"] else ""
                                    divs_html += (
                                        f'<div class="chart-block">'
                                        f'<h3>Chart: {c["chart_title"]}</h3>'
                                        f'<div id="{c["div_id"]}" style="width:100%;height:480px;"></div>'
                                        f'{eq_html}'
                                        f'</div>\n')

                                charts_json = _rjson.dumps(report_charts)

                                html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Trend Analysis Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'DM Mono',monospace;background:#f4f6fa;color:#1a2540;padding:30px;}}
  .rpt-header{{background:linear-gradient(135deg,#1a3a6e,#1565c0);color:#fff;border-radius:12px;padding:28px 40px;margin-bottom:28px;}}
  .rpt-header h1{{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;}}
  .rpt-header p{{font-size:.8rem;color:#90caf9;margin-top:6px;letter-spacing:.1em;text-transform:uppercase;}}
  .rpt-meta{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;}}
  .meta-item{{background:#fff;border:1px solid #d0daea;border-radius:8px;padding:12px 18px;}}
  .meta-label{{font-size:.62rem;color:#5a6e8a;text-transform:uppercase;letter-spacing:.1em;}}
  .meta-value{{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#1a2540;margin-top:3px;}}
  .chart-block{{background:#fff;border:1px solid #d0daea;border-radius:12px;padding:20px 24px;margin-bottom:24px;page-break-inside:avoid;box-shadow:0 2px 8px rgba(0,0,0,.05);}}
  .chart-block h3{{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#1565c0;margin-bottom:12px;letter-spacing:.04em;}}
  footer{{text-align:center;font-size:.7rem;color:#aaa;margin-top:30px;}}
  @media print{{body{{background:#fff;padding:10px;}}.chart-block{{box-shadow:none;border:1px solid #ccc;page-break-inside:avoid;}}}}
</style>
</head>
<body>
<div class="rpt-header">
  <h1>TREND ANALYSIS REPORT</h1>
  <p>Primary: {r_primary} &nbsp;·&nbsp; {r_start} to {r_end} &nbsp;·&nbsp; {len(report_charts)} Charts</p>
</div>
<div class="rpt-meta">
  <div class="meta-item"><div class="meta-label">Primary</div><div class="meta-value">{r_primary}</div></div>
  <div class="meta-item"><div class="meta-label">Period</div><div class="meta-value">{r_start} &rarr; {r_end}</div></div>
  <div class="meta-item"><div class="meta-label">Charts</div><div class="meta-value">{len(report_charts)}</div></div>
  <div class="meta-item"><div class="meta-label">Data Points</div><div class="meta-value">{len(df_r)}</div></div>
</div>
{divs_html}
<footer>Generated by Trend Analysis Dashboard</footer>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function() {{
  var CHARTS = {charts_json};
  function drawAll() {{
    CHARTS.forEach(function(c) {{
      var yRange  = c.y_range  || null;
      var y2Range = c.y2_range || null;
      var layout = {{
        title: {{
          text: '<b>' + c.chart_title + '</b>',
          font: {{family:'Syne,sans-serif', size:15, color:'#1a2540'}},
          x: 0.5, xanchor: 'center'
        }},
        height: 480,
        autosize: true,
        paper_bgcolor: '#ffffff',
        plot_bgcolor:  '#ffffff',
        font: {{family:'DM Mono,monospace', color:'#1a2540', size:11}},
        legend: {{orientation:'h', x:0.5, xanchor:'center', y:-0.2,
                  font:{{size:11}}, bgcolor:'#f8f9fc',
                  bordercolor:'#d0daea', borderwidth:1}},
        margin: {{l:70, r:80, t:55, b:90}},
        xaxis: {{
          type: 'date',
          title: {{text:'Date', font:{{color:'#333',size:12}}}},
          gridcolor: '#e8edf5', griddash: 'dash',
          showline: true, linecolor: '#c5d0e0',
          tickangle: -45, tickfont: {{color:'#2a3a55', size:10}}
        }},
        yaxis: {{
          title: {{text: c.y_left_title, font:{{color:'#1a6fba',size:12}}}},
          tickfont: {{color:'#1a6fba', size:11}},
          gridcolor: '#e8edf5', griddash: 'dash',
          zeroline: true, zerolinecolor: '#c5d0e0',
          showline: true, linecolor: '#c5d0e0'
        }},
        yaxis2: {{
          title: {{text: c.y_right_title, font:{{color:'#c0392b',size:12}}}},
          tickfont: {{color:'#c0392b', size:11}},
          overlaying: 'y', side: 'right',
          showline: true, linecolor: '#c5d0e0',
          zeroline: false
        }}
      }};
      if (yRange)  {{ layout.yaxis.range  = yRange;  }}
      if (y2Range) {{ layout.yaxis2.range = y2Range; }}
      Plotly.newPlot(c.div_id, c.traces, layout, {{
        responsive: true, displayModeBar: true, scrollZoom: false
      }});
    }});
  }}
  if (document.readyState === 'complete') {{ drawAll(); }}
  else {{ window.addEventListener('load', drawAll); }}
}})();
</script>
</body>
</html>"""
                                st.session_state["report_html"] = html_report

                            # Download button
                            if st.session_state.get("report_html"):
                                st.markdown("---")
                                st.download_button(
                                    label="⬇️  Download HTML Report  (open in browser → Print → Save as PDF)",
                                    data=st.session_state["report_html"].encode("utf-8"),
                                    file_name="trend_analysis_report.html",
                                    mime="text/html",
                                    key="dl_report",
                                    use_container_width=True
                                )
                                st.caption("💡 Open in Chrome/Edge → Ctrl+P → Save as PDF")

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;background:#fff;border:1px dashed #c5d0e0;
      border-radius:12px;margin-top:40px;box-shadow:0 1px 6px rgba(0,0,0,.05);">
      <div style="font-size:3rem;margin-bottom:16px;">📂</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#1565c0;font-weight:700;letter-spacing:.06em;">NO DATA LOADED</div>
      <div style="font-size:.78rem;color:#4a5e80;margin-top:10px;line-height:1.7;">
        Upload an Excel (.xlsx) file using the sidebar.<br>
        First column = Date, remaining columns = numeric variables.
      </div>
    </div>""", unsafe_allow_html=True)
