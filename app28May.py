import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, leaves_list
import base64, io, json
import datetime as _dt

# ── Optional libs for Excel / PDF export ──
# Matplotlib (chart rendering for embedding)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    plt = None
    mdates = None
    PdfPages = None

# Excel engines: prefer xlsxwriter, fall back to openpyxl
_EXCEL_ENGINE = None
try:
    import xlsxwriter  # noqa: F401
    _EXCEL_ENGINE = "xlsxwriter"
except Exception:
    try:
        import openpyxl  # noqa: F401
        _EXCEL_ENGINE = "openpyxl"
    except Exception:
        _EXCEL_ENGINE = None

# PIL (used by PDF builder to embed PNGs)
try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
    _PILImage = None

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
.dash-header{background:linear-gradient(135deg,#2d7a4f,#3a9e65 60%,#4caf7d);border:1px solid #2d7a4f;border-radius:12px;padding:28px 40px;margin-bottom:28px;position:relative;overflow:hidden;}
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
[data-testid="stFileUploaderDropzoneInstructions"] small{display:none !important;}
[data-testid="stFileUploaderDropzone"] small{display:none !important;}
[data-testid="stFileUploaderDropzoneInstructions"] span:not([data-testid]){display:none !important;}
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
# HELPER: matplotlib chart rendering (for Excel embedding & PDF export)
# ─────────────────────────────────────────────────────────
def _mpl_chart(df_plot, date_col, primary, secondaries,
               tl_type, show_eq, title, figsize=(11, 5.5)):
    """Render a chart with matplotlib. Returns (fig, eq_str, r2_val, ok).
    Primary on left axis; secondaries on right axis (if any)."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is not installed. Run: pip install matplotlib")
    fig, ax1 = plt.subplots(figsize=figsize, dpi=130)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    x_vals = pd.to_datetime(df_plot[date_col])
    yp     = df_plot[primary]

    # Primary
    ax1.plot(x_vals, yp, color=PRIMARY_COLOR, linewidth=2.2,
             label=primary, marker="o", markersize=3, markerfacecolor=PRIMARY_COLOR)
    ax1.set_ylabel(primary, color=PRIMARY_COLOR, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=PRIMARY_COLOR)
    ax1.grid(True, linestyle="--", linewidth=0.6, color="#e8edf5")

    # Trendline on primary
    dd, dy, eq, r2v, ok = compute_trendline(x_vals, yp, tl_type)
    if ok:
        ax1.plot(dd, dy, color=PRIMARY_COLOR, linewidth=1.5,
                 linestyle="--", alpha=0.85, label=f"{primary} — {tl_type} trend")
        if show_eq and eq:
            ax1.text(0.02, 0.97, f"{eq}\nR² = {r2v:.4f}",
                     transform=ax1.transAxes, fontsize=9,
                     verticalalignment="top", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="#f0f5ff",
                               edgecolor="#1565c0", linewidth=1))

    # Secondaries on shared right axis
    ax2 = None
    if secondaries:
        ax2 = ax1.twinx()
        sec_lines = []
        for i, sec in enumerate(secondaries):
            color = PALETTE[(i + 1) % len(PALETTE)]
            ax2.plot(x_vals, df_plot[sec], color=color, linewidth=2.0,
                     label=sec, marker="s", markersize=2.5, markerfacecolor=color)
            sec_lines.append(sec)
        # Y-right label: first secondary if one, generic if multiple
        if len(secondaries) == 1:
            ax2.set_ylabel(secondaries[0], color=SECONDARY_COLOR, fontsize=11, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor=SECONDARY_COLOR)
        else:
            ax2.set_ylabel("Other Variables", color="#374151", fontsize=11, fontweight="bold")

    # X-axis formatting
    ax1.set_xlabel("Date", fontsize=11)
    locator = mdates.AutoDateLocator()
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18), ncol=min(5, len(h1) + len(h2)),
                   frameon=True, fontsize=8.5)
    else:
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                   ncol=min(5, len(h1)), frameon=True, fontsize=8.5)

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#1a2540")
    fig.tight_layout()
    return fig, eq, r2v, ok


def _fig_to_png_bytes(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()


def _build_excel_report(report_meta, charts_payload, df_data):
    """Build an Excel report (xlsx bytes) with embedded chart PNGs.

    report_meta    : dict with keys 'primary', 'start', 'end', 'mode', 'n_charts'
    charts_payload : list of dicts: {'title','png_bytes','equation','r2','variables'}
    df_data        : DataFrame slice that produced the charts (saved as 'Data' sheet)
    """
    if _EXCEL_ENGINE is None:
        raise RuntimeError(
            "No Excel engine available. Install one of: 'pip install xlsxwriter' "
            "or 'pip install openpyxl'.")

    out = io.BytesIO()

    # Build the summary as a DataFrame
    summary_rows = [
        ["Report Title",     "Trend Analysis Report"],
        ["Primary Variable", report_meta.get("primary", "")],
        ["Period Start",     str(report_meta.get("start", ""))],
        ["Period End",       str(report_meta.get("end", ""))],
        ["Report Mode",      report_meta.get("mode", "")],
        ["Number of Charts", report_meta.get("n_charts", 0)],
        ["Data Points",      len(df_data)],
        ["Generated",        _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Field", "Value"])

    # Prepare a copy of df_data with datetimes formatted as strings
    _df_out = df_data.copy()
    for c in _df_out.columns:
        if pd.api.types.is_datetime64_any_dtype(_df_out[c]):
            _df_out[c] = _df_out[c].dt.strftime("%Y-%m-%d")

    if _EXCEL_ENGINE == "xlsxwriter":
        # ── xlsxwriter path (richer formatting + clean image embedding) ──
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            wb = writer.book

            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            ws_sum = writer.sheets["Summary"]
            hdr_fmt = wb.add_format({"bold": True, "bg_color": "#1565c0",
                                     "font_color": "white", "border": 1})
            ws_sum.set_row(0, None, hdr_fmt)
            ws_sum.set_column("A:A", 22)
            ws_sum.set_column("B:B", 50)

            ws_ch = wb.add_worksheet("Charts")
            ws_ch.set_column("A:A", 110)
            title_fmt = wb.add_format({"bold": True, "font_size": 13,
                                       "font_color": "#1565c0", "align": "left",
                                       "valign": "vcenter"})
            info_fmt = wb.add_format({"italic": True, "font_size": 10,
                                      "font_color": "#4a5e80"})

            cur_row = 0
            for idx, cp in enumerate(charts_payload, start=1):
                ws_ch.set_row(cur_row, 22)
                ws_ch.write(cur_row, 0, f"Chart {idx}: {cp['title']}", title_fmt)
                cur_row += 1
                info_line = ""
                if cp.get("variables"):
                    info_line += "Variables: " + ", ".join(cp["variables"])
                if cp.get("equation"):
                    info_line += f"   |   {cp['equation']}   |   R² = {cp.get('r2', 0):.4f}"
                if info_line:
                    ws_ch.write(cur_row, 0, info_line, info_fmt)
                    cur_row += 1
                if cp.get("png_bytes"):
                    ws_ch.insert_image(cur_row, 0, f"chart_{idx}.png", {
                        "image_data": io.BytesIO(cp["png_bytes"]),
                        "x_scale": 0.62, "y_scale": 0.62,
                        "x_offset": 4, "y_offset": 4,
                    })
                cur_row += 28

            _df_out.to_excel(writer, sheet_name="Data", index=False)
            ws_dat = writer.sheets["Data"]
            ws_dat.set_row(0, None, hdr_fmt)
            ws_dat.set_column(0, max(0, len(_df_out.columns) - 1), 18)

    else:
        # ── openpyxl fallback (simpler but works without xlsxwriter) ──
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.drawing.image import Image as OPImage
        from openpyxl.utils import get_column_letter

        wb = Workbook()

        # Summary
        ws = wb.active
        ws.title = "Summary"
        ws.append(["Field", "Value"])
        for r in summary_rows:
            ws.append(r)
        # Header style
        hdr_font = Font(bold=True, color="FFFFFF")
        hdr_fill = PatternFill("solid", fgColor="1565c0")
        for cell in ws[1]:
            cell.font = hdr_font
            cell.fill = hdr_fill
        ws.column_dimensions["A"].width = 22
        ws.column_dimensions["B"].width = 50

        # Charts sheet
        ws_ch = wb.create_sheet("Charts")
        ws_ch.column_dimensions["A"].width = 110
        title_font = Font(bold=True, size=13, color="1565c0")
        info_font  = Font(italic=True, size=10, color="4a5e80")

        cur_row = 1
        for idx, cp in enumerate(charts_payload, start=1):
            c = ws_ch.cell(row=cur_row, column=1,
                           value=f"Chart {idx}: {cp['title']}")
            c.font = title_font
            cur_row += 1
            info_line = ""
            if cp.get("variables"):
                info_line += "Variables: " + ", ".join(cp["variables"])
            if cp.get("equation"):
                info_line += f"   |   {cp['equation']}   |   R² = {cp.get('r2', 0):.4f}"
            if info_line:
                c2 = ws_ch.cell(row=cur_row, column=1, value=info_line)
                c2.font = info_font
                cur_row += 1
            if cp.get("png_bytes"):
                img = OPImage(io.BytesIO(cp["png_bytes"]))
                # Scale the image so it fits well (openpyxl uses pixels)
                # Original image is ~140 DPI; scale to ~70% for embedding
                img.width = int(img.width * 0.55)
                img.height = int(img.height * 0.55)
                ws_ch.add_image(img, f"A{cur_row}")
            cur_row += 28

        # Data sheet
        ws_d = wb.create_sheet("Data")
        ws_d.append(list(_df_out.columns))
        for cell in ws_d[1]:
            cell.font = hdr_font
            cell.fill = hdr_fill
        for row in _df_out.itertuples(index=False, name=None):
            ws_d.append(list(row))
        for col_idx in range(1, len(_df_out.columns) + 1):
            ws_d.column_dimensions[get_column_letter(col_idx)].width = 18

        wb.save(out)

    out.seek(0)
    return out.getvalue()


def _build_pdf_report(report_meta, charts_payload):
    """Build a multi-page PDF (bytes) using matplotlib PdfPages.
    Each chart is one page. The first page is a cover/summary page."""
    if not _HAS_MPL:
        raise RuntimeError(
            "matplotlib is required for PDF export. Install with: pip install matplotlib")

    out = io.BytesIO()
    with PdfPages(out) as pdf:
        # ── Cover page ──
        fig_c = plt.figure(figsize=(11, 8.5), dpi=130)
        fig_c.patch.set_facecolor("white")
        ax = fig_c.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.78, "TREND ANALYSIS REPORT",
                ha="center", va="center", fontsize=26, fontweight="bold",
                color="#1565c0", family="sans-serif")
        ax.text(0.5, 0.71, "Dynamic Variable Comparison · Trend Analysis",
                ha="center", va="center", fontsize=11, color="#5a6e8a",
                family="sans-serif")
        ax.add_patch(plt.Rectangle((0.15, 0.69), 0.70, 0.002,
                                   color="#1565c0", transform=ax.transAxes))
        meta_lines = [
            ("Primary Variable", report_meta.get("primary", "")),
            ("Period Start",     str(report_meta.get("start", ""))),
            ("Period End",       str(report_meta.get("end", ""))),
            ("Report Mode",      report_meta.get("mode", "")),
            ("Number of Charts", str(report_meta.get("n_charts", 0))),
            ("Data Points",      str(report_meta.get("data_points", 0))),
            ("Generated",        _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for i, (k, v) in enumerate(meta_lines):
            y = 0.58 - i * 0.05
            ax.text(0.22, y, k, fontsize=11, fontweight="bold", color="#1a2540")
            ax.text(0.50, y, str(v), fontsize=11, color="#1a2540", family="monospace")
        ax.text(0.5, 0.06, "Generated by Trend Analysis Dashboard",
                ha="center", fontsize=9, color="#aaa", style="italic")
        pdf.savefig(fig_c)
        plt.close(fig_c)

        # ── Chart pages — render PNG onto a fresh page ──
        for cp in charts_payload:
            img = None
            if cp.get("png_bytes") and _HAS_PIL:
                try:
                    img = _PILImage.open(io.BytesIO(cp["png_bytes"]))
                except Exception:
                    img = None
            fig_p = plt.figure(figsize=(11, 8.5), dpi=130)
            fig_p.patch.set_facecolor("white")
            # Title bar
            ax_t = fig_p.add_axes([0.05, 0.92, 0.90, 0.05])
            ax_t.axis("off")
            ax_t.text(0, 0.5, cp["title"], fontsize=15, fontweight="bold",
                      color="#1565c0", va="center")
            if cp.get("equation"):
                ax_t.text(1, 0.5, f"{cp['equation']}   |   R² = {cp.get('r2',0):.4f}",
                          fontsize=10, color="#1a2540", ha="right", va="center",
                          family="monospace")
            # Image area
            if img is not None:
                ax_i = fig_p.add_axes([0.05, 0.10, 0.90, 0.78])
                ax_i.axis("off")
                ax_i.imshow(img)
            else:
                ax_i = fig_p.add_axes([0.05, 0.10, 0.90, 0.78])
                ax_i.axis("off")
                ax_i.text(0.5, 0.5, "(chart image unavailable)",
                          ha="center", va="center", fontsize=12, color="#999")
            # Variables footer
            if cp.get("variables"):
                ax_f = fig_p.add_axes([0.05, 0.04, 0.90, 0.04])
                ax_f.axis("off")
                ax_f.text(0, 0.5, "Variables: " + ", ".join(cp["variables"]),
                          fontsize=9, color="#4a5e80", style="italic", va="center")
            pdf.savefig(fig_p)
            plt.close(fig_p)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "Trend Analysis Report"
        d["Author"]  = "Trend Analysis Dashboard"
        d["Subject"] = "Trend analysis charts"
        d["CreationDate"] = _dt.datetime.now()

    out.seek(0)
    return out.getvalue()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if file:
    # ─────────────────────────────────────────────────────────
    # HEADER / DATA ROW SELECTION  (shown before main processing)
    # ─────────────────────────────────────────────────────────
    # Read entire sheet WITHOUT assuming where the header is
    try:
        df_no_header = pd.read_excel(file, header=None)
    except Exception as _e:
        st.error(f"Could not read the Excel file: {_e}")
        st.stop()

    # Reset header-selection state on new file
    _file_sig = file.name + str(file.size)
    if st.session_state.get("_header_file_sig") != _file_sig:
        st.session_state["_header_file_sig"]   = _file_sig
        st.session_state["_header_row_pick"]   = 0   # 0-based row index → row 1
        st.session_state["_data_start_pick"]   = 1   # 0-based row index → row 2
        st.session_state["_header_confirmed"]  = False
        # Also reset downstream state
        st.session_state.update({
            "_loaded_file_id":       None,
            "df2_filtered":          None,
            "df3_filtered_out":      None,
            "df4_clean":             None,
            "preprocess_applied":    False,
            "outlier_report":        {},
            "run_preprocess":        False,
            "active_dataset_choice": "DF1",
            "report_html":           None,
        })

    if not st.session_state.get("_header_confirmed", False):
        st.markdown('<div class="section-label">📑 SELECT HEADER & DATA ROWS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            'Excel files sometimes contain title rows, metadata, or blank rows above the actual column headers. '
            'Use the preview below to pick:<br>'
            '&nbsp;&nbsp;• <b>Header Row</b> — the row that contains the column names<br>'
            '&nbsp;&nbsp;• <b>Data Start Row</b> — the first row that contains real data values<br>'
            'Row numbers shown below match Excel\'s 1-based numbering (Row&nbsp;1 is the very first row of the sheet).'
            '</div>',
            unsafe_allow_html=True)

        # Show preview with Excel-style 1-based row numbers
        n_preview = min(20, len(df_no_header))
        preview = df_no_header.head(n_preview).copy()
        preview.insert(0, "Excel Row #", [f"Row {i+1}" for i in range(n_preview)])
        # Rename anonymous columns A, B, C, ... like Excel does
        from string import ascii_uppercase
        def _excel_col(idx):
            # 0 -> A, 25 -> Z, 26 -> AA …
            s = ""
            n = idx
            while True:
                s = ascii_uppercase[n % 26] + s
                n = n // 26 - 1
                if n < 0:
                    break
            return s
        rename_map = {col: _excel_col(i) for i, col in enumerate(df_no_header.columns)}
        preview = preview.rename(columns=rename_map)
        st.markdown(f'<div class="section-label" style="margin-top:6px;">FILE PREVIEW  (first {n_preview} rows)</div>', unsafe_allow_html=True)
        st.dataframe(preview, use_container_width=True, hide_index=True)

        # Row pickers (1-based for the user)
        max_row = len(df_no_header)
        hr_col, dr_col, btn_col = st.columns([1, 1, 1])
        with hr_col:
            header_row_1b = st.number_input(
                "Header Row (1-based)",
                min_value=1, max_value=max_row,
                value=int(st.session_state["_header_row_pick"]) + 1,
                step=1, key="hdr_row_input",
                help="Excel row that holds the column names")
        with dr_col:
            data_start_1b = st.number_input(
                "Data Start Row (1-based)",
                min_value=1, max_value=max_row,
                value=max(int(st.session_state["_data_start_pick"]) + 1, header_row_1b + 1),
                step=1, key="data_row_input",
                help="First Excel row that contains real data")
        with btn_col:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            confirm_hdr = st.button("✅  Confirm & Load Data", use_container_width=True, key="btn_confirm_hdr")

        # Validation
        if data_start_1b <= header_row_1b:
            st.warning("⚠️  Data Start Row must be **after** the Header Row.")

        # Show what the chosen header looks like
        if 1 <= header_row_1b <= max_row:
            chosen_header = df_no_header.iloc[header_row_1b - 1].astype(str).tolist()
            st.markdown(
                f'<div style="background:#f0f5ff;border-left:3px solid #1565c0;border-radius:6px;'
                f'padding:10px 16px;font-size:.78rem;color:#2a3a55;margin-top:6px;line-height:1.6;">'
                f'<b>Selected column names (Row {header_row_1b}):</b> '
                f'{", ".join(repr(c) for c in chosen_header[:10])}'
                + (" ..." if len(chosen_header) > 10 else "") +
                f'</div>',
                unsafe_allow_html=True)

        if confirm_hdr and data_start_1b > header_row_1b:
            st.session_state["_header_row_pick"]  = int(header_row_1b) - 1
            st.session_state["_data_start_pick"]  = int(data_start_1b) - 1
            st.session_state["_header_confirmed"] = True
            st.rerun()

        st.stop()   # nothing else renders until the user confirms

    # ── Build df_raw using the chosen header / data-start rows ──
    _hdr_idx   = int(st.session_state["_header_row_pick"])     # 0-based
    _data_idx  = int(st.session_state["_data_start_pick"])     # 0-based
    skip_rows  = _data_idx - _hdr_idx - 1                       # rows BETWEEN header and data

    try:
        df_raw = pd.read_excel(
            file,
            header=_hdr_idx,
            skiprows=list(range(_hdr_idx + 1, _data_idx)) if skip_rows > 0 else None,
        ).dropna(how="all")
    except Exception as _e:
        st.error(f"Could not parse file with header row = {_hdr_idx+1}, data start = {_data_idx+1}: {_e}")
        if st.button("🔄 Re-select header / data rows"):
            st.session_state["_header_confirmed"] = False
            st.rerun()
        st.stop()

    # Allow user to go back and change the selection
    _spacer_col, _ch_col = st.columns([5, 1])
    with _ch_col:
        if st.button("🔄  Change Header Row", key="btn_change_hdr"):
            st.session_state["_header_confirmed"] = False
            st.rerun()

    df = df_raw.copy()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── Reset state on new file (data-level reset, keyed off file+header choice) ──
    file_id = file.name + str(file.size) + f"_h{_hdr_idx}_d{_data_idx}"
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
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️  Raw Data", "⚙️  Preprocess", "📊  Correlation", "📈  Analysis", "📑  Report", "📋  Multi-Chart Report"
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

        
        date_list = sorted(df_active[date_col].dt.date.unique())
        _min_d, _max_d = date_list[0], date_list[-1]

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            start_date = st.date_input(
                "Start Date",
                value=_min_d, min_value=_min_d, max_value=_max_d,
                key="sd", format="YYYY-MM-DD")
        with r2:
            end_date = st.date_input(
                "End Date",
                value=_max_d, min_value=_min_d, max_value=_max_d,
                key="ed", format="YYYY-MM-DD")
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
        st.markdown('<div class="section-label" style="margin-top:4px;">VARIABLE STATISTICS</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-label">MULTI-SERIES REPORT BUILDER</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            'Choose a <b>Report Mode</b>, pick your variables, configure the date range, then click '
            '<b>Generate Report</b>. All charts will render below, followed by <b>Excel</b> and <b>PDF</b> download buttons.'
            '</div>',
            unsafe_allow_html=True)

        # ── Library availability check ──
        _missing_libs = []
        if not _HAS_MPL:        _missing_libs.append("matplotlib")
        if _EXCEL_ENGINE is None: _missing_libs.append("xlsxwriter or openpyxl")
        if not _HAS_PIL:        _missing_libs.append("pillow")
        if _missing_libs:
            st.warning(
                "⚠️  Some optional libraries needed for Excel / PDF export are missing: "
                + ", ".join(f"`{x}`" for x in _missing_libs)
                + ".  Install them with: `pip install matplotlib xlsxwriter pillow`"
            )

        if len(active_numeric_cols) < 1:
            st.warning("No numeric columns in the active dataset.")
        else:
            _adc4 = st.session_state.get("active_dataset_choice", "RAW")
            _bc4, _bt4 = _badge_map.get(_adc4, ("#5a6e8a", "RAW DATA"))
            st.markdown(
                f'<div style="background:{_bc4};color:#fff;font-family:Syne,sans-serif;font-size:.7rem;'
                f'font-weight:700;letter-spacing:.12em;padding:6px 14px;border-radius:6px;display:inline-block;margin-bottom:12px;">'
                f'USING: {_bt4}</div>', unsafe_allow_html=True)

            # ── Date selection (calendar) ──
            date_list_r = sorted(df_active[date_col].dt.date.unique())
            if not date_list_r:
                st.warning("No valid dates in dataset.")
                st.stop()
            _r_min, _r_max = date_list_r[0], date_list_r[-1]
            rp1, rp2 = st.columns(2)
            with rp1:
                r_start = st.date_input("Start Date", value=_r_min,
                                        min_value=_r_min, max_value=_r_max,
                                        key="r_sd", format="YYYY-MM-DD")
            with rp2:
                r_end = st.date_input("End Date", value=_r_max,
                                      min_value=_r_min, max_value=_r_max,
                                      key="r_ed", format="YYYY-MM-DD")

            if r_start > r_end:
                st.error("Start Date cannot be after End Date.")
                st.stop()

            # ── Report mode selection ──
            st.markdown('<div class="section-label" style="margin-top:14px;">REPORT MODE</div>', unsafe_allow_html=True)
            REPORT_MODES = [
                "1. Primary Trends Only  — pick one or more variables; each gets its own primary-only trend chart",
                "2. Primary + Secondary (Combined Trend)  — pick a primary, a secondary, and optional 'other' variables → combined chart",
            ]
            report_mode = st.radio("Mode", REPORT_MODES, index=0,
                                   key="r_mode", label_visibility="collapsed")
            is_mode_1 = report_mode == REPORT_MODES[0]
            is_mode_2 = report_mode == REPORT_MODES[1]

            # ── Variable selection (depends on mode) ──
            st.markdown('<div class="section-label" style="margin-top:10px;">VARIABLE SELECTION</div>', unsafe_allow_html=True)

            chart_specs = []   # list of dicts describing each chart to build

            if is_mode_1:
                # Mode 1: pick any number of variables — one trend per variable, primary axis only
                rp_a, rp_b = st.columns([3, 1])
                with rp_a:
                    sel_lbls = st.multiselect(
                        "Variables to Plot (each one becomes its own primary-only trend chart)",
                        active_numeric_labeled,
                        default=active_numeric_labeled[:1],
                        key="r_m1_vars")
                    sel_vars = [active_label_to_col[l] for l in sel_lbls]
                with rp_b:
                    m1_tl = st.selectbox("Trendline Type",
                                         ["Linear", "Exponential", "Logarithmic", "Power"],
                                         key="r_m1_tl")

                m1_show_eq = st.checkbox("Show Equation & R² on each chart",
                                         value=True, key="r_m1_sheq")

                if not sel_vars:
                    st.info("Select at least one variable to plot.")
                else:
                    for v in sel_vars:
                        chart_specs.append({
                            "title":     f"{v} — Trend",
                            "primary":   v,
                            "secondaries": [],
                            "tl_type":   m1_tl,
                            "show_eq":   m1_show_eq,
                            "variables": [v],
                        })

                # Primary variable for the report header (use first selected)
                r_primary_hdr = sel_vars[0] if sel_vars else "—"

            else:
                # Mode 2: one combined chart with primary + secondary + others
                rp_p, rp_s = st.columns(2)
                with rp_p:
                    m2_prim_lbl = st.selectbox("Primary Variable", active_numeric_labeled, key="r_m2_pv")
                    m2_primary  = active_label_to_col[m2_prim_lbl]
                with rp_s:
                    _sec_options = [l for l in active_numeric_labeled if active_label_to_col[l] != m2_primary]
                    if _sec_options:
                        m2_sec_lbl = st.selectbox("Secondary Variable", _sec_options, key="r_m2_sv")
                        m2_secondary = active_label_to_col[m2_sec_lbl]
                    else:
                        m2_secondary = None
                        st.warning("Need at least two numeric variables for Mode 2.")

                # Other variables
                _other_options = [l for l in active_numeric_labeled
                                  if active_label_to_col[l] not in (m2_primary, m2_secondary)]
                m2_other_lbls = st.multiselect(
                    "Other Variables (optional — added to the same combined chart)",
                    _other_options, default=[], key="r_m2_others")
                m2_others = [active_label_to_col[l] for l in m2_other_lbls]

                rp_t, rp_e = st.columns([3, 1])
                with rp_t:
                    m2_tl = st.selectbox("Trendline Type (on Primary)",
                                         ["Linear", "Exponential", "Logarithmic", "Power"],
                                         key="r_m2_tl")
                with rp_e:
                    m2_show_eq = st.checkbox("Show Equation & R²", value=True, key="r_m2_sheq")

                m2_title = st.text_input(
                    "Chart Title",
                    value=f"{m2_primary} vs {m2_secondary}" + (
                        f" + {len(m2_others)} other" if m2_others else ""),
                    key="r_m2_title")

                if m2_secondary is not None:
                    chart_specs.append({
                        "title":       m2_title,
                        "primary":     m2_primary,
                        "secondaries": [m2_secondary] + m2_others,
                        "tl_type":     m2_tl,
                        "show_eq":     m2_show_eq,
                        "variables":   [m2_primary, m2_secondary] + m2_others,
                    })

                r_primary_hdr = m2_primary if m2_secondary is not None else "—"

            # ── Generate button ──
            st.markdown("---")
            generate = st.button(
                f"🔄  Generate Report  ({len(chart_specs)} chart{'s' if len(chart_specs) != 1 else ''})",
                key="gen_report", use_container_width=True,
                disabled=(len(chart_specs) == 0))

            if generate:
                # Wipe any previous run so we never mix old + new bytes
                for _k in ("report_charts", "report_xlsx", "report_pdf",
                           "report_meta_str", "report_specs_cache"):
                    st.session_state.pop(_k, None)

                # Filter dataset to selected period
                df_r = df_active[
                    (df_active[date_col].dt.date >= r_start) &
                    (df_active[date_col].dt.date <= r_end)
                ].copy()
                df_r[date_col] = pd.to_datetime(df_r[date_col])

                if len(df_r) < 2:
                    st.warning("Not enough data points in the selected period.")
                else:
                    with st.spinner("Building charts…"):
                        charts_payload = []         # for Excel/PDF builders
                        preview_dl_cols = set()     # columns needed for the Data sheet

                        for i, spec in enumerate(chart_specs):
                            prim = spec["primary"]
                            secs = spec["secondaries"]
                            preview_dl_cols.add(prim)
                            for s in secs:
                                preview_dl_cols.add(s)

                            # Build matplotlib version (used both for screen preview AND embedding)
                            try:
                                fig_mpl, eq_mpl, r2_mpl, ok_mpl = _mpl_chart(
                                    df_r, date_col, prim, secs,
                                    spec["tl_type"], spec["show_eq"], spec["title"])
                                png_bytes = _fig_to_png_bytes(fig_mpl)
                                plt.close(fig_mpl)
                            except Exception as _ex:
                                st.warning(f"Could not build embed image for chart '{spec['title']}': {_ex}")
                                png_bytes = b""
                                eq_mpl, r2_mpl, ok_mpl = "", 0.0, False

                            charts_payload.append({
                                "title":       spec["title"],
                                "png_bytes":   png_bytes,
                                "equation":    eq_mpl if spec["show_eq"] else "",
                                "r2":          r2_mpl if r2_mpl is not None else 0.0,
                                "variables":   spec["variables"],
                                "tl_type":     spec["tl_type"],
                                "ok":          ok_mpl,
                                "primary":     prim,
                                "secondaries": secs,
                            })

                        # ── Build the data slice used for the Data sheet ──
                        ordered_cols = [date_col] + [c for c in df_r.columns
                                                     if c in preview_dl_cols and c != date_col]
                        df_data_for_excel = df_r[ordered_cols].copy()

                        report_meta = {
                            "primary":     r_primary_hdr,
                            "start":       r_start,
                            "end":         r_end,
                            "mode":        "Primary Trends Only" if is_mode_1 else "Primary + Secondary (Combined)",
                            "n_charts":    len(charts_payload),
                            "data_points": len(df_r),
                        }

                    # ── Build Excel ──
                    with st.spinner("Building Excel…"):
                        try:
                            xlsx_bytes = _build_excel_report(report_meta, charts_payload, df_data_for_excel)
                            st.session_state["report_xlsx"] = xlsx_bytes
                        except Exception as _ex:
                            st.error(f"Excel build failed: {_ex}")
                            st.session_state["report_xlsx"] = None

                    # ── Build PDF ──
                    with st.spinner("Building PDF…"):
                        try:
                            pdf_bytes = _build_pdf_report(report_meta, charts_payload)
                            st.session_state["report_pdf"] = pdf_bytes
                        except Exception as _ex:
                            st.error(f"PDF build failed: {_ex}")
                            st.session_state["report_pdf"] = None

                    # Cache everything so charts + downloads stay visible after rerun
                    st.session_state["report_charts"] = charts_payload
                    st.session_state["report_meta_str"] = (
                        f"{report_meta['mode']} · {report_meta['n_charts']} chart(s) · "
                        f"{report_meta['start']} → {report_meta['end']} · {report_meta['data_points']} data points")

            # ─────────────────────────────────────────────────────────────
            # RENDER CHARTS + DOWNLOADS  (from session-state cache)
            # Stays visible across reruns until Generate is clicked again
            # ─────────────────────────────────────────────────────────────
            _cached_charts = st.session_state.get("report_charts")
            if _cached_charts:
                st.markdown('<div class="section-label" style="margin-top:14px;">GENERATED CHARTS</div>',
                            unsafe_allow_html=True)

                for i, cp in enumerate(_cached_charts):
                    st.markdown(
                        f'<div style="background:#fff;border:1px solid #d0daea;border-radius:12px;'
                        f'padding:14px 18px 4px;margin-bottom:4px;box-shadow:0 2px 8px rgba(0,0,0,.06);">'
                        f'<div style="font-family:Syne,sans-serif;font-size:.72rem;font-weight:700;'
                        f'color:#1565c0;letter-spacing:.1em;text-transform:uppercase;">'
                        f'Chart {i+1}: {cp["title"]}</div></div>',
                        unsafe_allow_html=True)

                    if cp.get("png_bytes"):
                        st.image(cp["png_bytes"], use_container_width=True)
                    else:
                        st.warning(f"No image available for chart: {cp['title']}")

                    if cp.get("ok") and cp.get("equation"):
                        _r2 = cp.get("r2", 0.0)
                        tl_col = "#1a7a4a" if _r2 >= 0.7 else ("#d97706" if _r2 >= 0.4 else "#c0392b")
                        st.markdown(
                            f'<div style="background:#f0f5ff;border-left:3px solid #1565c0;border-radius:6px;'
                            f'padding:8px 16px;display:flex;gap:30px;margin:6px 0 18px;flex-wrap:wrap;">'
                            f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Trendline</span>'
                            f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#1565c0;">{cp["tl_type"]}</div></div>'
                            f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Equation</span>'
                            f'<div style="font-family:DM Mono,monospace;font-weight:600;color:#1a2540;">{cp["equation"]}</div></div>'
                            f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">R²</span>'
                            f'<div style="font-family:Syne,sans-serif;font-weight:700;color:{tl_col};">{_r2:.4f}</div></div>'
                            f'</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="margin-bottom:18px;"></div>', unsafe_allow_html=True)

                # ── Download buttons (Excel | PDF side by side) ──
                xlsx_ready = bool(st.session_state.get("report_xlsx"))
                pdf_ready  = bool(st.session_state.get("report_pdf"))

                if xlsx_ready or pdf_ready:
                    st.markdown("---")
                    st.markdown('<div class="section-label">📥 DOWNLOAD REPORT</div>', unsafe_allow_html=True)
                    if st.session_state.get("report_meta_str"):
                        st.markdown(
                            f'<div style="background:#f0f5ff;border-left:3px solid #1565c0;border-radius:6px;'
                            f'padding:10px 16px;font-size:.78rem;color:#2a3a55;margin-bottom:12px;">'
                            f'{st.session_state["report_meta_str"]}</div>', unsafe_allow_html=True)

                    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dlc1, dlc2 = st.columns(2)
                    with dlc1:
                        if xlsx_ready:
                            st.download_button(
                                label="⬇️  Download Excel (.xlsx)",
                                data=st.session_state["report_xlsx"],
                                file_name=f"trend_report_{stamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_xlsx", use_container_width=True)
                    with dlc2:
                        if pdf_ready:
                            st.download_button(
                                label="⬇️  Download PDF (.pdf)",
                                data=st.session_state["report_pdf"],
                                file_name=f"trend_report_{stamp}.pdf",
                                mime="application/pdf",
                                key="dl_pdf", use_container_width=True)

    # ══════════════════════════════
    # TAB 5 — MULTI-CHART REPORT
    # ══════════════════════════════
    with tab5:
        st.markdown('<div class="section-label">MULTI-CHART REPORT BUILDER</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Select one primary variable and multiple secondary variables. Configure each chart individually. Click <b>Generate Report Charts</b> — all charts will render below, followed by <b>Excel</b> and <b>PDF</b> download buttons.</div>', unsafe_allow_html=True)

        if len(active_numeric_cols) < 2:
            st.warning("Not enough numeric columns in active dataset.")
        else:
            _adc5 = st.session_state.get("active_dataset_choice", "DF1")
            _bc5, _bt5 = _badge_map.get(_adc5, ("#5a6e8a", "DF1 — RAW DATA"))
            st.markdown(
                f'<div style="background:{_bc5};color:#fff;font-family:Syne,sans-serif;font-size:.7rem;'
                f'font-weight:700;letter-spacing:.12em;padding:6px 14px;border-radius:6px;display:inline-block;margin-bottom:12px;">'
                f'USING: {_bt5}</div>', unsafe_allow_html=True)

            date_list_r5 = sorted(df_active[date_col].dt.date.unique())
            if not date_list_r5:
                st.warning("No valid dates in dataset.")
            else:
                _r5_min, _r5_max = date_list_r5[0], date_list_r5[-1]

                rp1, rp2 = st.columns(2)
                with rp1:
                    r_start = st.date_input(
                        "Start Date",
                        value=_r5_min, min_value=_r5_min, max_value=_r5_max,
                        key="t5_r_sd", format="YYYY-MM-DD")
                with rp2:
                    r_end = st.date_input(
                        "End Date",
                        value=_r5_max, min_value=_r5_min, max_value=_r5_max,
                        key="t5_r_ed", format="YYYY-MM-DD")

                if r_start > r_end:
                    st.error("Start Date cannot be after End Date.")
                else:
                    rpp1, rpp2 = st.columns([1, 2])
                    with rpp1:
                        r_prim_lbl = st.selectbox("Primary Variable (X-axis reference)", active_numeric_labeled, key="t5_r_pv")
                        r_primary  = active_label_to_col[r_prim_lbl]
                    with rpp2:
                        r_sec_lbls = st.multiselect("Secondary Variables (one chart each)", active_numeric_labeled, key="t5_r_svs")
                        r_secondaries = [active_label_to_col[l] for l in r_sec_lbls]

                    if not r_secondaries:
                        st.info("Select at least one secondary variable above to configure charts.")
                    else:
                        st.markdown('<div class="section-label" style="margin-top:18px;">PER-CHART CONFIGURATION</div>', unsafe_allow_html=True)

                        chart_configs = []
                        for i, sec in enumerate(r_secondaries):
                            with st.expander(f"📊 Chart {i+1}: {r_primary}  vs  {sec}", expanded=(i == 0)):
                                cc1, cc2, cc3, cc4 = st.columns(4)
                                with cc1:
                                    cm_p = st.selectbox("Primary Chart Type", list(CHART_MODES.keys()), key=f"t5_r_cmp_{i}")
                                with cc2:
                                    cm_s = st.selectbox("Secondary Chart Type", list(CHART_MODES.keys()), index=2, key=f"t5_r_cms_{i}")
                                with cc3:
                                    tl_t = st.selectbox("Trendline Type", ["Exponential", "Linear", "Logarithmic", "Power"], key=f"t5_r_tlt_{i}")
                                with cc4:
                                    sh_eq = st.checkbox("Show Equation", value=True, key=f"t5_r_sheq_{i}")

                                sc1, sc2 = st.columns(2)
                                with sc1:
                                    r_auto = st.checkbox("Auto Scale", value=True, key=f"t5_r_auto_{i}")
                                with sc2:
                                    chart_title = st.text_input("Chart Title (optional)", value=f"{r_primary} vs {sec}", key=f"t5_r_title_{i}")

                                if r_auto:
                                    yr1, yr2 = None, None
                                else:
                                    ac1a, ac1b, ac2a, ac2b = st.columns(4)
                                    df_tmp = df_active[(df_active[date_col].dt.date >= r_start) & (df_active[date_col].dt.date <= r_end)]
                                    xmn2 = float(df_tmp[r_primary].min()) if not df_tmp[r_primary].isna().all() else 0.0
                                    xmx2 = float(df_tmp[r_primary].max()) if not df_tmp[r_primary].isna().all() else 100.0
                                    ymn2 = float(df_tmp[sec].min()) if not df_tmp[sec].isna().all() else 0.0
                                    ymx2 = float(df_tmp[sec].max()) if not df_tmp[sec].isna().all() else 100.0
                                    xp2  = (xmx2 - xmn2) * 0.1 or 10
                                    yp2  = (ymx2 - ymn2) * 0.1 or 10
                                    with ac1a: yr1a = st.number_input("Y-Left Min",  value=round(xmn2 - xp2, 4), format="%.4f", key=f"t5_r_y1min_{i}")
                                    with ac1b: yr1b = st.number_input("Y-Left Max",  value=round(xmx2 + xp2, 4), format="%.4f", key=f"t5_r_y1max_{i}")
                                    with ac2a: yr2a = st.number_input("Y-Right Min", value=round(ymn2 - yp2, 4), format="%.4f", key=f"t5_r_y2min_{i}")
                                    with ac2b: yr2b = st.number_input("Y-Right Max", value=round(ymx2 + yp2, 4), format="%.4f", key=f"t5_r_y2max_{i}")
                                    yr1, yr2 = [yr1a, yr1b], [yr2a, yr2b]

                                chart_configs.append({
                                    "secondary": sec, "cm_p": cm_p, "cm_s": cm_s,
                                    "tl_type": tl_t, "show_eq": sh_eq,
                                    "title": chart_title, "y_range": yr1, "y2_range": yr2
                                })

                        st.markdown("---")
                        generate5 = st.button(
                            f"🔄  Generate Report Charts  ({len(r_secondaries)} chart{'s' if len(r_secondaries) != 1 else ''})",
                            key="t5_gen_report", use_container_width=True)

                        if generate5:
                            # Wipe any previous run
                            for _k in ("t5_report_charts", "t5_report_xlsx",
                                       "t5_report_pdf", "t5_meta_str"):
                                st.session_state.pop(_k, None)

                            df_r = df_active[(df_active[date_col].dt.date >= r_start) & (df_active[date_col].dt.date <= r_end)].copy()
                            df_r[date_col] = pd.to_datetime(df_r[date_col])

                            if len(df_r) < 2:
                                st.warning("Not enough data in selected range.")
                            else:
                                with st.spinner("Building charts…"):
                                    charts_payload5 = []
                                    used_cols5 = {r_primary}

                                    for i, cfg in enumerate(chart_configs):
                                        sec = cfg["secondary"]
                                        used_cols5.add(sec)

                                        # Build matplotlib PNG (used for both screen + Excel/PDF)
                                        try:
                                            fig_mpl, eq_mpl, r2_mpl, ok_mpl = _mpl_chart(
                                                df_r, date_col, r_primary, [sec],
                                                cfg["tl_type"], cfg["show_eq"], cfg["title"])
                                            png_bytes = _fig_to_png_bytes(fig_mpl)
                                            plt.close(fig_mpl)
                                        except Exception as _ex:
                                            st.warning(f"Could not build image for chart '{cfg['title']}': {_ex}")
                                            png_bytes = b""
                                            eq_mpl, r2_mpl, ok_mpl = "", 0.0, False

                                        charts_payload5.append({
                                            "title":     cfg["title"],
                                            "png_bytes": png_bytes,
                                            "equation":  eq_mpl if cfg["show_eq"] else "",
                                            "r2":        r2_mpl if r2_mpl is not None else 0.0,
                                            "variables": [r_primary, sec],
                                            "tl_type":   cfg["tl_type"],
                                            "ok":        ok_mpl,
                                        })

                                    # Data slice for the Data sheet
                                    ordered_cols5 = [date_col] + [c for c in df_r.columns
                                                                  if c in used_cols5 and c != date_col]
                                    df_data5 = df_r[ordered_cols5].copy()

                                    report_meta5 = {
                                        "primary":     r_primary,
                                        "start":       r_start,
                                        "end":         r_end,
                                        "mode":        "Multi-Chart (Primary vs each Secondary)",
                                        "n_charts":    len(charts_payload5),
                                        "data_points": len(df_r),
                                    }

                                # Build Excel
                                with st.spinner("Building Excel…"):
                                    try:
                                        st.session_state["t5_report_xlsx"] = _build_excel_report(report_meta5, charts_payload5, df_data5)
                                    except Exception as _ex:
                                        st.error(f"Excel build failed: {_ex}")
                                        st.session_state["t5_report_xlsx"] = None

                                # Build PDF
                                with st.spinner("Building PDF…"):
                                    try:
                                        st.session_state["t5_report_pdf"] = _build_pdf_report(report_meta5, charts_payload5)
                                    except Exception as _ex:
                                        st.error(f"PDF build failed: {_ex}")
                                        st.session_state["t5_report_pdf"] = None

                                # Cache rendered charts so they persist across reruns
                                st.session_state["t5_report_charts"] = charts_payload5
                                st.session_state["t5_meta_str"] = (
                                    f"{report_meta5['mode']} · {report_meta5['n_charts']} chart(s) · "
                                    f"{report_meta5['start']} → {report_meta5['end']} · {report_meta5['data_points']} data points")

                        # ─────────────────────────────────────────────────
                        # RENDER CHARTS + DOWNLOADS from session-state cache
                        # ─────────────────────────────────────────────────
                        _cached5 = st.session_state.get("t5_report_charts")
                        if _cached5:
                            st.markdown('<div class="section-label" style="margin-top:8px;">GENERATED CHARTS</div>', unsafe_allow_html=True)

                            for i, cp in enumerate(_cached5):
                                st.markdown(
                                    f'<div style="background:#fff;border:1px solid #d0daea;border-radius:12px;padding:16px 20px 4px;margin-bottom:4px;box-shadow:0 2px 8px rgba(0,0,0,.06);">'
                                    f'<div style="font-family:Syne,sans-serif;font-size:.72rem;font-weight:700;color:#1565c0;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">Chart {i+1}: {cp["title"]}</div></div>',
                                    unsafe_allow_html=True)

                                if cp.get("png_bytes"):
                                    st.image(cp["png_bytes"], use_container_width=True)
                                else:
                                    st.warning(f"No image available for chart: {cp['title']}")

                                if cp.get("ok") and cp.get("equation"):
                                    _r2 = cp.get("r2", 0.0)
                                    tl_r2_col2 = "#1a7a4a" if _r2 >= 0.7 else ("#d97706" if _r2 >= 0.4 else "#c0392b")
                                    st.markdown(
                                        f'<div style="background:#f0f5ff;border:1px solid #c5d0e0;border-left:3px solid #1565c0;'
                                        f'border-radius:6px;padding:8px 16px;display:flex;gap:30px;margin:6px 0 18px;flex-wrap:wrap;">'
                                        f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Trendline</span>'
                                        f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#1565c0;">{cp["tl_type"]}</div></div>'
                                        f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">Equation</span>'
                                        f'<div style="font-family:DM Mono,monospace;font-weight:600;color:#1a2540;">{cp["equation"]}</div></div>'
                                        f'<div><span style="font-size:.65rem;color:#4a5e80;text-transform:uppercase;">R²</span>'
                                        f'<div style="font-family:Syne,sans-serif;font-weight:700;color:{tl_r2_col2};">{_r2:.4f}</div></div>'
                                        f'</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div style="margin-bottom:18px;"></div>', unsafe_allow_html=True)

                            # ── Download buttons (Excel | PDF side-by-side) ──
                            xlsx_ready5 = bool(st.session_state.get("t5_report_xlsx"))
                            pdf_ready5  = bool(st.session_state.get("t5_report_pdf"))

                            if xlsx_ready5 or pdf_ready5:
                                st.markdown("---")
                                st.markdown('<div class="section-label">📥 DOWNLOAD REPORT</div>', unsafe_allow_html=True)
                                if st.session_state.get("t5_meta_str"):
                                    st.markdown(
                                        f'<div style="background:#f0f5ff;border-left:3px solid #1565c0;border-radius:6px;'
                                        f'padding:10px 16px;font-size:.78rem;color:#2a3a55;margin-bottom:12px;">'
                                        f'{st.session_state["t5_meta_str"]}</div>', unsafe_allow_html=True)

                                stamp5 = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                                dl5_a, dl5_b = st.columns(2)
                                with dl5_a:
                                    if xlsx_ready5:
                                        st.download_button(
                                            label="⬇️  Download Excel (.xlsx)",
                                            data=st.session_state["t5_report_xlsx"],
                                            file_name=f"multi_chart_report_{stamp5}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key="t5_dl_xlsx", use_container_width=True)
                                with dl5_b:
                                    if pdf_ready5:
                                        st.download_button(
                                            label="⬇️  Download PDF (.pdf)",
                                            data=st.session_state["t5_report_pdf"],
                                            file_name=f"multi_chart_report_{stamp5}.pdf",
                                            mime="application/pdf",
                                            key="t5_dl_pdf", use_container_width=True)
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
