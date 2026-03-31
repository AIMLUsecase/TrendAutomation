import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, leaves_list

st.set_page_config(layout="wide", page_title="Analysis Dashboard")

# ============================
# HEADER
# ============================
st.markdown("""
<h1 style='text-align:center;color:#1f4e79;'>
ANALYSIS DASHBOARD | Dynamic Variable Comparison & Trend Analysis
</h1>
""", unsafe_allow_html=True)

# ============================
# FILE UPLOAD
# ============================
file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if file:

    # ============================
    # LOAD DATA
    # ============================
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

    # ============================
    # TABS
    # ============================
    tab1, tab2, tab3 = st.tabs(["⚙️ Preprocess", "📊 Correlation", "📈 Analysis"])

    # ============================
    # TAB 1: PREPROCESS
    # ============================
    with tab1:

        st.subheader("Data Cleaning & Outlier Treatment")

        col1, col2 = st.columns(2)

        with col1:
            lower = st.slider("Lower Percentile", 0, 50, 5)

        with col2:
            upper = st.slider("Upper Percentile", 50, 100, 95)

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

            st.success("Outliers removed successfully")

        st.markdown("### Cleaned Data Preview")
        st.dataframe(df_clean, use_container_width=True, hide_index=True)

    # ============================
    # TAB 2: CORRELATION
    # ============================
    with tab2:

        st.subheader("📊 Advanced Correlation Analysis")

        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns")
            st.stop()

        # ============================
        # FULL CORRELATION MATRIX
        # ============================
        corr = df[numeric_cols].corr()

        st.markdown("### 🔷 Full Correlation Heatmap")

        fig_full = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12}
        ))

        fig_full.update_layout(
            height=900,
            width=1400,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10))
        )

        st.plotly_chart(
            fig_full,
            use_container_width=True,
            config={"scrollZoom": True}   # ✅ zoom enabled
        )

        # ============================
        # TOP CORRELATED FEATURES
        # ============================
        st.markdown("### 🔷 Top Correlated Variables")

        target_col = st.selectbox("Select Variable for Focused Correlation", numeric_cols)

        top_n = st.slider("Number of Top Variables", 5, 25, 15)

        corr_target = corr[target_col].abs().sort_values(ascending=False).head(top_n).index
        corr_filtered = corr.loc[corr_target, corr_target]

        fig_top = go.Figure(data=go.Heatmap(
            z=corr_filtered.values,
            x=corr_filtered.columns,
            y=corr_filtered.columns,
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=np.round(corr_filtered.values, 2),
            texttemplate="%{text}",
            textfont={"size": 14}
        ))

        fig_top.update_layout(
            height=700,
            width=1000,
            xaxis=dict(tickangle=-45)
        )

        st.plotly_chart(fig_top, use_container_width=True)

        # ============================
        # ADVANCED: CLUSTERED HEATMAP
        # ============================
        st.markdown("### 🔷 Clustered Correlation (Advanced Analytics)")

        try:
            

            # Hierarchical clustering
            linked = linkage(corr, method='ward')

            # Reorder columns based on clustering
            order = leaves_list(linked)
            corr_clustered = corr.iloc[order, order]

            fig_cluster = go.Figure(data=go.Heatmap(
                z=corr_clustered.values,
                x=corr_clustered.columns,
                y=corr_clustered.columns,
                colorscale="RdYlGn",
                zmin=-1, zmax=1,
                text=np.round(corr_clustered.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12}
            ))

            fig_cluster.update_layout(
                height=900,
                width=1400,
                xaxis=dict(tickangle=-45)
            )

            st.plotly_chart(fig_cluster, use_container_width=True)

            st.success("Variables grouped automatically based on similarity")

        except Exception as e:
            st.warning("Clustering not available. Install scipy if needed.")    # ============================
    # TAB 3: ANALYSIS
    # ============================
    with tab3:

        st.subheader("Dynamic Analysis")

        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns")
            st.stop()

        date_list = sorted(df[date_col].dt.date.unique())

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            start_date = st.selectbox("Start Date", date_list)

        with col2:
            end_date = st.selectbox("End Date", date_list, index=len(date_list)-1)

        with col3:
            primary = st.selectbox("Primary Selection", numeric_cols)

        with col4:
            secondary = st.selectbox("Secondary Selection", numeric_cols, index=1)

        if start_date > end_date:
            st.error("Start Date cannot be greater than End Date")
            st.stop()

        # FILTER DATA
        df_f = df[
            (df[date_col].dt.date >= start_date) &
            (df[date_col].dt.date <= end_date)
        ]

        if len(df_f) < 2:
            st.warning("Not enough data")
            st.stop()

        x = df_f[primary]
        y = df_f[secondary]

        # ============================
        # STATS
        # ============================
        slope, intercept, r_val, p, std_err = linregress(x, y)
        r2 = r_val**2
        n = len(x)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
        corr_val = x.corr(y)

        def relation_strength(c):
            if abs(c) < 0.3:
                return "Weak"
            elif abs(c) < 0.7:
                return "Moderate"
            else:
                return "Strong"

        strength = relation_strength(corr_val)
        relation = "Positive" if corr_val > 0 else "Negative"

        # ============================
        # CHART WITH EXP TREND
        # ============================
        # ============================
        # CHART WITH EXP TRENDLINE
        # ============================
        fig = go.Figure()

        # Primary
        fig.add_trace(go.Scatter(
            x=df_f[date_col],
            y=x,
            name=primary,
            line=dict(width=3)
        ))

        # Secondary
        fig.add_trace(go.Scatter(
        x=df_f[date_col],
        y=y,
        name=secondary,
        yaxis="y2",
        line=dict(width=3)
        ))

        # ✅ Proper Exponential Trendline (on secondary variable)
        try:
            df_trend = df_f[[date_col, secondary]].dropna()
            df_trend = df_trend[df_trend[secondary] > 0]  # log-safe

            t = np.arange(len(df_trend))
            y_log = np.log(df_trend[secondary])

            coef = np.polyfit(t, y_log, 1)
            trend = np.exp(coef[1]) * np.exp(coef[0] * t)

            fig.add_trace(go.Scatter(
                x=df_trend[date_col],
                y=trend,
                name="Exponential Trend",
                line=dict(dash="dash", width=3)
            ))

        except:
            st.warning("Trendline could not be computed (check data positivity)")

        fig.update_layout(
            height=500,
            title=f"{primary} vs {secondary}",
            yaxis=dict(title=primary),
            yaxis2=dict(title=secondary, overlaying='y', side='right')
        )

        st.plotly_chart(fig, use_container_width=True)


        # ============================
        # 📊 REGRESSION TABLE (2 COLUMN ONLY)
        # ============================
        reg_table = pd.DataFrame({
            "Metric": ["Slope", "R²", "Adjusted R²"],
            "Value": [
                round(slope, 4),
                round(r2, 4),
                round(adj_r2, 4)
            ]
        })

        st.markdown("### 📈 Regression Metrics")
        st.dataframe(reg_table, use_container_width=True, hide_index=True)
        #st.table(reg_table)
        # ============================
        # SUMMARY TABLE
        # ============================
        summary = pd.DataFrame({
            "Metric": ["Min", "Max", "Average", "Std Dev"],
            primary: [
                x.min(), x.max(), x.mean(), x.std()
            ],
            secondary: [
                y.min(), y.max(), y.mean(), y.std(),
               
            ]
        })

        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ============================
        # KEY INSIGHTS BLOCK
        # ============================
        st.markdown(f"""
        <div style="background-color:#2f2f2f;padding:25px;border-radius:10px;color:white">

        <h2 style="color:#9be564;">Key Insights</h2>

        There is a <b>{strength} {relation}</b> relationship between <b>{primary}</b> and <b>{secondary}</b>.

        <h2 style="color:#9be564;">Process Trend Interpretation</h2>

        <b>Primary Variable:</b> {primary}<br>
        <b>Secondary Variable:</b> {secondary}<br><br>

        <b>Selected Date Range:</b> {start_date} to {end_date}

        <h2 style="color:#9be564;">Statistical Summary</h2>

        The <b>{primary}</b> ranges from <b>{x.min():.2f}</b> to <b>{x.max():.2f}</b>, 
        with an average of <b>{x.mean():.2f}</b> and standard deviation <b>{x.std():.2f}</b>.

        <br><br>

        The <b>{secondary}</b> ranges from <b>{y.min():.2f}</b> to <b>{y.max():.2f}</b>, 
        with an average of <b>{y.mean():.2f}</b> and standard deviation <b>{y.std():.2f}</b>.

        <h2 style="color:#9be564;">Correlation Analysis</h2>

        The calculated correlation coefficient is <b>{corr_val:.3f}</b>, indicating a 
        <b>{relation}</b> relationship.

        </div>
        """, unsafe_allow_html=True)