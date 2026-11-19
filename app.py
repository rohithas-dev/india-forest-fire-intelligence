"""
AI-Driven Forest Fire Intelligence System — Overview Dashboard.
Main entry point for the Streamlit multi-page application.
"""
from datetime import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func, desc

from database.db import get_db_session, get_database_url
from database.models import (
    FireEvent,
    Dataset,
    EnvironmentalRecord,
    GeographicalCluster,
    AnomalyRecord,
    ModelExperiment,
    RiskPrediction
)
from database.schema import initialize_schema
from src.utils.ui import (
    inject_css,
    render_hero,
    render_metric_card,
    apply_dark_theme,
    COLOR_ACCENT,
    COLOR_GOLD,
    COLOR_SECONDARY,
    COLOR_BG
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🔥 Forest Fire Intelligence",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# 1. Silently initialize schema
try:
    initialize_schema()
except Exception:
    pass

# 2. Inject CSS
inject_css()

# -----------------------------------------------------------------------------
# 3. Hero Section
# -----------------------------------------------------------------------------
render_hero(
    title="AI-Driven Forest Fire Intelligence System",
    subtitle="Satellite Wildfire Surveillance, Spatial Clustering & Predictive Risk Modeling",
    description=(
        "An enterprise geospatial intelligence platform combining NASA FIRMS thermal anomaly streams, "
        "meteorological parameters, unsupervised clustering (DBSCAN & K-Means), anomaly detection (Isolation Forests & LOF), "
        "and supervised gradient boosting algorithms to model, track, and forecast catastrophic wildfire conditions."
    )
)

# -----------------------------------------------------------------------------
# Data Loading & Aggregations
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_kpi_counts():
    """Fetch database total counts for top metric cards."""
    counts = {"fire_events": 0, "datasets": 0, "anomalies": 0, "clusters": 0}
    try:
        with get_db_session() as session:
            counts["fire_events"] = session.query(func.count(FireEvent.event_id)).scalar() or 0
            counts["datasets"] = session.query(func.count(Dataset.dataset_id)).scalar() or 0
            counts["anomalies"] = session.query(func.count(AnomalyRecord.anomaly_id)).scalar() or 0
            counts["clusters"] = session.query(func.count(GeographicalCluster.cluster_id)).scalar() or 0
    except Exception:
        pass
    return counts


@st.cache_data(ttl=60)
def load_overview_stats():
    """Fetch quick statistics (Top country, peak month, avg FRP) and timeline."""
    stats = {
        "top_country": "N/A",
        "top_country_count": 0,
        "peak_month": "N/A",
        "peak_month_count": 0,
        "avg_frp": 0.0,
        "timeline_df": pd.DataFrame()
    }
    try:
        with get_db_session() as session:
            # Top country
            top_c = (
                session.query(FireEvent.country, func.count(FireEvent.event_id).label("c"))
                .filter(FireEvent.country.isnot(None), FireEvent.country != "")
                .group_by(FireEvent.country)
                .order_by(desc("c"))
                .first()
            )
            if top_c:
                stats["top_country"] = top_c[0]
                stats["top_country_count"] = top_c[1]

            # Avg FRP
            avg_frp = session.query(func.avg(FireEvent.fire_radiative_power)).scalar()
            if avg_frp is not None:
                stats["avg_frp"] = round(float(avg_frp), 1)

            # Monthly timeline
            dates = (
                session.query(FireEvent.acquisition_date)
                .filter(FireEvent.acquisition_date.isnot(None))
                .all()
            )
            if dates:
                date_list = [d[0] for d in dates if d[0]]
                df_dates = pd.DataFrame({"acquisition_date": pd.to_datetime(date_list)})
                df_dates["year_month"] = df_dates["acquisition_date"].dt.to_period("M").dt.to_timestamp()
                monthly = df_dates.groupby("year_month").size().reset_index(name="count")
                monthly = monthly.sort_values("year_month")
                stats["timeline_df"] = monthly

                if not monthly.empty:
                    top_m_row = monthly.loc[monthly["count"].idxmax()]
                    stats["peak_month"] = top_m_row["year_month"].strftime("%B %Y")
                    stats["peak_month_count"] = int(top_m_row["count"])
    except Exception:
        pass
    return stats


kpis = load_kpi_counts()
stats = load_overview_stats()
is_db_empty = (kpis["fire_events"] == 0 and kpis["datasets"] == 0)

# -----------------------------------------------------------------------------
# 4. 4 KPI Metric Cards
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        render_metric_card(
            label="Total Fire Observations",
            value=f"{kpis['fire_events']:,}",
            subtext="Satellite thermal points"
        ),
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        render_metric_card(
            label="Datasets Loaded",
            value=f"{kpis['datasets']:,}",
            subtext="Imported or simulated catalogs"
        ),
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        render_metric_card(
            label="Anomalies Detected",
            value=f"{kpis['anomalies']:,}",
            subtext="Isolation Forest & LOF flagged"
        ),
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        render_metric_card(
            label="Geographical Clusters",
            value=f"{kpis['clusters']:,}",
            subtext="Centroids identified"
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Empty Database Notice
# -----------------------------------------------------------------------------
if is_db_empty:
    st.markdown(
        """
        <div class="fire-alert-box">
            <h3 style="margin-top:0; color:#FF6B35;">🚀 Getting Started with Forest Fire Intelligence</h3>
            <p>Your database is currently empty. To populate the system with data, you can:</p>
            <ol style="margin-bottom: 8px; line-height: 1.6;">
                <li>Navigate to <b>📊 Dataset Explorer</b> in the sidebar to upload a NASA FIRMS CSV or JSON dataset.</li>
                <li>Or run <code>python initialize_database.py</code> from the terminal to generate realistic demonstration wildfire data.</li>
            </ol>
            <p style="margin-bottom:0; font-size:0.88rem; color:#A0AEC0;">
                All subsequent analytics, spatial clustering, and AI risk engines will dynamically activate once data is imported.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 6. Fire Activity Over Time Chart
# -----------------------------------------------------------------------------
st.markdown("### 📈 Fire Activity Over Time")

timeline_df = stats["timeline_df"]

if not timeline_df.empty:
    fig_time = px.line(
        timeline_df,
        x="year_month",
        y="count",
        markers=True,
        labels={"year_month": "Month & Year", "count": "Active Fires Detected"},
        title="Monthly Satellite Fire Observation Volume"
    )
    fig_time.update_traces(
        line=dict(color=COLOR_ACCENT, width=3),
        marker=dict(size=7, color=COLOR_GOLD, symbol="circle")
    )
    apply_dark_theme(fig_time)
    fig_time.update_layout(height=360)
    st.plotly_chart(fig_time, use_container_width=True)
else:
    # Placeholder chart
    dummy_dates = pd.date_range(end=datetime.now(), periods=12, freq="MS")
    dummy_counts = [0] * 12
    placeholder_df = pd.DataFrame({"Month": dummy_dates, "Observations": dummy_counts})
    fig_ph = px.line(
        placeholder_df,
        x="Month",
        y="Observations",
        title="Monthly Satellite Fire Observations (No Data Loaded Yet)"
    )
    fig_ph.update_traces(line=dict(color="#4A5568", dash="dash"))
    apply_dark_theme(fig_ph)
    fig_ph.update_layout(height=300)
    st.plotly_chart(fig_ph, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. Quick Stats (3 Columns)
# -----------------------------------------------------------------------------
st.markdown("### ⚡ Quick Operational Insights")
q1, q2, q3 = st.columns(3)

with q1:
    top_c_label = stats["top_country"]
    sub_c = f"{stats['top_country_count']:,} observations" if stats["top_country_count"] > 0 else "No data"
    st.markdown(
        render_metric_card("Top Affected Country", top_c_label, sub_c),
        unsafe_allow_html=True
    )

with q2:
    sub_m = f"{stats['peak_month_count']:,} fire events" if stats["peak_month_count"] > 0 else "No data"
    st.markdown(
        render_metric_card("Most Active Month", stats["peak_month"], sub_m),
        unsafe_allow_html=True
    )

with q3:
    frp_val = f"{stats['avg_frp']} MW" if stats["avg_frp"] > 0 else "0.0 MW"
    st.markdown(
        render_metric_card("Avg Fire Radiative Power", frp_val, "Radiative energy intensity"),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. Navigation Cards (2x2 Grid)
# -----------------------------------------------------------------------------
st.markdown("### 🧭 Intelligence Modules")
nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-icon">📊</div>
            <div class="nav-title">Dataset Explorer</div>
            <div class="nav-desc">
                Upload NASA FIRMS, MODIS, or VIIRS CSV/JSON feeds. Inspect schema data quality, missingness profiles,
                and filter thermal observations by region and confidence thresholds.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-icon">🌡️</div>
            <div class="nav-title">Environmental Analysis</div>
            <div class="nav-desc">
                Correlate temperature, relative humidity, wind speed, and precipitation against fire intensity.
                Compute seasonal trends and multi-variate Pearson heatmaps.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col2:
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-icon">🗺️</div>
            <div class="nav-title">Geographical Intelligence</div>
            <div class="nav-desc">
                Interactive spatial cluster discovery with K-Means and density-based DBSCAN.
                Render high-risk regional hotspot boundaries and thermal density maps.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-icon">🧪</div>
            <div class="nav-title">AI Laboratory & Risk Engine</div>
            <div class="nav-desc">
                Train XGBoost, Random Forest, and Gradient Boosting fire classifiers.
                Detect extreme thermal anomalies with Isolation Forest and generate predictive risk grids.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 9. Sidebar System Info & Quick Nav Links
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🛰️ System Information")
db_url = get_database_url()
db_backend = "PostgreSQL (Enterprise)" if "postgresql" in db_url.lower() else "SQLite (Local Storage)"
st.sidebar.info(
    f"**Engine:** {db_backend}\n\n"
    f"**Status:** Connected & Ready\n\n"
    f"**Sync Timestamp:**\n`{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}`"
)

st.sidebar.markdown("### 📌 Quick Navigation")
st.sidebar.page_link("app.py", label="🔥 Overview Dashboard", icon="🏠")
st.sidebar.page_link("pages/1_Dataset_Explorer.py", label="Dataset Explorer", icon="📊")
st.sidebar.page_link("pages/2_Geographical_Intelligence.py", label="Geographical Intelligence", icon="🗺️")
st.sidebar.page_link("pages/3_Environmental_Analysis.py", label="Environmental Analysis", icon="🌡️")
st.sidebar.page_link("pages/4_AI_Laboratory.py", label="AI Laboratory", icon="🧪")
st.sidebar.page_link("pages/5_Anomaly_Investigation.py", label="Anomaly Investigation", icon="🔍")
st.sidebar.page_link("pages/6_Risk_Intelligence.py", label="Risk Intelligence", icon="⚠️")
st.sidebar.page_link("pages/7_Reports.py", label="Reports & Export", icon="📄")

st.sidebar.markdown("---")
st.sidebar.caption("AI-Driven Forest Fire Intelligence System v2.0")
