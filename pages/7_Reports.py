"""
Reports and Data Export page — PDF Executive Intelligence Briefings, CSV Registry Dumps & Chart Exports.
"""
from datetime import datetime, date
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import func

from database.db import get_db_session
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
    apply_dark_theme,
    get_all_datasets,
    load_fire_events_df,
    COLOR_ACCENT,
    COLOR_GOLD,
    COLOR_SECONDARY
)
from src.reporting.generator import ReportGenerator
from src.spatial.hotspots import HotspotAnalyzer

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="📄 Reports & Exports — Forest Fire Intelligence",
    layout="wide",
    page_icon="📄"
)
initialize_schema()
inject_css()

st.title("📄 Executive Reports & Data Export Center")
st.caption("Generate formal PDF intelligence dossiers, extract database catalogs into CSV format, and export high-resolution analytical graphics.")

# -----------------------------------------------------------------------------
# Global Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Report Filters")
datasets = get_all_datasets()
dataset_options = {"All Datasets": None}
for d in datasets:
    dataset_options[f"#{d.dataset_id} - {d.dataset_name} ({d.record_count:,})"] = d.dataset_id

selected_ds_label = st.sidebar.selectbox("Catalog Filter", options=list(dataset_options.keys()), index=0)
selected_ds_id = dataset_options[selected_ds_label]

fire_df = load_fire_events_df(dataset_id=selected_ds_id)

# -----------------------------------------------------------------------------
# Section 1: GENERATE PDF REPORT
# -----------------------------------------------------------------------------
st.markdown("### 📑 Executive PDF Intelligence Briefing")

with st.container():
    p_col1, p_col2 = st.columns([2, 1])

    with p_col1:
        report_title = st.text_input(
            "Report Title",
            value="Wildfire Geospatial Intelligence Assessment",
            help="Custom document title for header banner"
        )
        selected_sections = st.multiselect(
            "Select Dossier Sections to Compile",
            options=["Overview Statistics", "Clustering Results", "Environmental Analysis", "Anomaly Summary", "Risk Summary"],
            default=["Overview Statistics", "Clustering Results", "Environmental Analysis", "Anomaly Summary", "Risk Summary"]
        )

    with p_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        gen_pdf_btn = st.button("🚀 Compile & Build PDF Dossier", type="primary")

    if gen_pdf_btn:
        with st.spinner("Compiling database records into styled PDF document..."):
            # Prepare statistics
            kpis = {
                "Total Fire Events": f"{len(fire_df):,}",
                "Datasets Tracked": f"{len(datasets):,}",
                "Active Grid Scope": "Global / Regional"
            }
            with get_db_session() as session:
                kpis["Detected Anomalies"] = f"{session.query(func.count(AnomalyRecord.anomaly_id)).scalar() or 0:,}"
                kpis["Identified Clusters"] = f"{session.query(func.count(GeographicalCluster.cluster_id)).scalar() or 0:,}"

            # Hotspots
            hotspot_list = []
            if not fire_df.empty:
                top_h = HotspotAnalyzer.get_top_hotspots(fire_df, n=5)
                hotspot_list = top_h.to_dict("records")

            # Risk counts
            risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
            try:
                with get_db_session() as session:
                    rc_rows = session.query(RiskPrediction.risk_level, func.count(RiskPrediction.prediction_id)).group_by(RiskPrediction.risk_level).all()
                    for lvl, cnt in rc_rows:
                        if lvl in risk_counts:
                            risk_counts[lvl] = cnt
            except Exception:
                pass

            pdf_bytes = ReportGenerator.generate_summary_report(
                title=report_title,
                sections=selected_sections,
                kpi_data=kpis,
                hotspot_data=hotspot_list,
                risk_summary=risk_counts
            )

            st.success("✅ Executive PDF dossier compiled successfully.")
            st.download_button(
                label="📥 Download Generated PDF Report",
                data=pdf_bytes,
                file_name=f"Wildfire_Intelligence_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary"
            )

# -----------------------------------------------------------------------------
# Section 2: CSV EXPORTS (Raw Entities)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💾 Raw Table CSV Exports")

csv_tab1, csv_tab2, csv_tab3, csv_tab4, csv_tab5 = st.tabs([
    "Fire Events",
    "Environmental Records",
    "Anomaly Records",
    "Model Experiments",
    "Risk Predictions"
])

# 1. Fire Events
with csv_tab1:
    fe_col1, fe_col2, fe_col3 = st.columns(3)
    with fe_col1:
        f_country = st.selectbox("Country Filter", options=["All"] + sorted([c for c in fire_df["country"].unique() if pd.notna(c)]) if not fire_df.empty else ["All"])
    with fe_col2:
        f_season = st.selectbox("Season Filter", options=["All", "Spring", "Summer", "Fall", "Winter"])
    with fe_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        export_fe_df = fire_df.copy()
        if not export_fe_df.empty:
            if f_country != "All" and "country" in export_fe_df.columns:
                export_fe_df = export_fe_df[export_fe_df["country"] == f_country]
            if f_season != "All" and "season" in export_fe_df.columns:
                export_fe_df = export_fe_df[export_fe_df["season"] == f_season]

            st.download_button(
                label=f"📥 Download Fire Events ({len(export_fe_df):,} rows)",
                data=export_fe_df.to_csv(index=False).encode("utf-8"),
                file_name="fire_events_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No fire events available to export.")

# 2. Environmental Records
with csv_tab2:
    with get_db_session() as session:
        envs = session.query(EnvironmentalRecord).limit(5000).all()
        if envs:
            env_export_df = pd.DataFrame([{
                "record_id": r.record_id,
                "dataset_id": r.dataset_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "timestamp": r.timestamp,
                "temperature": r.temperature,
                "humidity": r.humidity,
                "wind_speed": r.wind_speed,
                "wind_direction": r.wind_direction,
                "precipitation": r.precipitation
            } for r in envs])
            st.dataframe(env_export_df.head(5), use_container_width=True)
            st.download_button(
                label=f"📥 Download Environmental Records ({len(env_export_df):,} rows)",
                data=env_export_df.to_csv(index=False).encode("utf-8"),
                file_name="environmental_records_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No environmental records found in database.")

# 3. Anomaly Records
with csv_tab3:
    with get_db_session() as session:
        anoms = session.query(AnomalyRecord).all()
        if anoms:
            anom_export_df = pd.DataFrame([{
                "anomaly_id": a.anomaly_id,
                "event_id": a.event_id,
                "anomaly_score": a.anomaly_score,
                "anomaly_type": a.anomaly_type,
                "latitude": a.latitude,
                "longitude": a.longitude,
                "brightness": a.brightness,
                "frp": a.frp,
                "detected_at": a.detected_at
            } for a in anoms])
            st.dataframe(anom_export_df.head(5), use_container_width=True)
            st.download_button(
                label=f"📥 Download Anomaly Records ({len(anom_export_df):,} rows)",
                data=anom_export_df.to_csv(index=False).encode("utf-8"),
                file_name="anomaly_records_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No anomaly records detected or stored.")

# 4. Model Experiments
with csv_tab4:
    with get_db_session() as session:
        exps = session.query(ModelExperiment).all()
        if exps:
            exp_export_df = pd.DataFrame([{
                "experiment_id": e.experiment_id,
                "model_name": e.model_name,
                "accuracy": e.accuracy,
                "precision": e.precision_score,
                "recall": e.recall,
                "f1_score": e.f1_score,
                "roc_auc": e.roc_auc,
                "created_at": e.created_at
            } for e in exps])
            st.dataframe(exp_export_df, use_container_width=True)
            st.download_button(
                label=f"📥 Download Model Experiments ({len(exp_export_df):,} rows)",
                data=exp_export_df.to_csv(index=False).encode("utf-8"),
                file_name="model_experiments_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No model experiments found.")

# 5. Risk Predictions
with csv_tab5:
    with get_db_session() as session:
        risks = session.query(RiskPrediction).all()
        if risks:
            risk_export_df = pd.DataFrame([{
                "prediction_id": r.prediction_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "risk_level": r.risk_level,
                "prediction_score": r.prediction_score,
                "model_name": r.model_name,
                "created_at": r.created_at
            } for r in risks])
            st.dataframe(risk_export_df.head(5), use_container_width=True)
            st.download_button(
                label=f"📥 Download Risk Predictions ({len(risk_export_df):,} rows)",
                data=risk_export_df.to_csv(index=False).encode("utf-8"),
                file_name="risk_predictions_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No risk predictions generated yet.")

# -----------------------------------------------------------------------------
# Section 3: STATISTICS EXPORT
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Statistical Summaries Export")

st_col1, st_col2, st_col3 = st.columns(3)

with st_col1:
    st.markdown("**Monthly Activity Summary**")
    if not fire_df.empty and "acquisition_date" in fire_df.columns:
        m_df = fire_df.copy()
        m_df["year_month"] = pd.to_datetime(m_df["acquisition_date"]).dt.to_period("M").astype(str)
        monthly_stats = m_df.groupby("year_month").agg(
            total_observations=("latitude", "count"),
            avg_brightness=("brightness", "mean") if "brightness" in m_df.columns else ("latitude", "count"),
            avg_frp=("fire_radiative_power", "mean") if "fire_radiative_power" in m_df.columns else ("latitude", "count")
        ).reset_index()
        monthly_stats["avg_brightness"] = monthly_stats["avg_brightness"].round(1)
        monthly_stats["avg_frp"] = monthly_stats["avg_frp"].round(1)

        st.download_button(
            label="📥 Download Monthly Summary CSV",
            data=monthly_stats.to_csv(index=False).encode("utf-8"),
            file_name="monthly_fire_summary.csv",
            mime="text/csv"
        )
    else:
        st.write("No date data available.")

with st_col2:
    st.markdown("**Regional Aggregations**")
    if not fire_df.empty and "country" in fire_df.columns:
        reg_stats = fire_df.groupby("country").agg(
            total_fires=("latitude", "count"),
            avg_frp=("fire_radiative_power", "mean") if "fire_radiative_power" in fire_df.columns else ("latitude", "count"),
            peak_brightness=("brightness", "max") if "brightness" in fire_df.columns else ("latitude", "count")
        ).reset_index().sort_values(by="total_fires", ascending=False)
        reg_stats["avg_frp"] = reg_stats["avg_frp"].round(1)
        reg_stats["peak_brightness"] = reg_stats["peak_brightness"].round(1)

        st.download_button(
            label="📥 Download Regional Summary CSV",
            data=reg_stats.to_csv(index=False).encode("utf-8"),
            file_name="regional_fire_summary.csv",
            mime="text/csv"
        )
    else:
        st.write("No regional data available.")

with st_col3:
    st.markdown("**Spatial Cluster Summary**")
    with get_db_session() as session:
        clusters = session.query(GeographicalCluster).all()
        if clusters:
            clust_df = pd.DataFrame([{
                "cluster_id": c.cluster_id,
                "algorithm": c.algorithm,
                "centroid_latitude": c.centroid_latitude,
                "centroid_longitude": c.centroid_longitude,
                "observation_count": c.observation_count,
                "created_at": c.created_at
            } for c in clusters])

            st.download_button(
                label="📥 Download Cluster Summary CSV",
                data=clust_df.to_csv(index=False).encode("utf-8"),
                file_name="cluster_centroids_summary.csv",
                mime="text/csv"
            )
        else:
            st.write("No clusters stored.")

# -----------------------------------------------------------------------------
# Section 4: CHARTS (PNG via Kaleido)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🖼️ High-Resolution Graphic Exports (PNG)")

ch_col1, ch_col2 = st.columns(2)

# Build timeline chart
timeline_fig = None
if not fire_df.empty and "acquisition_date" in fire_df.columns:
    df_t = fire_df.copy()
    df_t["year_month"] = pd.to_datetime(df_t["acquisition_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = df_t.groupby("year_month").size().reset_index(name="count")
    timeline_fig = px.line(
        monthly,
        x="year_month",
        y="count",
        markers=True,
        title="Fire Activity Timeline",
        labels={"year_month": "Month", "count": "Observations"}
    )
    apply_dark_theme(timeline_fig)

# Build correlation chart
corr_fig = None
num_cols = [c for c in ["brightness", "fire_radiative_power", "confidence"] if c in fire_df.columns]
if len(num_cols) >= 2:
    corr_fig = px.imshow(
        fire_df[num_cols].corr(),
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    apply_dark_theme(corr_fig)

with ch_col1:
    st.markdown("**Fire Activity Timeline Graphic**")
    if timeline_fig is not None:
        try:
            img_bytes = timeline_fig.to_image(format="png", width=1000, height=500, scale=2)
            st.download_button(
                label="📥 Download Timeline Graphic (PNG)",
                data=img_bytes,
                file_name="fire_activity_timeline.png",
                mime="image/png"
            )
        except Exception:
            st.warning("Kaleido image export engine is not available in the current environment. PNG download skipped.")
    else:
        st.info("No timeline data loaded.")

with ch_col2:
    st.markdown("**Correlation Matrix Graphic**")
    if corr_fig is not None:
        try:
            img_bytes_c = corr_fig.to_image(format="png", width=800, height=600, scale=2)
            st.download_button(
                label="📥 Download Correlation Heatmap (PNG)",
                data=img_bytes_c,
                file_name="correlation_heatmap.png",
                mime="image/png"
            )
        except Exception:
            st.warning("Kaleido image export engine is not available in the current environment. PNG download skipped.")
    else:
        st.info("No correlation data loaded.")
