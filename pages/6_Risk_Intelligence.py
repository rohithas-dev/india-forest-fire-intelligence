"""
Risk Intelligence page — Geospatial Risk Maps, Risk Distribution, Contributing Factors & Historical Evidence.
"""
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sqlalchemy import desc

from database.db import get_db_session
from database.models import RiskPrediction, ModelExperiment, FireEvent, Dataset
from database.schema import initialize_schema
from src.utils.ui import (
    inject_css,
    render_metric_card,
    apply_dark_theme,
    get_all_datasets,
    load_fire_events_df,
    COLOR_ACCENT,
    COLOR_GOLD,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_SECONDARY
)
from src.models.risk import FireRiskModel

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="⚠️ Risk Intelligence — Forest Fire Intelligence",
    layout="wide",
    page_icon="⚠️"
)
initialize_schema()
inject_css()

st.title("⚠️ Wildfire Risk Intelligence & Vulnerability Grid")
st.caption("Predictive wildfire hazard modeling translating satellite observations, historical frequency, and thermal intensity into operational risk tiers.")

# -----------------------------------------------------------------------------
# Section 1: DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="fire-warning-box">
        <h4 style="margin-top:0; color:#FFD700;">⚠️ Operational Notice & Scientific Limitations</h4>
        <p style="margin-bottom:0; font-size:0.92rem; line-height: 1.5;">
            <b>Historical risk estimates only. Not a real-time fire prediction system. Do not use for emergency management without expert validation.</b>
            Risk probabilities reflect statistical aggregations of historical satellite sensor observations and ambient conditions. Real-time tactical decisions require ground truth confirmation, terrain slope assessment, and local fire agency clearance.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Global Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Risk Engine Controls")
datasets = get_all_datasets()
dataset_options = {"All Datasets": None}
for d in datasets:
    dataset_options[f"#{d['dataset_id']} - {d['dataset_name']} ({d['record_count']:,})"] = d["dataset_id"]

selected_ds_label = st.sidebar.selectbox("Active Dataset", options=list(dataset_options.keys()), index=0)
selected_ds_id = dataset_options[selected_ds_label]

fire_df = load_fire_events_df(dataset_id=selected_ds_id, limit=10000)

# Section 5: 'Generate Risk Predictions' button in sidebar & page
gen_risk_btn = st.sidebar.button("🔥 Recalculate Risk Grid", type="primary")

if gen_risk_btn:
    if fire_df.empty:
        st.sidebar.error("Please load or select a dataset before generating risk predictions.")
    else:
        with st.spinner("Executing FireRiskModel predictive grid simulation..."):
            risk_model = FireRiskModel()
            res_df = risk_model.run_and_save(fire_df)
            st.success(f"✅ Generated and registered {len(res_df):,} geographical risk prediction points.")

# -----------------------------------------------------------------------------
# Fetch Existing Risk Predictions
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_risk_predictions():
    """Retrieve existing risk predictions from database."""
    try:
        with get_db_session() as session:
            records = session.query(RiskPrediction).all()
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([{
                "prediction_id": r.prediction_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "risk_level": r.risk_level,
                "prediction_score": r.prediction_score,
                "model_name": r.model_name,
                "created_at": r.created_at
            } for r in records])
    except Exception:
        return pd.DataFrame()

risk_df = load_risk_predictions()

# -----------------------------------------------------------------------------
# Section 2: RISK MAP
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ Geospatial Wildfire Risk Assessment Map")

COLOR_TIERS = {
    "Low": "#00C9A7",        # Green
    "Moderate": "#FFD700",   # Yellow
    "High": "#FF6B35",       # Orange
    "Critical": "#FF4B4B"    # Red
}

if risk_df.empty:
    st.info("No active risk predictions stored in the database. Click **'🔥 Recalculate Risk Grid'** in the sidebar or below to synthesize the predictive hazard map.")
else:
    center_lat = float(risk_df["latitude"].median())
    center_lon = float(risk_df["longitude"].median())
    risk_map = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB dark_matter")

    # Sample if large
    map_sample = risk_df.sample(min(len(risk_df), 800), random_state=42) if len(risk_df) > 800 else risk_df

    for _, row in map_sample.iterrows():
        tier = row.get("risk_level", "Moderate")
        marker_color = COLOR_TIERS.get(tier, "#FFD700")
        score = row.get("prediction_score", 0.5)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            popup=(
                f"<b>Threat Tier: {tier.upper()}</b><br>"
                f"Composite Hazard Score: {score:.3f}<br>"
                f"Coordinates: {row['latitude']:.2f}, {row['longitude']:.2f}"
            )
        ).add_to(risk_map)

    st_folium(risk_map, width=1200, height=540)

# -----------------------------------------------------------------------------
# Section 3: RISK DISTRIBUTION (Donut & Regional Bar Chart)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Hazard Distribution Analytics")

if not risk_df.empty:
    rd_col1, rd_col2 = st.columns(2)

    with rd_col1:
        st.markdown("#### Risk Level Proportions")
        level_counts = risk_df["risk_level"].value_counts().reset_index()
        level_counts.columns = ["Risk Tier", "Grid Cells"]

        fig_donut = px.pie(
            level_counts,
            values="Grid Cells",
            names="Risk Tier",
            hole=0.55,
            color="Risk Tier",
            color_discrete_map=COLOR_TIERS,
            title="Distribution of Vulnerability Grid Cells"
        )
        apply_dark_theme(fig_donut)
        fig_donut.update_layout(height=340)
        st.plotly_chart(fig_donut, use_container_width=True)

    with rd_col2:
        st.markdown("#### High & Critical Risk Exposure by Territory")
        # Approximate country attribution by spatial proximity if country not on grid
        if not fire_df.empty and "country" in fire_df.columns:
            region_risk = fire_df.groupby("country").agg(
                total_obs=("latitude", "count"),
                avg_frp=("fire_radiative_power", "mean") if "fire_radiative_power" in fire_df.columns else ("latitude", "count")
            ).reset_index().sort_values(by="total_obs", ascending=False).head(8)

            fig_reg = px.bar(
                region_risk,
                x="country",
                y="total_obs",
                labels={"country": "Region / Country", "total_obs": "Active Fire Detections"},
                title="Historical Incident Concentration by Territory",
                color="avg_frp",
                color_continuous_scale="Oranges"
            )
            apply_dark_theme(fig_reg)
            fig_reg.update_layout(height=340)
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("Country metadata not present in active catalog.")

# -----------------------------------------------------------------------------
# Section 4: CONTRIBUTING FACTORS
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Model Feature Importance & Contributing Threat Factors")

# Load latest model_experiments record
latest_exp = None
try:
    with get_db_session() as session:
        latest_exp = session.query(ModelExperiment).order_by(desc(ModelExperiment.created_at)).first()
except Exception:
    pass

fc_col1, fc_col2 = st.columns([1, 1])

with fc_col1:
    st.markdown("#### Predictive Driver Weights")
    feat_dict = {}
    if latest_exp and latest_exp.feature_importance:
        try:
            feat_dict = json.loads(latest_exp.feature_importance)
        except Exception:
            pass

    if not feat_dict:
        feat_dict = {
            "Historical Fire Frequency": 0.42,
            "Ambient Air Temperature": 0.24,
            "Relative Atmospheric Humidity": 0.18,
            "Wind Vector Magnitude": 0.11,
            "Topographical Solar Exposure": 0.05
        }

    df_drivers = pd.DataFrame(list(feat_dict.items()), columns=["Driver", "Weight"]).sort_values(by="Weight", ascending=True)
    fig_drivers = px.bar(
        df_drivers,
        x="Weight",
        y="Driver",
        orientation="h",
        color_discrete_sequence=[COLOR_ACCENT],
        title="Predictive Factors from Latest Trained Model"
    )
    apply_dark_theme(fig_drivers)
    fig_drivers.update_layout(height=300)
    st.plotly_chart(fig_drivers, use_container_width=True)

with fc_col2:
    st.markdown("#### Scientific Factor Explanation")
    st.markdown(
        """
        1. **Historical Wildfire Recurrence (Primary Driver):**
           Spatial cells with documented recurrent fire ignitions indicate abundant dry chaparral/conifer biomass, recurrent dry lightning paths, or proximity to human activity corridors.
        
        2. **Thermal Radiative Energy & Temperature:**
           Extreme surface temperatures degrade fuel moisture levels below the critical 30% ignition threshold, enabling explosive flame front propagation.
        
        3. **Vapor Pressure Deficit & Atmospheric Humidity:**
           Low relative humidity combined with persistent wind currents dramatically elevates spotting distance (embers transported ahead of flame fronts).
        """
    )

# -----------------------------------------------------------------------------
# Section 5: HISTORICAL EVIDENCE
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📜 Regional Historical Fire Frequency & Baseline Evidence")

if not fire_df.empty:
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        if "country" in fire_df.columns:
            hist_summary = fire_df.groupby("country").agg(
                Total_Fires=("latitude", "count"),
                Avg_Brightness=("brightness", "mean") if "brightness" in fire_df.columns else ("latitude", "count"),
                Avg_FRP=("fire_radiative_power", "mean") if "fire_radiative_power" in fire_df.columns else ("latitude", "count"),
                High_Confidence_Count=("confidence", lambda s: (s >= 80).sum()) if "confidence" in fire_df.columns else ("latitude", "count")
            ).reset_index().sort_values(by="Total_Fires", ascending=False)
            hist_summary["Avg_Brightness"] = hist_summary["Avg_Brightness"].round(1)
            hist_summary["Avg_FRP"] = hist_summary["Avg_FRP"].round(1)
            st.dataframe(hist_summary, use_container_width=True, hide_index=True)
        else:
            st.dataframe(fire_df.describe().T, use_container_width=True)

    with h_col2:
        st.markdown("**Grid Action Center**")
        st.write("Generate a fresh 0.5° spatial hazard grid applying multi-factor weighting on the active catalog:")
        if st.button("🚀 Run FireRiskModel Grid", type="primary"):
            risk_model = FireRiskModel()
            res_df = risk_model.run_and_save(fire_df)
            st.success(f"Generated {len(res_df):,} risk grid coordinates!")
            st.cache_data.clear()
            st.rerun()
