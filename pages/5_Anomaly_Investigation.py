"""
Anomaly Investigation page — Isolation Forest & LOF Wildfire Thermal Anomaly Detection.
"""
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from database.db import get_db_session
from database.models import AnomalyRecord, FireEvent, Dataset
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

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🔍 Anomaly Investigation — Forest Fire Intelligence",
    layout="wide",
    page_icon="🔍"
)
initialize_schema()
inject_css()

st.title("🔍 Thermal Anomaly Investigation")
st.caption("Unsupervised anomaly detection engines (Isolation Forest & Local Outlier Factor) to identify unprecedented thermal flare-ups, sensor corruption, or extreme firestorms.")

# -----------------------------------------------------------------------------
# Section 1: DETECTION CONTROLS (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🔍 Anomaly Controls")

datasets = get_all_datasets()
dataset_options = {"All Datasets": None}
for d in datasets:
    dataset_options[f"#{d.dataset_id} - {d.dataset_name} ({d.record_count:,})"] = d.dataset_id

selected_ds_label = st.sidebar.selectbox("Target Dataset", options=list(dataset_options.keys()), index=0)
selected_ds_id = dataset_options[selected_ds_label]

algo_choice = st.sidebar.selectbox(
    "Anomaly Algorithm",
    options=["Isolation Forest", "Local Outlier Factor (LOF)", "Both"],
    index=2,
    help="Isolation Forest isolates anomalies by random partitioning. LOF measures local density deviation."
)

contamination = st.sidebar.slider(
    "Expected Contamination Rate",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Expected fraction of anomalous observations in dataset (default 5%)"
)

run_detect_btn = st.sidebar.button("🚨 Run Anomaly Detection", type="primary")

# Load fire records
fire_df = load_fire_events_df(dataset_id=selected_ds_id, limit=5000)

if fire_df.empty:
    st.info("No fire events available. Import data in Dataset Explorer to run anomaly investigations.")
    st.stop()

# -----------------------------------------------------------------------------
# Anomaly Detection Execution
# -----------------------------------------------------------------------------
if "anomaly_results" not in st.session_state:
    st.session_state["anomaly_results"] = None

# Features for anomaly scoring
features_to_use = [f for f in ["brightness", "fire_radiative_power", "confidence", "latitude", "longitude"] if f in fire_df.columns]

if run_detect_btn:
    with st.spinner("Executing unsupervised anomaly detection ensembles..."):
        df_eval = fire_df.copy().dropna(subset=features_to_use)
        X = df_eval[features_to_use].values

        if len(X) < 10:
            st.error("Insufficient observation points for anomaly fitting.")
        else:
            # 1. Isolation Forest
            iso = IsolationForest(contamination=contamination, random_state=42)
            df_eval["if_pred"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
            df_eval["if_score"] = iso.decision_function(X)  # Lower is more abnormal

            # 2. LOF
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            df_eval["lof_pred"] = lof.fit_predict(X)
            df_eval["lof_score"] = lof.negative_outlier_factor_

            if algo_choice == "Isolation Forest":
                df_eval["is_anomaly"] = (df_eval["if_pred"] == -1)
                df_eval["anomaly_score"] = df_eval["if_score"]
                active_algo_label = "Isolation Forest"
            elif algo_choice == "Local Outlier Factor (LOF)":
                df_eval["is_anomaly"] = (df_eval["lof_pred"] == -1)
                df_eval["anomaly_score"] = df_eval["lof_score"]
                active_algo_label = "Local Outlier Factor"
            else:  # Both
                df_eval["is_anomaly"] = (df_eval["if_pred"] == -1) | (df_eval["lof_pred"] == -1)
                df_eval["anomaly_score"] = (df_eval["if_score"] + df_eval["lof_score"]) / 2.0
                active_algo_label = "Isolation Forest + LOF"

            # Persist detected anomalies to DB
            anomalies_to_db = []
            anom_subset = df_eval[df_eval["is_anomaly"]]
            for _, row in anom_subset.head(500).iterrows():
                anomalies_to_db.append(AnomalyRecord(
                    event_id=int(row["event_id"]) if "event_id" in row and pd.notna(row["event_id"]) else None,
                    anomaly_score=float(row["anomaly_score"]),
                    anomaly_type=algo_choice.lower().replace(" ", "_"),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    brightness=float(row["brightness"]) if "brightness" in row and pd.notna(row["brightness"]) else None,
                    frp=float(row["fire_radiative_power"]) if "fire_radiative_power" in row and pd.notna(row["fire_radiative_power"]) else None,
                    detected_at=datetime.utcnow()
                ))

            if anomalies_to_db:
                try:
                    with get_db_session() as session:
                        session.bulk_save_objects(anomalies_to_db)
                except Exception:
                    pass

            st.session_state["anomaly_results"] = {
                "df": df_eval,
                "algo_choice": algo_choice,
                "active_label": active_algo_label,
                "n_anomalies": int(df_eval["is_anomaly"].sum()),
                "rate": float(df_eval["is_anomaly"].mean()) * 100,
                "threshold": float(np.percentile(df_eval["anomaly_score"], contamination * 100))
            }
            st.success("✅ Anomaly detection executed and flagged records logged.")

results = st.session_state.get("anomaly_results")

# -----------------------------------------------------------------------------
# Section 1 (cont): SUMMARY METRICS
# -----------------------------------------------------------------------------
if results is not None:
    df_eval = results["df"]
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(
            render_metric_card("Anomalies Detected", f"{results['n_anomalies']:,}", f"Flagged by {results['active_label']}"),
            unsafe_allow_html=True
        )
    with sc2:
        st.markdown(
            render_metric_card("Anomaly Rate", f"{results['rate']:.2f}%", f"Target contamination: {contamination*100:.1f}%"),
            unsafe_allow_html=True
        )
    with sc3:
        st.markdown(
            render_metric_card("Evaluated Sample", f"{len(df_eval):,} points", "Active fire detections"),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Section 2: ANOMALY MAP
    # -------------------------------------------------------------------------
    st.markdown("### 🗺️ Geospatial Anomaly Distribution Map")
    st.caption("Red markers represent anomalous detections exceeding density/thermal isolation bounds; blue markers represent baseline observations.")

    center_lat = float(df_eval["latitude"].median())
    center_lon = float(df_eval["longitude"].median())
    anom_map = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB dark_matter")

    # Sample normal points for map responsiveness
    normal_pts = df_eval[~df_eval["is_anomaly"]]
    anom_pts = df_eval[df_eval["is_anomaly"]]

    normal_sample = normal_pts.sample(min(len(normal_pts), 600), random_state=42) if len(normal_pts) > 600 else normal_pts

    for _, row in normal_sample.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#3B82F6",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.6,
            popup=f"Normal Event<br>FRP: {row.get('fire_radiative_power', 0):.1f} MW"
        ).add_to(anom_map)

    for _, row in anom_pts.head(300).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color="#FF4B4B",
            fill=True,
            fill_color="#FF4B4B",
            fill_opacity=0.9,
            popup=(
                f"<b>⚠️ ANOMALY DETECTED</b><br>"
                f"Score: {row['anomaly_score']:.3f}<br>"
                f"Brightness: {row.get('brightness', 0):.1f} K<br>"
                f"FRP: {row.get('fire_radiative_power', 0):.1f} MW"
            )
        ).add_to(anom_map)

    st_folium(anom_map, width=1200, height=520)

    # -------------------------------------------------------------------------
    # Section 3: ANOMALY SCORE DISTRIBUTION
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Anomaly Score Histogram & Contamination Cutoff")

    fig_hist = px.histogram(
        df_eval,
        x="anomaly_score",
        nbins=50,
        color="is_anomaly",
        color_discrete_map={True: "#FF4B4B", False: "#3B82F6"},
        labels={"anomaly_score": "Composite Anomaly Score", "count": "Observation Count", "is_anomaly": "Is Anomaly"},
        title=f"Anomaly Score Distribution ({results['active_label']})"
    )
    fig_hist.add_vline(
        x=results["threshold"],
        line_dash="dash",
        line_color=COLOR_GOLD,
        annotation_text="Contamination Threshold",
        annotation_position="top right"
    )
    apply_dark_theme(fig_hist)
    fig_hist.update_layout(height=340)
    st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------------------
    # Section 4: ANOMALY TABLE
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Flagged Anomaly Registry")

    anom_display = anom_pts.copy()
    display_cols = ["acquisition_date", "latitude", "longitude", "brightness", "fire_radiative_power", "confidence", "anomaly_score"]
    display_cols = [c for c in display_cols if c in anom_display.columns]

    anom_display = anom_display[display_cols].sort_values(by="anomaly_score", ascending=True)
    anom_display.columns = [c.replace("_", " ").title() for c in anom_display.columns]

    st.dataframe(anom_display.head(200), use_container_width=True, hide_index=True)

    csv_anom = anom_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Anomalies CSV",
        data=csv_anom,
        file_name=f"detected_fire_anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        type="primary"
    )

    # -------------------------------------------------------------------------
    # Section 5: ALGORITHM COMPARISON (If Both Selected)
    # -------------------------------------------------------------------------
    if algo_choice == "Both" and "if_pred" in df_eval.columns and "lof_pred" in df_eval.columns:
        st.markdown("---")
        st.markdown("### ⚖️ Isolation Forest vs Local Outlier Factor Cross-Comparison")

        both_anom = int(((df_eval["if_pred"] == -1) & (df_eval["lof_pred"] == -1)).sum())
        if_only = int(((df_eval["if_pred"] == -1) & (df_eval["lof_pred"] == 1)).sum())
        lof_only = int(((df_eval["if_pred"] == 1) & (df_eval["lof_pred"] == -1)).sum())
        neither = int(((df_eval["if_pred"] == 1) & (df_eval["lof_pred"] == 1)).sum())

        total_anom = both_anom + if_only + lof_only
        agreement_rate = (both_anom / total_anom * 100) if total_anom > 0 else 0.0

        ac1, ac2 = st.columns([1, 2])
        with ac1:
            st.markdown(
                render_metric_card("Ensemble Agreement Rate", f"{agreement_rate:.1f}%", f"{both_anom} mutually flagged anomalies"),
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 14px; line-height: 1.5;">
                    • <b>Isolation Forest Only:</b> {if_only:,} events<br>
                    • <b>LOF Only:</b> {lof_only:,} events<br>
                    • <b>Mutual Consensus:</b> {both_anom:,} events
                </div>
                """,
                unsafe_allow_html=True
            )

        with ac2:
            venn_df = pd.DataFrame({
                "Category": ["Both Algorithms Agree", "Isolation Forest Only", "LOF Only"],
                "Count": [both_anom, if_only, lof_only]
            })
            fig_venn = px.bar(
                venn_df,
                x="Count",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_sequence=[COLOR_ACCENT, COLOR_GOLD, "#00C9A7"],
                title="Consensus Breakdown: Detection Overlap"
            )
            apply_dark_theme(fig_venn)
            fig_venn.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_venn, use_container_width=True)
else:
    st.info("Click '🚨 Run Anomaly Detection' in the sidebar to initiate analysis on the current dataset.")
