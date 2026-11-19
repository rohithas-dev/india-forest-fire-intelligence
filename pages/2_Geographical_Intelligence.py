"""
Geographical Intelligence page — Spatial Clustering (K-Means & DBSCAN), Folium Maps & Hotspots.
"""
from datetime import datetime, date
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from database.db import get_db_session
from database.models import FireEvent, Dataset, GeographicalCluster
from database.schema import initialize_schema
from src.utils.ui import (
    inject_css,
    apply_dark_theme,
    get_all_datasets,
    load_fire_events_df,
    COLOR_ACCENT,
    COLOR_GOLD,
    COLOR_SUCCESS
)
from src.spatial.clustering import KMeansClusterer, DBSCANClusterer
from src.spatial.hotspots import HotspotAnalyzer

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🗺️ Geographical Intelligence — Forest Fire Intelligence",
    layout="wide",
    page_icon="🗺️"
)
initialize_schema()
inject_css()

st.title("🗺️ Geographical Intelligence & Spatial Clustering")
st.caption("Identify wildfire concentration epicenters and dense thermal hotspots using geospatial clustering algorithms.")

# -----------------------------------------------------------------------------
# Section 1: SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🗺️ Clustering Parameters")

datasets = get_all_datasets()
dataset_options = {"All Datasets": None}
for d in datasets:
    dataset_options[f"#{d['dataset_id']} - {d['dataset_name']} ({d['record_count']:,})"] = d["dataset_id"]

selected_ds_label = st.sidebar.selectbox("Target Dataset", options=list(dataset_options.keys()), index=0)
selected_ds_id = dataset_options[selected_ds_label]

algo_choice = st.sidebar.selectbox(
    "Clustering Algorithm",
    options=["K-Means", "DBSCAN", "Both"],
    index=2,
    help="K-Means partitions into k centroids. DBSCAN discovers arbitrary-density clusters and rejects isolated noise points."
)

kmeans_k = 6
dbscan_eps = 50.0
dbscan_min_samples = 5

if algo_choice in ["K-Means", "Both"]:
    kmeans_k = st.sidebar.slider("K-Means: Number of Clusters (k)", min_value=2, max_value=15, value=6, step=1)

if algo_choice in ["DBSCAN", "Both"]:
    dbscan_eps = st.sidebar.slider("DBSCAN: Epsilon (km radius)", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
    dbscan_min_samples = st.sidebar.slider("DBSCAN: Min Samples", min_value=3, max_value=20, value=5, step=1)

# Country Filter
country_options = ["All"]
try:
    with get_db_session() as session:
        c_rows = session.query(FireEvent.country).distinct().all()
        for r in c_rows:
            if r[0] and r[0] not in country_options:
                country_options.append(r[0])
except Exception:
    pass

selected_country = st.sidebar.selectbox("Country Filter", options=country_options, index=0)
show_heatmap_toggle = st.sidebar.checkbox("Overlay Heatmap Layer", value=True)
run_clustering_btn = st.sidebar.button("🚀 Run Spatial Analysis", type="primary")

# -----------------------------------------------------------------------------
# Data Loading & Filtering
# -----------------------------------------------------------------------------
raw_fire_df = load_fire_events_df(dataset_id=selected_ds_id, limit=8000)

if not raw_fire_df.empty:
    filtered_df = raw_fire_df.copy()
    if selected_country != "All" and "country" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["country"] == selected_country]
else:
    filtered_df = pd.DataFrame()

# -----------------------------------------------------------------------------
# Clustering Execution & DB Persistence
# -----------------------------------------------------------------------------
kmeans_res = None
dbscan_res = None

if "kmeans_model" not in st.session_state:
    st.session_state["kmeans_model"] = None
if "dbscan_model" not in st.session_state:
    st.session_state["dbscan_model"] = None

if run_clustering_btn:
    if filtered_df.empty or len(filtered_df) < 5:
        st.error("Insufficient fire observations in database to run spatial clustering.")
    else:
        with st.spinner("Executing spatial clustering algorithms..."):
            centroids_to_save = []

            if algo_choice in ["K-Means", "Both"]:
                km = KMeansClusterer(n_clusters=kmeans_k)
                km.fit(filtered_df)
                st.session_state["kmeans_model"] = km
                if km.cluster_centers_ is not None:
                    summary_km = km.get_summary(filtered_df)
                    for _, row in summary_km.iterrows():
                        centroids_to_save.append(GeographicalCluster(
                            algorithm="kmeans",
                            centroid_latitude=float(row["Centroid Lat"]),
                            centroid_longitude=float(row["Centroid Lon"]),
                            observation_count=int(row["Points Count"]),
                            n_clusters=kmeans_k,
                            parameters_json=json.dumps({"n_clusters": kmeans_k, "inertia": km.inertia_}),
                            created_at=datetime.utcnow()
                        ))

            if algo_choice in ["DBSCAN", "Both"]:
                db = DBSCANClusterer(eps_km=dbscan_eps, min_samples=dbscan_min_samples)
                db.fit(filtered_df)
                st.session_state["dbscan_model"] = db
                if db.cluster_summaries_:
                    for c_info in db.cluster_summaries_:
                        centroids_to_save.append(GeographicalCluster(
                            algorithm="dbscan",
                            centroid_latitude=float(c_info["Centroid Lat"]),
                            centroid_longitude=float(c_info["Centroid Lon"]),
                            observation_count=int(c_info["Points Count"]),
                            n_clusters=db.n_clusters_,
                            parameters_json=json.dumps({"eps_km": dbscan_eps, "min_samples": dbscan_min_samples}),
                            created_at=datetime.utcnow()
                        ))

            if centroids_to_save:
                try:
                    with get_db_session() as session:
                        session.bulk_save_objects(centroids_to_save)
                except Exception as ex:
                    pass

        st.success("✅ Spatial clustering completed and centroids saved to database.")

kmeans_res = st.session_state.get("kmeans_model")
dbscan_res = st.session_state.get("dbscan_model")

# -----------------------------------------------------------------------------
# Section 2: MAP VIEW
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ Wildfire Thermal Distribution & Cluster Map")

if filtered_df.empty:
    st.info("No fire event records loaded. Import a dataset to generate geospatial maps.")
else:
    # Prepare centroids for map rendering
    active_centroids = []
    if kmeans_res is not None and kmeans_res.cluster_centers_ is not None:
        for center in kmeans_res.cluster_centers_:
            active_centroids.append({
                "centroid_latitude": center[0],
                "centroid_longitude": center[1],
                "algorithm": "K-Means",
                "observation_count": 0
            })
    if dbscan_res is not None and dbscan_res.cluster_summaries_:
        for c_item in dbscan_res.cluster_summaries_:
            active_centroids.append({
                "centroid_latitude": c_item["Centroid Lat"],
                "centroid_longitude": c_item["Centroid Lon"],
                "algorithm": "DBSCAN",
                "observation_count": c_item["Points Count"]
            })

    fire_map = HotspotAnalyzer.create_fire_map(
        filtered_df,
        show_heatmap=show_heatmap_toggle,
        cluster_centroids=active_centroids,
        max_points=2500
    )
    st_folium(fire_map, width=1200, height=550)

# -----------------------------------------------------------------------------
# Section 3: CLUSTERING RESULTS (2 Columns)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔬 Spatial Clustering Detailed Results")

c_col1, c_col2 = st.columns(2)

with c_col1:
    st.markdown("#### K-Means Partitioning")
    if kmeans_res is not None and kmeans_res.labels_ is not None:
        km_summary = kmeans_res.get_summary(filtered_df)
        st.markdown(f"**Discovered Centroids:** `{kmeans_k}` | **Inertia:** `{kmeans_res.inertia_:.1f}`")
        if kmeans_res.silhouette_ is not None:
            st.markdown(f"**Silhouette Score:** `{kmeans_res.silhouette_:.3f}` *(Higher indicates well-separated clusters)*")
        st.dataframe(km_summary, use_container_width=True, hide_index=True)

        # Elbow curve
        if len(filtered_df) >= 10:
            with st.expander("📊 View K-Means Inertia / Elbow Optimization Curve", expanded=False):
                elbow_df = kmeans_res.compute_elbow_curve(filtered_df, max_k=10)
                if not elbow_df.empty:
                    fig_elbow = px.line(
                        elbow_df,
                        x="k",
                        y="inertia",
                        markers=True,
                        title="Elbow Method for Optimal k Centroids",
                        labels={"k": "Number of Clusters (k)", "inertia": "Sum of Squared Distances (Inertia)"}
                    )
                    fig_elbow.update_traces(line=dict(color=COLOR_ACCENT, width=3), marker=dict(color=COLOR_GOLD, size=8))
                    apply_dark_theme(fig_elbow)
                    fig_elbow.update_layout(height=300)
                    st.plotly_chart(fig_elbow, use_container_width=True)
    else:
        st.info("Run analysis from the sidebar to inspect K-Means cluster summaries.")

with c_col2:
    st.markdown("#### DBSCAN Density Clustering")
    if dbscan_res is not None and dbscan_res.labels_ is not None:
        db_summary = dbscan_res.get_summary()
        st.markdown(f"**Clusters Formed:** `{dbscan_res.n_clusters_}` | **Isolated Noise Points:** `{dbscan_res.n_noise_:,}`")
        if dbscan_res.silhouette_ is not None:
            st.markdown(f"**Geodesic Silhouette:** `{dbscan_res.silhouette_:.3f}`")
        if not db_summary.empty:
            st.dataframe(db_summary, use_container_width=True, hide_index=True)
        else:
            st.warning("DBSCAN identified all points as ambient noise under current parameters. Try increasing Epsilon radius.")
    else:
        st.info("Run analysis from the sidebar to inspect DBSCAN density-based clusters.")

# -----------------------------------------------------------------------------
# Section 4: HOTSPOT ANALYSIS
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔥 Top Wildfire Hotspot Epicenters")

if not filtered_df.empty:
    hotspot_col1, hotspot_col2 = st.columns([3, 2])

    with hotspot_col1:
        st.markdown("**Top 10 High-Density Regional Thermal Hotspots**")
        top_hotspots_df = HotspotAnalyzer.get_top_hotspots(filtered_df, n=10)
        st.dataframe(top_hotspots_df, use_container_width=True, hide_index=True)

    with hotspot_col2:
        st.markdown("**Regional Aggregation Summary**")
        if "country" in filtered_df.columns:
            region_summary = filtered_df.groupby("country").agg(
                total_fires=("latitude", "count"),
                avg_frp=("fire_radiative_power", "mean") if "fire_radiative_power" in filtered_df.columns else ("latitude", "count")
            ).reset_index().sort_values(by="total_fires", ascending=False)
            region_summary.columns = ["Territory", "Total Fires", "Avg FRP (MW)"]
            region_summary["Avg FRP (MW)"] = region_summary["Avg FRP (MW)"].round(1)
            st.dataframe(region_summary, use_container_width=True, hide_index=True)
        else:
            st.write("Regional attribute not available in current observations.")
