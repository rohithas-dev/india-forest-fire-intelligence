"""Hotspot analysis and geospatial interactive mapping."""
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


class HotspotAnalyzer:
    """Geospatial hotspot discovery and map visualization."""

    @staticmethod
    def get_top_hotspots(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Aggregate fire observations into regional grid cells to identify top fire hotspots.
        """
        if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            return pd.DataFrame()

        df_c = df.copy()
        # Bin to 0.5 degree grid (~50km)
        df_c["grid_lat"] = df_c["latitude"].round(1)
        df_c["grid_lon"] = df_c["longitude"].round(1)

        grouped = df_c.groupby(["grid_lat", "grid_lon"]).agg(
            observation_count=("latitude", "count"),
            avg_brightness=("brightness", "mean") if "brightness" in df_c.columns else ("latitude", "count"),
            max_frp=("fire_radiative_power", "max") if "fire_radiative_power" in df_c.columns else ("latitude", "count"),
            country=("country", "first") if "country" in df_c.columns else ("latitude", "first")
        ).reset_index()

        grouped = grouped.sort_values(by="observation_count", ascending=False).head(n)
        grouped.columns = ["Latitude Grid", "Longitude Grid", "Observation Count", "Avg Brightness (K)", "Max FRP (MW)", "Country"]
        grouped["Avg Brightness (K)"] = grouped["Avg Brightness (K)"].round(1)
        grouped["Max FRP (MW)"] = grouped["Max FRP (MW)"].round(1)
        return grouped

    @staticmethod
    def create_fire_map(
        df: pd.DataFrame,
        show_heatmap: bool = True,
        cluster_centroids: Optional[List[Dict[str, Any]]] = None,
        anomalies: Optional[pd.DataFrame] = None,
        max_points: int = 2500
    ) -> folium.Map:
        """
        Build a high-performance Folium map with dark styling, heatmaps, and markers.
        """
        if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            # Default empty world map centered on Mediterranean/California
            return folium.Map(location=[35.0, -10.0], zoom_start=3, tiles="CartoDB dark_matter")

        center_lat = float(df["latitude"].median())
        center_lon = float(df["longitude"].median())
        if pd.isna(center_lat) or pd.isna(center_lon):
            center_lat, center_lon = 37.5, -119.5

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles="CartoDB dark_matter"
        )

        sample_df = df.sample(min(len(df), max_points), random_state=42) if len(df) > max_points else df

        # Optional HeatMap Layer
        if show_heatmap and len(sample_df) > 0:
            heat_data = []
            for _, row in sample_df.iterrows():
                intensity = float(row.get("fire_radiative_power", 1.0))
                intensity = min(max(intensity, 1.0), 100.0) / 100.0
                heat_data.append([row["latitude"], row["longitude"], intensity])
            HeatMap(
                heat_data,
                radius=14,
                blur=12,
                max_zoom=10,
                gradient={0.2: "#FFEB3B", 0.5: "#FF9800", 0.8: "#FF5722", 1.0: "#D50000"}
            ).add_to(m)

        # Cluster Centroids overlay
        if cluster_centroids:
            for c in cluster_centroids:
                lat = c.get("centroid_latitude") or c.get("lat")
                lon = c.get("centroid_longitude") or c.get("lon")
                algo = c.get("algorithm", "Cluster")
                cnt = c.get("observation_count", c.get("points", 0))
                if lat and lon:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=12,
                        popup=f"<b>{algo.upper()} Centroid</b><br>Observations: {cnt:,}<br>Lat: {lat:.3f}, Lon: {lon:.3f}",
                        color="#00C9A7",
                        fill=True,
                        fill_color="#00C9A7",
                        fill_opacity=0.85
                    ).add_to(m)

        # Anomalies overlay
        if anomalies is not None and not anomalies.empty:
            for _, anom in anomalies.head(200).iterrows():
                folium.CircleMarker(
                    location=[anom["latitude"], anom["longitude"]],
                    radius=8,
                    popup=f"<b>ANOMALY DETECTED</b><br>Score: {anom.get('anomaly_score', 0):.3f}<br>FRP: {anom.get('frp', 0):.1f} MW",
                    color="#FF4B4B",
                    fill=True,
                    fill_color="#FF4B4B",
                    fill_opacity=0.9
                ).add_to(m)

        return m
