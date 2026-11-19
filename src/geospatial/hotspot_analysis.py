"""
Geospatial visualization and density-based wildfire hotspot analysis.

Builds Folium interactive maps, density HeatMaps, and spatial aggregation grids.
"""
import logging
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

logger = logging.getLogger(__name__)


class HotspotAnalyzer:
    """Performs spatial binning, density modeling, and Folium map rendering."""

    @staticmethod
    def _get_frp_color(frp: float) -> str:
        """Assign color coding based on Fire Radiative Power (MW)."""
        if frp < 10.0:
            return "#2ecc71"   # Green (< 10 MW)
        elif frp < 50.0:
            return "#f1c40f"   # Yellow (10-50 MW)
        elif frp <= 200.0:
            return "#e67e22"   # Orange (50-200 MW)
        else:
            return "#e74c3c"   # Red (> 200 MW)

    def create_fire_map(
        self,
        df: pd.DataFrame,
        cluster_result: Optional[Dict[str, Any]] = None,
        zoom_start: int = 3,
        max_markers: int = 1500,
    ) -> folium.Map:
        """
        Create a Folium leaflet map of fire observations and cluster centroids.

        Args:
            df: DataFrame containing latitude, longitude, and frp (or fire_radiative_power).
            cluster_result: Optional dict output from KMeansClusterer or DBSCANClusterer.
            zoom_start: Initial zoom level.
            max_markers: Cap on rendered CircleMarkers to prevent browser lag.

        Returns:
            folium.Map object.
        """
        # Default global center if empty
        if df is None or df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            return folium.Map(location=[20.0, 0.0], zoom_start=zoom_start, tiles="CartoDB positron")

        valid_df = df.dropna(subset=["latitude", "longitude"])
        if valid_df.empty:
            return folium.Map(location=[20.0, 0.0], zoom_start=zoom_start, tiles="CartoDB positron")

        center_lat = float(valid_df["latitude"].mean())
        center_lon = float(valid_df["longitude"].mean())

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="CartoDB positron",
        )

        frp_col = (
            "fire_radiative_power"
            if "fire_radiative_power" in valid_df.columns
            else ("frp" if "frp" in valid_df.columns else None)
        )
        bright_col = "brightness" if "brightness" in valid_df.columns else None

        # Sample if too many points to keep interactive performance smooth
        plot_df = valid_df.sample(n=min(len(valid_df), max_markers), random_state=42)

        for _, row in plot_df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            frp = float(row[frp_col]) if frp_col and pd.notna(row[frp_col]) else 0.0
            bright = float(row[bright_col]) if bright_col and pd.notna(row[bright_col]) else 0.0
            color = self._get_frp_color(frp)

            popup_text = (
                f"<b>Fire Detection</b><br>"
                f"Lat: {lat:.3f}, Lon: {lon:.3f}<br>"
                f"FRP: {frp:.1f} MW<br>"
                f"Brightness: {bright:.1f} K"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=4 if frp < 50 else (6 if frp <= 200 else 8),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=1,
                popup=folium.Popup(popup_text, max_width=250),
            ).add_to(m)

        # Overlay cluster centroids if provided
        if cluster_result and "centroids" in cluster_result:
            for idx, (c_lat, c_lon) in enumerate(cluster_result["centroids"]):
                size = cluster_result.get("cluster_sizes", {}).get(idx, "")
                folium.Marker(
                    location=[c_lat, c_lon],
                    icon=folium.Icon(color="darkpurple", icon="fire", prefix="fa"),
                    popup=f"<b>Cluster {idx} Centroid</b><br>Detections: {size}<br>({c_lat:.2f}, {c_lon:.2f})",
                    tooltip=f"Cluster {idx}",
                ).add_to(m)

        return m

    def add_heatmap_layer(
        self,
        folium_map: folium.Map,
        df: pd.DataFrame,
    ) -> folium.Map:
        """
        Add a density HeatMap layer to an existing Folium map.

        Args:
            folium_map: folium.Map to augment.
            df: DataFrame containing latitude, longitude, and frp.

        Returns:
            Updated folium.Map object.
        """
        if df is None or df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            return folium_map

        frp_col = (
            "fire_radiative_power"
            if "fire_radiative_power" in df.columns
            else ("frp" if "frp" in df.columns else None)
        )

        valid_df = df.dropna(subset=["latitude", "longitude"])
        if valid_df.empty:
            return folium_map

        # HeatMap takes list of [lat, lon, weight]
        heat_data = []
        for _, row in valid_df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            frp = float(row[frp_col]) if frp_col and pd.notna(row[frp_col]) else 1.0
            weight = min(max(frp / 20.0, 0.1), 10.0)
            heat_data.append([lat, lon, weight])

        HeatMap(
            heat_data,
            radius=18,
            blur=14,
            max_zoom=10,
            min_opacity=0.3,
        ).add_to(folium_map)

        return folium_map

    def compute_fire_density(
        self,
        df: pd.DataFrame,
        grid_size: float = 1.0,
    ) -> pd.DataFrame:
        """
        Aggregate fire counts and average intensity into spatial grid cells.

        Args:
            df: DataFrame containing latitude, longitude, and FRP.
            grid_size: Cell dimension in degrees.

        Returns:
            pd.DataFrame with 'lat_bin', 'lon_bin', 'fire_count', 'avg_frp'.
        """
        if df is None or df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "fire_count", "avg_frp"])

        data = df.dropna(subset=["latitude", "longitude"]).copy()
        if data.empty:
            return pd.DataFrame(columns=["lat_bin", "lon_bin", "fire_count", "avg_frp"])

        frp_col = (
            "fire_radiative_power"
            if "fire_radiative_power" in data.columns
            else ("frp" if "frp" in data.columns else None)
        )
        if frp_col is None:
            data["frp_val"] = 0.0
            frp_col = "frp_val"
        else:
            data["frp_val"] = pd.to_numeric(data[frp_col], errors="coerce").fillna(0.0)
            frp_col = "frp_val"

        data["lat_bin"] = (data["latitude"] / grid_size).apply(np.floor) * grid_size
        data["lon_bin"] = (data["longitude"] / grid_size).apply(np.floor) * grid_size

        density = (
            data.groupby(["lat_bin", "lon_bin"])
            .agg(
                fire_count=("latitude", "count"),
                avg_frp=(frp_col, "mean"),
            )
            .reset_index()
        )

        density["avg_frp"] = density["avg_frp"].round(2)
        density = density.sort_values(by="fire_count", ascending=False).reset_index(drop=True)
        return density

    def get_top_hotspots(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Retrieve the top N densest spatial wildfire hotspots.

        Args:
            df: Fire events DataFrame.
            n: Number of top cells to return.

        Returns:
            pd.DataFrame of top N dense cells.
        """
        density = self.compute_fire_density(df, grid_size=1.0)
        return density.head(n)

    def regional_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize detections by regional jurisdiction or country.

        Args:
            df: Cleaned fire events DataFrame.

        Returns:
            pd.DataFrame with columns: country, count, avg_brightness, avg_frp, date_range.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["country", "count", "avg_brightness", "avg_frp", "date_range"])

        data = df.copy()
        group_col = "country" if "country" in data.columns and data["country"].notna().any() else None

        if not group_col:
            # Fall back to season or general summary
            data["country"] = "Global / Unspecified"
            group_col = "country"

        bright_col = "brightness" if "brightness" in data.columns else None
        frp_col = (
            "fire_radiative_power"
            if "fire_radiative_power" in data.columns
            else ("frp" if "frp" in data.columns else None)
        )
        date_col = (
            "acquisition_date"
            if "acquisition_date" in data.columns
            else ("timestamp" if "timestamp" in data.columns else None)
        )

        rows = []
        for country_name, grp in data.groupby(group_col):
            cnt = len(grp)
            avg_b = float(grp[bright_col].mean()) if bright_col else 0.0
            avg_f = float(grp[frp_col].mean()) if frp_col else 0.0

            if date_col and grp[date_col].notna().any():
                dates = pd.to_datetime(grp[date_col], errors="coerce").dropna()
                if not dates.empty:
                    d_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                else:
                    d_range = "N/A"
            else:
                d_range = "N/A"

            rows.append({
                "country": str(country_name),
                "count": cnt,
                "avg_brightness": round(avg_b, 2),
                "avg_frp": round(avg_f, 2),
                "date_range": d_range,
            })

        return pd.DataFrame(rows).sort_values(by="count", ascending=False).reset_index(drop=True)
