"""
K-Means spatial clustering algorithm for wildfire event grouping.

Identifies regional fire density centers and calculates optimal cluster counts
via elbow analysis and silhouette metrics.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from database.models import GeographicalCluster

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """Partitions wildfire events into K spatial clusters using coordinates."""

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        """
        Initialize KMeansClusterer.

        Args:
            n_clusters: Number of clusters (centroids) to form.
            random_state: Seed for reproducibility.
        """
        self.n_clusters: int = max(2, n_clusters)
        self.random_state: int = random_state

    def fit(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Fit K-Means clustering on geographic coordinates ('latitude', 'longitude').

        Args:
            df: DataFrame containing 'latitude' and 'longitude'.

        Returns:
            Dict containing labels, centroids, inertia, silhouette_score, and cluster_sizes,
            or None if insufficient data.
        """
        if df is None or df.empty or len(df) < self.n_clusters:
            logger.warning(
                f"Cannot fit K-Means: DataFrame has {0 if df is None else len(df)} rows, "
                f"need at least {self.n_clusters}."
            )
            return None

        coords = df[["latitude", "longitude"]].dropna().values
        if len(coords) < self.n_clusters:
            return None

        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(coords_scaled)

        # Invert scaling to obtain true latitude and longitude coordinates
        centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids: List[Tuple[float, float]] = [
            (float(c[0]), float(c[1])) for c in centroids_unscaled
        ]

        # Calculate silhouette score (requires at least 2 distinct clusters)
        try:
            sil_score = float(silhouette_score(coords_scaled, labels))
        except Exception:
            sil_score = 0.0

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}

        return {
            "labels": labels,
            "centroids": centroids,
            "inertia": float(kmeans.inertia_),
            "n_clusters": self.n_clusters,
            "silhouette_score": round(sil_score, 4),
            "cluster_sizes": cluster_sizes,
        }

    def find_optimal_k(
        self,
        df: pd.DataFrame,
        k_range: range = range(2, 11),
    ) -> Dict[str, List[Any]]:
        """
        Compute inertia and silhouette scores over a range of K for elbow curve analysis.

        Args:
            df: DataFrame containing 'latitude' and 'longitude'.
            k_range: Range of cluster counts to test.

        Returns:
            Dict with 'k_values', 'inertias', and 'silhouette_scores'.
        """
        result = {"k_values": [], "inertias": [], "silhouette_scores": []}

        if df is None or df.empty or len(df) < max(k_range):
            logger.warning("Insufficient data for optimal K search.")
            return result

        coords = df[["latitude", "longitude"]].dropna().values
        if len(coords) < 3:
            return result

        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        for k in k_range:
            if k >= len(coords_scaled):
                continue
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(coords_scaled)
            result["k_values"].append(k)
            result["inertias"].append(float(km.inertia_))
            try:
                sil = float(silhouette_score(coords_scaled, labels))
            except Exception:
                sil = 0.0
            result["silhouette_scores"].append(round(sil, 4))

        return result

    def get_cluster_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate statistical summary for each cluster.

        Args:
            df: DataFrame used in clustering.
            labels: Cluster assignments per row.

        Returns:
            pd.DataFrame with columns:
                cluster_id, centroid_lat, centroid_lon, count, avg_brightness, avg_frp
        """
        if df is None or df.empty or labels is None or len(df) != len(labels):
            return pd.DataFrame(
                columns=["cluster_id", "centroid_lat", "centroid_lon", "count", "avg_brightness", "avg_frp"]
            )

        data = df.copy()
        data["cluster_id"] = labels

        # Ensure frp and brightness columns exist
        bright_col = "brightness" if "brightness" in data.columns else None
        frp_col = (
            "fire_radiative_power"
            if "fire_radiative_power" in data.columns
            else ("frp" if "frp" in data.columns else None)
        )

        rows = []
        for cid, group in data.groupby("cluster_id"):
            centroid_lat = float(group["latitude"].mean())
            centroid_lon = float(group["longitude"].mean())
            cnt = len(group)
            avg_b = float(group[bright_col].mean()) if bright_col else 0.0
            avg_frp = float(group[frp_col].mean()) if frp_col else 0.0

            rows.append({
                "cluster_id": int(cid),
                "centroid_lat": round(centroid_lat, 4),
                "centroid_lon": round(centroid_lon, 4),
                "count": cnt,
                "avg_brightness": round(avg_b, 2),
                "avg_frp": round(avg_frp, 2),
            })

        return pd.DataFrame(rows)

    def save_results_to_db(self, result: Dict[str, Any], session) -> List[int]:
        """
        Persist cluster centroids to the geographical_clusters database table.

        Args:
            result: Result dictionary returned by fit().
            session: SQLAlchemy database session.

        Returns:
            List of generated cluster_id primary keys.
        """
        if not result or "centroids" not in result:
            return []

        saved_ids: List[int] = []
        params = json.dumps({
            "algorithm": "kmeans",
            "n_clusters": result.get("n_clusters"),
            "inertia": result.get("inertia"),
            "silhouette_score": result.get("silhouette_score"),
        })

        cluster_sizes = result.get("cluster_sizes", {})

        for idx, (lat, lon) in enumerate(result["centroids"]):
            record = GeographicalCluster(
                algorithm="kmeans",
                centroid_latitude=lat,
                centroid_longitude=lon,
                observation_count=cluster_sizes.get(idx, 0),
                n_clusters=result.get("n_clusters"),
                parameters_json=params,
            )
            session.add(record)
            session.flush()
            saved_ids.append(record.cluster_id)

        session.commit()
        logger.info(f"Saved {len(saved_ids)} KMeans cluster centroids to database.")
        return saved_ids


def find_optimal_k(
    df: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[str, List[Any]]:
    """Standalone helper function to compute elbow curve metrics."""
    return KMeansClusterer(random_state=random_state).find_optimal_k(df, k_range=k_range)

