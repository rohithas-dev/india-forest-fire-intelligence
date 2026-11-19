"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for wildfires.

Uses spherical haversine distance metric to cluster geographic coordinates on the globe
without requiring a pre-specified cluster count, identifying spatial noise and dense fire outbreaks.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from database.models import GeographicalCluster

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM: float = 6371.0


class DBSCANClusterer:
    """Density-based spatial clusterer using geographic haversine distance."""

    def __init__(self, eps_km: float = 50.0, min_samples: int = 5) -> None:
        """
        Initialize DBSCAN clusterer.

        Args:
            eps_km: Neighborhood radius in kilometers.
            min_samples: Minimum number of fire detections required to form a dense core.
        """
        self.eps_km: float = float(eps_km)
        self.min_samples: int = int(min_samples)

    def fit(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Fit DBSCAN using spherical haversine distance.

        Args:
            df: DataFrame containing 'latitude' and 'longitude'.

        Returns:
            Dict containing labels, n_clusters, n_noise, cluster_sizes, centroids, silhouette_score,
            or None if df is empty or insufficient.
        """
        if df is None or df.empty or len(df) < self.min_samples:
            logger.warning(
                f"Cannot fit DBSCAN: DataFrame has {0 if df is None else len(df)} rows, "
                f"need at least {self.min_samples}."
            )
            return None

        coords_deg = df[["latitude", "longitude"]].dropna().values
        if len(coords_deg) < self.min_samples:
            return None

        # Convert degrees to radians for haversine metric: [lat_rad, lon_rad]
        coords_rad = np.radians(coords_deg)
        eps_rad = self.eps_km / EARTH_RADIUS_KM

        db = DBSCAN(
            eps=eps_rad,
            min_samples=self.min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = db.fit_predict(coords_rad)

        unique_labels = set(labels)
        n_noise = int(np.sum(labels == -1))
        cluster_labels = [lbl for lbl in unique_labels if lbl != -1]
        n_clusters = len(cluster_labels)

        # Compute cluster sizes and centroids (mean lat/lon of members)
        cluster_sizes: Dict[int, int] = {}
        centroids: List[Tuple[float, float]] = []

        for cid in sorted(cluster_labels):
            mask = labels == cid
            cluster_sizes[cid] = int(np.sum(mask))
            mean_lat = float(np.mean(coords_deg[mask, 0]))
            mean_lon = float(np.mean(coords_deg[mask, 1]))
            centroids.append((mean_lat, mean_lon))

        # Silhouette score on non-noise points if >= 2 clusters
        sil_score: Optional[float] = None
        if n_clusters >= 2:
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > n_clusters:
                try:
                    sil_score = round(
                        float(silhouette_score(coords_rad[non_noise_mask], labels[non_noise_mask], metric="haversine")),
                        4,
                    )
                except Exception:
                    sil_score = None

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": cluster_sizes,
            "centroids": centroids,
            "silhouette_score": sil_score,
        }

    def get_cluster_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate statistical summary for each cluster found by DBSCAN.

        Args:
            df: DataFrame used in clustering.
            labels: Cluster assignments per row (-1 indicates noise).

        Returns:
            pd.DataFrame with summary statistics per cluster.
        """
        if df is None or df.empty or labels is None or len(df) != len(labels):
            return pd.DataFrame(
                columns=["cluster_id", "centroid_lat", "centroid_lon", "count", "avg_brightness", "avg_frp"]
            )

        data = df.copy()
        data["cluster_id"] = labels

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
        Persist DBSCAN cluster centroids to the geographical_clusters table.

        Args:
            result: Result dictionary returned by fit().
            session: SQLAlchemy database session.

        Returns:
            List of generated cluster_id primary keys.
        """
        if not result or not result.get("centroids"):
            return []

        saved_ids: List[int] = []
        params = json.dumps({
            "algorithm": "dbscan",
            "eps_km": self.eps_km,
            "min_samples": self.min_samples,
            "n_clusters": result.get("n_clusters"),
            "n_noise": result.get("n_noise"),
            "silhouette_score": result.get("silhouette_score"),
        })

        cluster_sizes = result.get("cluster_sizes", {})

        for idx, (lat, lon) in enumerate(result["centroids"]):
            record = GeographicalCluster(
                algorithm="dbscan",
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
        logger.info(f"Saved {len(saved_ids)} DBSCAN cluster centroids to database.")
        return saved_ids
