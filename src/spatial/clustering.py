"""Spatial clustering algorithms (K-Means & DBSCAN) for wildfire hotspot discovery."""
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    """K-Means spatial clustering for fire observations."""

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.silhouette_ = None

    def fit(self, df: pd.DataFrame) -> "KMeansClusterer":
        """Fit K-Means on latitude and longitude coordinates."""
        if df.empty or len(df) < self.n_clusters:
            return self

        coords = df[["latitude", "longitude"]].dropna().values
        self.model.fit(coords)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = float(self.model.inertia_)

        if len(coords) > self.n_clusters and self.n_clusters > 1:
            try:
                # Sample up to 3000 points for silhouette computation speed
                if len(coords) > 3000:
                    idx = np.random.RandomState(42).choice(len(coords), 3000, replace=False)
                    self.silhouette_ = float(silhouette_score(coords[idx], self.labels_[idx]))
                else:
                    self.silhouette_ = float(silhouette_score(coords, self.labels_))
            except Exception:
                self.silhouette_ = 0.0
        return self

    def compute_elbow_curve(self, df: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
        """Calculate inertia for k from 2 to max_k."""
        coords = df[["latitude", "longitude"]].dropna().values
        if len(coords) < 3:
            return pd.DataFrame({"k": [], "inertia": []})

        max_k = min(max_k, len(coords) - 1)
        results = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            km.fit(coords)
            results.append({"k": k, "inertia": float(km.inertia_)})
        return pd.DataFrame(results)

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return summary dataframe for each K-Means cluster."""
        if self.labels_ is None or self.cluster_centers_ is None:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy["cluster"] = self.labels_
        summary = []
        for i, center in enumerate(self.cluster_centers_):
            c_points = df_copy[df_copy["cluster"] == i]
            summary.append({
                "Cluster ID": f"Cluster {i+1}",
                "Centroid Lat": round(center[0], 4),
                "Centroid Lon": round(center[1], 4),
                "Points Count": len(c_points),
                "Avg Brightness": round(c_points["brightness"].mean(), 1) if "brightness" in c_points.columns else 0.0,
                "Avg FRP (MW)": round(c_points["fire_radiative_power"].mean(), 1) if "fire_radiative_power" in c_points.columns else 0.0
            })
        return pd.DataFrame(summary)


class DBSCANClusterer:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""

    def __init__(self, eps_km: float = 50.0, min_samples: int = 5):
        self.eps_km = eps_km
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0
        self.silhouette_ = None
        self.cluster_summaries_ = []

    def fit(self, df: pd.DataFrame) -> "DBSCANClusterer":
        """Fit DBSCAN using haversine geodesic metric."""
        if df.empty or len(df) < self.min_samples:
            return self

        coords = df[["latitude", "longitude"]].dropna().values
        kms_per_radian = 6371.0088
        eps_rad = self.eps_km / kms_per_radian
        coords_rad = np.radians(coords)

        db = DBSCAN(eps=eps_rad, min_samples=self.min_samples, metric="haversine")
        db.fit(coords_rad)

        self.labels_ = db.labels_
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels - {-1})
        self.n_noise_ = int(np.sum(self.labels_ == -1))

        if self.n_clusters_ > 1:
            try:
                non_noise_mask = (self.labels_ != -1)
                if np.sum(non_noise_mask) > self.n_clusters_:
                    pts = coords_rad[non_noise_mask]
                    lbls = self.labels_[non_noise_mask]
                    if len(pts) > 3000:
                        idx = np.random.RandomState(42).choice(len(pts), 3000, replace=False)
                        self.silhouette_ = float(silhouette_score(pts[idx], lbls[idx], metric="haversine"))
                    else:
                        self.silhouette_ = float(silhouette_score(pts, lbls, metric="haversine"))
            except Exception:
                self.silhouette_ = 0.0

        # Build centroids & summaries
        df_copy = df.copy()
        df_copy["cluster"] = self.labels_
        summaries = []
        for lbl in sorted(list(unique_labels - {-1})):
            c_points = df_copy[df_copy["cluster"] == lbl]
            summaries.append({
                "Cluster ID": f"DBSCAN #{lbl+1}",
                "Centroid Lat": round(c_points["latitude"].mean(), 4),
                "Centroid Lon": round(c_points["longitude"].mean(), 4),
                "Points Count": len(c_points),
                "Avg Brightness": round(c_points["brightness"].mean(), 1) if "brightness" in c_points.columns else 0.0,
                "Avg FRP (MW)": round(c_points["fire_radiative_power"].mean(), 1) if "fire_radiative_power" in c_points.columns else 0.0
            })
        self.cluster_summaries_ = summaries
        return self

    def get_summary(self) -> pd.DataFrame:
        """Return summary of identified DBSCAN clusters."""
        return pd.DataFrame(self.cluster_summaries_)
