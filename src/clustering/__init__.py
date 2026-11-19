"""
Geospatial clustering module for identifying wildfire hotspots.

Provides:
    - KMeansClusterer: Centroid-based regional clustering and elbow evaluation
    - DBSCANClusterer: Density-based spatial clustering with haversine distance metric
"""
from src.clustering.kmeans_clustering import KMeansClusterer, find_optimal_k
from src.clustering.dbscan_clustering import DBSCANClusterer

KMeans = KMeansClusterer
DBSCAN = DBSCANClusterer

__all__ = [
    "KMeansClusterer",
    "DBSCANClusterer",
    "KMeans",
    "DBSCAN",
    "find_optimal_k",
]

