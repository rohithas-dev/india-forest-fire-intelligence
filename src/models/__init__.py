"""
Unified Machine Learning and Evaluation Models Package.

Re-exports clustering, anomaly detection, predictive risk models, and evaluators.
"""
from src.clustering.kmeans_clustering import KMeansClusterer, find_optimal_k
from src.clustering.dbscan_clustering import DBSCANClusterer
from src.anomaly_detection.isolation_forest_detector import IsolationForestDetector
from src.anomaly_detection.lof_detector import LOFDetector
from src.prediction.risk_model import FireRiskModel, compare_models
from src.evaluation.model_evaluator import ModelEvaluator

# Common aliases
KMeans = KMeansClusterer
DBSCAN = DBSCANClusterer
IsolationForest = IsolationForestDetector
LocalOutlierFactor = LOFDetector
LOF = LOFDetector

__all__ = [
    "KMeansClusterer",
    "KMeans",
    "find_optimal_k",
    "DBSCANClusterer",
    "DBSCAN",
    "IsolationForestDetector",
    "IsolationForest",
    "LOFDetector",
    "LocalOutlierFactor",
    "LOF",
    "FireRiskModel",
    "compare_models",
    "ModelEvaluator",
]
