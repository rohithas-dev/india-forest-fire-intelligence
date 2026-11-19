"""
Wildfire anomaly detection module using unsupervised machine learning.

Provides:
    - IsolationForestDetector: Tree-based isolation anomaly detection
    - LOFDetector: Local density-based outlier factor detection
"""
from src.anomaly_detection.isolation_forest_detector import IsolationForestDetector
from src.anomaly_detection.lof_detector import LOFDetector

IsolationForest = IsolationForestDetector
LocalOutlierFactor = LOFDetector
LOF = LOFDetector

__all__ = [
    "IsolationForestDetector",
    "LOFDetector",
    "IsolationForest",
    "LocalOutlierFactor",
    "LOF",
]

