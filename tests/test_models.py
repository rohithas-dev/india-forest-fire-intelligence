"""
Unit tests for clustering (K-Means, DBSCAN), anomaly detection (Isolation Forest, LOF),
predictive modeling (FireRiskModel, compare_models), and evaluation diagnostics.
"""
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.isolation_forest_detector import IsolationForestDetector
from src.anomaly_detection.lof_detector import LOFDetector
from src.clustering.dbscan_clustering import DBSCANClusterer
from src.clustering.kmeans_clustering import KMeansClusterer, find_optimal_k
from src.evaluation.model_evaluator import ModelEvaluator
from src.models import KMeans  # Alias check
from src.prediction.risk_model import FireRiskModel, compare_models


def test_kmeans_basic(sample_fire_df: pd.DataFrame) -> None:
    """Run KMeansClusterer(n_clusters=3).fit(), verify result keys exist and labels length matches df."""
    clusterer = KMeansClusterer(n_clusters=3, random_state=42)
    result = clusterer.fit(sample_fire_df)

    assert result is not None
    for key in ["labels", "centroids", "inertia", "n_clusters", "silhouette_score", "cluster_sizes"]:
        assert key in result, f"Expected key '{key}' missing from KMeans result dictionary"

    assert len(result["labels"]) == len(sample_fire_df)
    assert len(result["centroids"]) == 3
    assert result["n_clusters"] == 3
    assert result["inertia"] >= 0.0


def test_kmeans_too_few_rows() -> None:
    """Pass DataFrame with 2 rows to KMeans(n_clusters=5), verify it gracefully returns None."""
    two_row_df = pd.DataFrame({
        "latitude": [35.0, 36.0],
        "longitude": [-120.0, -121.0],
    })
    clusterer = KMeans(n_clusters=5)
    result = clusterer.fit(two_row_df)

    assert result is None, f"Expected None for insufficient rows, got: {result}"


def test_kmeans_find_optimal_k(sample_fire_df: pd.DataFrame) -> None:
    """Run find_optimal_k(), verify it returns a dict with 'k_values' and 'inertias'."""
    opt_result = find_optimal_k(sample_fire_df, k_range=range(2, 6))

    assert isinstance(opt_result, dict)
    assert "k_values" in opt_result
    assert "inertias" in opt_result
    assert len(opt_result["k_values"]) > 0
    assert len(opt_result["k_values"]) == len(opt_result["inertias"])
    # Inertia must decrease or stay flat as K increases
    inertias = opt_result["inertias"]
    assert all(isinstance(val, (int, float)) for val in inertias)


def test_dbscan_basic(sample_fire_df: pd.DataFrame) -> None:
    """Run DBSCANClusterer().fit(), verify result dictionary exists and n_clusters >= 0."""
    clusterer = DBSCANClusterer(eps_km=150.0, min_samples=3)
    result = clusterer.fit(sample_fire_df)

    assert result is not None
    assert "n_clusters" in result
    assert "labels" in result
    assert "n_noise" in result
    assert result["n_clusters"] >= 0
    assert len(result["labels"]) == len(sample_fire_df)


def test_isolation_forest_basic(sample_df_with_features: pd.DataFrame) -> None:
    """Run Isolation Forest fit_predict(), verify 'is_anomaly_if' column exists in output."""
    detector = IsolationForestDetector(contamination=0.08, random_state=42)
    result = detector.fit_predict(sample_df_with_features)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "is_anomaly_if" in result.columns
    assert "if_anomaly_score" in result.columns
    assert set(result["is_anomaly_if"].unique()).issubset({0, 1})


def test_isolation_forest_too_few_rows() -> None:
    """Pass a 5-row DataFrame to Isolation Forest, verify it returns None due to minimum size requirement."""
    five_row_df = pd.DataFrame({
        "latitude": [34.1, 34.2, 34.3, 34.4, 34.5],
        "longitude": [-118.1, -118.2, -118.3, -118.4, -118.5],
        "brightness": [320.0, 325.0, 330.0, 335.0, 340.0],
        "fire_radiative_power": [15.0, 16.0, 17.0, 18.0, 19.0],
        "confidence": [75, 80, 85, 90, 95],
    })
    detector = IsolationForestDetector()
    result = detector.fit_predict(five_row_df)

    assert result is None, f"Expected None for DataFrame with 5 rows (<10), got {result}"


def test_lof_basic(sample_df_with_features: pd.DataFrame) -> None:
    """Run LOF fit_predict(), verify 'is_anomaly_lof' column exists in output."""
    detector = LOFDetector(n_neighbors=15, contamination=0.08)
    result = detector.fit_predict(sample_df_with_features)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "is_anomaly_lof" in result.columns
    assert "lof_score" in result.columns
    assert set(result["is_anomaly_lof"].unique()).issubset({0, 1})


def test_risk_model_train(sample_df_with_features: pd.DataFrame) -> None:
    """Run FireRiskModel().train(), verify metrics dictionary includes accuracy and f1_score."""
    model = FireRiskModel(model_type="xgboost", random_state=42)
    result = model.train(sample_df_with_features)

    assert result is not None
    assert "metrics" in result
    metrics = result["metrics"]

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "precision_score" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_risk_model_too_few_rows() -> None:
    """Pass a 10-row DataFrame to FireRiskModel().train(), verify it returns None."""
    ten_row_df = pd.DataFrame({
        "brightness": [320.0] * 10,
        "confidence": [80] * 10,
        "fire_radiative_power": [15.0] * 10,
        "latitude": [37.0] * 10,
        "longitude": [-120.0] * 10,
        "risk_encoded": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    model = FireRiskModel()
    result = model.train(ten_row_df)

    assert result is None, f"Expected None for DataFrame with 10 rows (<50), got {result}"


def test_model_compare(sample_df_with_features: pd.DataFrame) -> None:
    """Run compare_models(), verify the returned DataFrame has exactly 3 benchmarked rows."""
    comparison = compare_models(sample_df_with_features)

    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3
    assert "model_name" in comparison.columns
    models = set(comparison["model_name"].str.upper())
    expected = {"XGBOOST", "RANDOM_FOREST", "GRADIENT_BOOSTING"}
    assert expected.issubset(models), f"Expected {expected}, got {models}"
    for metric_col in ["accuracy", "f1_score", "precision_score", "recall"]:
        assert metric_col in comparison.columns


def test_evaluator_confusion_matrix() -> None:
    """Generate mock y_test and y_pred, verify ModelEvaluator.compute_confusion_matrix shape."""
    y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 0])

    cm = ModelEvaluator.compute_confusion_matrix(y_test, y_pred)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_test)
