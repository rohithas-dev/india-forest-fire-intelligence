"""
Model performance evaluation, metric extraction, and diagnostic plotting.

Generates seaborn confusion matrices, interactive Plotly ROC curves,
and feature importance charts.
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from database.models import ModelExperiment

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = ["Low", "Moderate", "High", "Critical"]


class ModelEvaluator:
    """Evaluation utilities and interactive charts for wildfire classification models."""

    @staticmethod
    def compute_classification_report(
        y_test: Any,
        y_pred: Any,
        labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Generate classification report dictionary from ground truth and predictions.

        Args:
            y_test: True labels.
            y_pred: Predicted labels.
            labels: Optional list of label indices.

        Returns:
            Dictionary matching sklearn's classification_report format.
        """
        try:
            return classification_report(
                y_test,
                y_pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
        except Exception as exc:
            logger.warning(f"Error computing classification report: {exc}")
            return {}

    @staticmethod
    def compute_confusion_matrix(
        y_test: Any,
        y_pred: Any,
    ) -> np.ndarray:
        """
        Compute standard confusion matrix.

        Args:
            y_test: Ground truth.
            y_pred: Predicted labels.

        Returns:
            2D numpy array of counts.
        """
        try:
            return confusion_matrix(y_test, y_pred)
        except Exception as exc:
            logger.warning(f"Error computing confusion matrix: {exc}")
            return np.zeros((2, 2), dtype=int)

    @staticmethod
    def plot_confusion_matrix(
        y_test: Any,
        y_pred: Any,
        class_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Render a seaborn heatmap figure for the confusion matrix.

        Args:
            y_test: Ground truth.
            y_pred: Predictions.
            class_names: Optional display names for classes.

        Returns:
            matplotlib.figure.Figure suitable for Streamlit st.pyplot(fig).
        """
        cm = confusion_matrix(y_test, y_pred)
        classes = class_names or DEFAULT_CLASS_NAMES[: cm.shape[0]]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar=True,
        )
        ax.set_title("Wildfire Risk Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Risk Level", fontsize=11)
        ax.set_ylabel("True Risk Level", fontsize=11)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_importance(
        importance_dict: Dict[str, float],
        top_n: int = 15,
    ) -> go.Figure:
        """
        Create an interactive Plotly horizontal bar chart of top predictor features.

        Args:
            importance_dict: Mapping of feature names to importance scores.
            top_n: Maximum features to show.

        Returns:
            plotly.graph_objects.Figure.
        """
        if not importance_dict:
            fig = go.Figure()
            fig.update_layout(title="No Feature Importance Available")
            return fig

        # Sort top N descending
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in sorted_items][::-1]  # reverse for horizontal display
        scores = [x[1] for x in sorted_items][::-1]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=features,
                orientation="h",
                marker=dict(
                    color=scores,
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
        )
        fig.update_layout(
            title=f"Top {len(features)} Feature Importances",
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            margin=dict(l=140, r=40, t=50, b=40),
            height=450,
            template="plotly_white",
        )
        return fig

    @staticmethod
    def plot_roc_curve(
        y_test: Any,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> go.Figure:
        """
        Construct One-vs-Rest (OvR) interactive ROC curves with Plotly.

        Args:
            y_test: Ground truth target classes.
            y_proba: Model prediction probabilities (N, C).
            class_names: Optional names for each class index.

        Returns:
            plotly.graph_objects.Figure.
        """
        fig = go.Figure()

        if y_proba is None or len(y_proba) == 0:
            fig.update_layout(title="ROC Curve: No Probabilities Provided")
            return fig

        n_classes = y_proba.shape[1]
        classes = class_names or DEFAULT_CLASS_NAMES[:n_classes]
        y_test_arr = np.array(y_test)

        # Binarize labels for multi-class ROC
        if n_classes > 2:
            y_bin = label_binarize(y_test_arr, classes=list(range(n_classes)))
        else:
            y_bin = np.zeros((len(y_test_arr), 2))
            for i, val in enumerate(y_test_arr):
                y_bin[i, int(val)] = 1

        # Add curve for each class
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                cls_label = classes[i] if i < len(classes) else f"Class {i}"
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{cls_label} (AUC = {roc_auc:.3f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )
            except Exception as exc:
                logger.debug(f"Could not compute ROC curve for class {i}: {exc}")

        # Baseline random chance line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance (AUC = 0.50)",
                line=dict(color="grey", dash="dash"),
            )
        )

        fig.update_layout(
            title="Multi-Class ROC Curves (One-vs-Rest)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            template="plotly_white",
            legend=dict(x=0.6, y=0.1),
            height=500,
        )
        return fig

    @staticmethod
    def get_experiment_history(session) -> pd.DataFrame:
        """
        Query historical model training runs from the model_experiments table.

        Args:
            session: SQLAlchemy database session.

        Returns:
            pd.DataFrame sorted by created_at descending.
        """
        try:
            records = (
                session.query(ModelExperiment)
                .order_by(ModelExperiment.created_at.desc())
                .all()
            )
            rows = []
            for r in records:
                rows.append({
                    "experiment_id": r.experiment_id,
                    "model_name": r.model_name,
                    "dataset_id": r.dataset_id,
                    "accuracy": r.accuracy,
                    "precision": r.precision_score,
                    "recall": r.recall,
                    "f1_score": r.f1_score,
                    "roc_auc": r.roc_auc,
                    "created_at": r.created_at,
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning(f"Error fetching experiment history: {exc}")
            return pd.DataFrame()
