"""Model evaluation metrics, confusion matrix, ROC curves, and feature importance."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


class ModelEvaluator:
    """Evaluation utilities for ML classification and risk assessment models."""

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """Compute standard classification benchmarks."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": 0.5
        }
        if y_prob is not None:
            try:
                # Handle binary or 2-column probability array
                if len(np.shape(y_prob)) > 1 and np.shape(y_prob)[1] > 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = 0.5
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None):
        """
        Plot styled seaborn confusion matrix formatted for dark theme display.
        Returns matplotlib.figure.Figure.
        """
        if class_names is None:
            class_names = ["Standard / Low", "High Fire Threat"]

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        fig.patch.set_facecolor("#141928")
        ax.set_facecolor("#141928")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold", "color": "#0A0E1A"}
        )

        ax.set_xlabel("Predicted Label", color="#E8EAF0", fontsize=11, labelpad=8)
        ax.set_ylabel("Ground Truth", color="#E8EAF0", fontsize=11, labelpad=8)
        ax.tick_params(colors="#8C9BAE", labelsize=10)
        for _, spine in ax.spines.items():
            spine.set_color("#FF6B35")
            spine.set_linewidth(1.0)

        plt.title("Confusion Matrix", color="#FFFFFF", fontsize=12, pad=12, weight="bold")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], top_n: int = 15):
        """
        Extract feature importance from tree-based estimator and return Plotly figure.
        """
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=True).tail(top_n)

        fig = px.bar(
            df_imp,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance Score (Gain)", "feature": "Predictor Variable"},
            title=f"Top {min(top_n, len(df_imp))} Most Influential Features",
            color_discrete_sequence=["#FF6B35"]
        )
        fig.update_layout(
            paper_bgcolor="#141928",
            plot_bgcolor="#141928",
            font=dict(color="#E8EAF0", family="Inter, sans-serif"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
        )
        return fig
