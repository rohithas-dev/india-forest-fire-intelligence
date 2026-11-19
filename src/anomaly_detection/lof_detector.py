"""
Local Outlier Factor (LOF) detector for wildfire density-based anomaly detection.

Compares local density of fire observations with that of their neighbors,
identifying clusters of isolated or disproportionately intense fire outbreaks.
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ANOMALY_FEATURE_CANDIDATES: List[str] = [
    "brightness",
    "fire_radiative_power",
    "confidence",
    "latitude",
    "longitude",
]


class LOFDetector:
    """Detects spatial and intensity anomalies using Local Outlier Factor."""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05) -> None:
        """
        Initialize the Local Outlier Factor detector.

        Args:
            n_neighbors: Number of neighbors to consider for density calculation.
            contamination: Expected proportion of outliers in the dataset.
        """
        self.n_neighbors: int = n_neighbors
        self.contamination: float = float(contamination)
        self.model: Optional[LocalOutlierFactor] = None
        self.scaler: StandardScaler = StandardScaler()

    def fit_predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fit LOF and assign anomaly flags and local density scores.

        Args:
            df: DataFrame containing fire detections.

        Returns:
            pd.DataFrame with added columns:
                'lof_score': Negative outlier factor score
                'is_anomaly_lof': 1 for anomaly, 0 for normal
            or None if fewer than 10 rows.
        """
        if df is None or df.empty or len(df) < 10:
            logger.warning(
                f"Insufficient data for LOF: {0 if df is None else len(df)} rows, "
                "need at least 10."
            )
            return None

        out = df.copy()

        # Handle FRP column alias
        if "fire_radiative_power" not in out.columns and "frp" in out.columns:
            out["fire_radiative_power"] = out["frp"]
        elif "fire_radiative_power" in out.columns and "frp" not in out.columns:
            out["frp"] = out["fire_radiative_power"]

        available_features = [c for c in ANOMALY_FEATURE_CANDIDATES if c in out.columns]
        if len(available_features) < 2:
            logger.warning("Fewer than 2 valid features for LOF anomaly detection.")
            return None

        # Impute missing feature values with median
        X = out[available_features].copy()
        for col in available_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            med = float(X[col].median()) if X[col].notna().any() else 0.0
            X[col] = X[col].fillna(med)

        X_scaled = self.scaler.fit_transform(X)

        # Ensure n_neighbors < len(X)
        effective_k = min(self.n_neighbors, max(2, len(X) - 1))

        self.model = LocalOutlierFactor(
            n_neighbors=effective_k,
            contamination=self.contamination,
            novelty=False,
        )
        preds = self.model.fit_predict(X_scaled)  # 1: normal, -1: outlier
        scores = self.model.negative_outlier_factor_

        out["lof_score"] = scores.round(4)
        out["is_anomaly_lof"] = (preds == -1).astype(int)

        n_anomalies = int(out["is_anomaly_lof"].sum())
        logger.info(
            f"LOF completed: detected {n_anomalies} anomalies "
            f"({n_anomalies / len(out) * 100:.1f}%)."
        )
        return out

    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return anomaly counts and percentage for LOF.

        Args:
            df: DataFrame produced by fit_predict().

        Returns:
            Dict containing 'n_anomalies', 'anomaly_rate'.
        """
        if df is None or df.empty or "is_anomaly_lof" not in df.columns:
            return {"n_anomalies": 0, "anomaly_rate": 0.0}

        n_anomalies = int(df["is_anomaly_lof"].sum())
        total = len(df)
        rate = round((n_anomalies / total) * 100.0, 2) if total > 0 else 0.0

        return {
            "n_anomalies": n_anomalies,
            "anomaly_rate": rate,
        }

    @staticmethod
    def compare_with_isolation_forest(
        df_if: pd.DataFrame,
        df_lof: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compute concordance between Isolation Forest and LOF anomaly detections.

        Args:
            df_if: DataFrame containing 'is_anomaly_if' column.
            df_lof: DataFrame containing 'is_anomaly_lof' column.

        Returns:
            Dict with 'agreement_rate', 'both_anomalous', 'if_only', 'lof_only'.
        """
        if df_if is None or df_lof is None:
            return {"agreement_rate": 0.0, "both_anomalous": 0, "if_only": 0, "lof_only": 0}

        if "is_anomaly_if" not in df_if.columns or "is_anomaly_lof" not in df_lof.columns:
            return {"agreement_rate": 0.0, "both_anomalous": 0, "if_only": 0, "lof_only": 0}

        s_if = df_if["is_anomaly_if"].values[:min(len(df_if), len(df_lof))]
        s_lof = df_lof["is_anomaly_lof"].values[:min(len(df_if), len(df_lof))]

        if len(s_if) == 0:
            return {"agreement_rate": 0.0, "both_anomalous": 0, "if_only": 0, "lof_only": 0}

        agree = np.sum(s_if == s_lof)
        both = int(np.sum((s_if == 1) & (s_lof == 1)))
        if_only = int(np.sum((s_if == 1) & (s_lof == 0)))
        lof_only = int(np.sum((s_if == 0) & (s_lof == 1)))
        agreement_rate = round((agree / len(s_if)) * 100.0, 2)

        return {
            "agreement_rate": agreement_rate,
            "both_anomalous": both,
            "if_only": if_only,
            "lof_only": lof_only,
        }
