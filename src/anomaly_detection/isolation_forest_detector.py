"""
Isolation Forest detector for extreme wildfire intensity and spatial outliers.

Detects abnormally intense fire radiative power, anomalous brightness temperatures,
or detections occurring in unexpected geographic locations.
"""
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from database.models import AnomalyRecord

logger = logging.getLogger(__name__)

ANOMALY_FEATURE_CANDIDATES: List[str] = [
    "brightness",
    "fire_radiative_power",
    "confidence",
    "latitude",
    "longitude",
]


class IsolationForestDetector:
    """Unsupervised anomaly detection using tree isolation partitioning."""

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> None:
        """
        Initialize the Isolation Forest detector.

        Args:
            contamination: Expected proportion of outliers in the data.
            random_state: Seed for reproducible random splits.
            n_estimators: Number of isolation trees to construct.
        """
        self.contamination: float = float(contamination)
        self.random_state: int = random_state
        self.n_estimators: int = n_estimators
        self.model: Optional[IsolationForest] = None
        self.scaler: StandardScaler = StandardScaler()

    def fit_predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fit Isolation Forest and predict anomaly flags and scores.

        Features used:
            - brightness
            - fire_radiative_power (or frp)
            - confidence
            - latitude
            - longitude

        Args:
            df: DataFrame containing fire detections.

        Returns:
            pd.DataFrame with added columns:
                'if_anomaly_score': Decision function score (negative indicates anomaly)
                'is_anomaly_if': 1 for anomaly, 0 for normal
            or None if fewer than 10 rows.
        """
        if df is None or df.empty or len(df) < 10:
            logger.warning(
                f"Insufficient data for Isolation Forest: {0 if df is None else len(df)} rows, "
                "need at least 10."
            )
            return None

        out = df.copy()

        # Resolve frp / fire_radiative_power alias
        if "fire_radiative_power" not in out.columns and "frp" in out.columns:
            out["fire_radiative_power"] = out["frp"]
        elif "fire_radiative_power" in out.columns and "frp" not in out.columns:
            out["frp"] = out["fire_radiative_power"]

        available_features = [c for c in ANOMALY_FEATURE_CANDIDATES if c in out.columns]
        if len(available_features) < 2:
            logger.warning("Fewer than 2 valid features available for anomaly detection.")
            return None

        # Impute missing feature values with median
        X = out[available_features].copy()
        for col in available_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            med = float(X[col].median()) if X[col].notna().any() else 0.0
            X[col] = X[col].fillna(med)

        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        preds = self.model.fit_predict(X_scaled)  # 1: normal, -1: anomaly
        raw_scores = self.model.decision_function(X_scaled)

        out["if_anomaly_score"] = raw_scores.round(4)
        out["is_anomaly_if"] = (preds == -1).astype(int)

        n_anomalies = int(out["is_anomaly_if"].sum())
        logger.info(
            f"Isolation Forest completed: detected {n_anomalies} anomalies "
            f"({n_anomalies / len(out) * 100:.1f}%)."
        )
        return out

    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary metrics from Isolation Forest output.

        Args:
            df: DataFrame produced by fit_predict().

        Returns:
            Dict containing 'n_anomalies', 'anomaly_rate', 'mean_score'.
        """
        if df is None or df.empty or "is_anomaly_if" not in df.columns:
            return {"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": 0.0}

        n_anomalies = int(df["is_anomaly_if"].sum())
        total = len(df)
        rate = round((n_anomalies / total) * 100.0, 2) if total > 0 else 0.0
        mean_score = (
            round(float(df["if_anomaly_score"].mean()), 4)
            if "if_anomaly_score" in df.columns
            else 0.0
        )

        return {
            "n_anomalies": n_anomalies,
            "anomaly_rate": rate,
            "mean_score": mean_score,
        }

    def save_anomalies_to_db(self, df: pd.DataFrame, session) -> int:
        """
        Persist identified anomalies (is_anomaly_if == 1) to the anomaly_records table.

        Args:
            df: DataFrame containing detection results.
            session: Active SQLAlchemy session.

        Returns:
            Number of anomaly records saved.
        """
        if df is None or df.empty or "is_anomaly_if" not in df.columns:
            return 0

        anomalies = df[df["is_anomaly_if"] == 1]
        if anomalies.empty:
            return 0

        now = datetime.utcnow()
        saved_count = 0

        for _, row in anomalies.iterrows():
            event_id = None
            if "event_id" in row and pd.notna(row["event_id"]):
                try:
                    event_id = int(row["event_id"])
                except Exception:
                    event_id = None

            frp_val = row.get("fire_radiative_power")
            if pd.isna(frp_val):
                frp_val = row.get("frp", 0.0)

            record = AnomalyRecord(
                event_id=event_id,
                anomaly_score=float(row.get("if_anomaly_score", 0.0)),
                anomaly_type="isolation_forest",
                latitude=float(row["latitude"]) if "latitude" in row and pd.notna(row["latitude"]) else None,
                longitude=float(row["longitude"]) if "longitude" in row and pd.notna(row["longitude"]) else None,
                brightness=float(row["brightness"]) if "brightness" in row and pd.notna(row["brightness"]) else None,
                frp=float(frp_val) if pd.notna(frp_val) else 0.0,
                detected_at=now,
            )
            session.add(record)
            saved_count += 1

        session.commit()
        logger.info(f"Saved {saved_count} Isolation Forest anomaly records to database.")
        return saved_count
