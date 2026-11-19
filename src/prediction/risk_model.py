"""
Wildfire risk modeling and multi-class danger classification.

Models include XGBoost, Random Forest, and Gradient Boosting.

IMPORTANT DISCLAIMER:
    This model produces statistical likelihood approximations for decision support
    and educational evaluation. Wildfire propagation is governed by non-linear physical
    dynamics, micro-climates, and terrain characteristics. Predictions are NOT guaranteed
    forecasts and MUST NOT supersede official wildfire agency directives or evacuation orders.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb

from database.models import ModelExperiment

logger = logging.getLogger(__name__)


class FireRiskModel:
    """
    Supervised classifier for wildfire risk categorization.

    Classes:
        0: Low
        1: Moderate
        2: High
        3: Critical
    """

    FEATURE_COLS: List[str] = [
        "brightness",
        "confidence",
        "fire_radiative_power",
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation",
        "month_num",
        "day_of_year",
        "season_encoded",
        "daynight_encoded",
        "latitude",
        "longitude",
    ]

    TARGET_COL: str = "risk_encoded"

    RISK_LABELS: Dict[int, str] = {
        0: "Low",
        1: "Moderate",
        2: "High",
        3: "Critical",
    }

    DISCLAIMER: str = (
        "Model Disclaimer: Model predictions are statistical estimates intended "
        "solely for analysis and situational awareness, not definitive real-world forecasts."
    )

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the risk model.

        Args:
            model_type: 'xgboost' | 'random_forest' | 'gradient_boosting'
            random_state: Random state for deterministic training.
        """
        self.model_type: str = model_type.lower().strip()
        self.random_state: int = random_state
        self.trained_features: List[str] = []

    def _instantiate_model(self):
        """Construct the chosen scikit-learn or XGBoost estimator."""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                random_state=self.random_state,
                eval_metric="mlogloss",
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=120,
                max_depth=10,
                random_state=self.random_state,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.random_state,
            )
        else:
            logger.warning(f"Unknown model_type '{self.model_type}', falling back to xgboost.")
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract predictor features and target vector, imputing missing values.

        Args:
            df: Enriched dataset containing features and target column.

        Returns:
            Tuple of (X, y) as (pd.DataFrame, pd.Series).
        """
        if df is None or df.empty or self.TARGET_COL not in df.columns:
            return pd.DataFrame(), pd.Series(dtype=int)

        # Select available features from canonical list
        avail_features = [c for c in self.FEATURE_COLS if c in df.columns]
        if not avail_features:
            return pd.DataFrame(), pd.Series(dtype=int)

        X = df[avail_features].copy()
        for col in avail_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            med = float(X[col].median()) if X[col].notna().any() else 0.0
            X[col] = X[col].fillna(med)

        y = pd.to_numeric(df[self.TARGET_COL], errors="coerce").fillna(0).astype(int)
        return X, y

    def evaluate(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate classification performance metrics.

        Args:
            y_test: Ground truth labels.
            y_pred: Predicted class labels.
            y_proba: Predicted class probability distribution.

        Returns:
            Dict containing accuracy, precision_score, recall, f1_score, roc_auc.
        """
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        roc_auc = 0.0
        if y_proba is not None:
            try:
                # Multiclass ROC AUC OVR
                unique_classes = np.unique(y_test)
                if len(unique_classes) > 1 and y_proba.shape[1] >= len(unique_classes):
                    roc_auc = float(
                        roc_auc_score(
                            y_test,
                            y_proba,
                            multi_class="ovr",
                            average="weighted",
                            labels=np.arange(y_proba.shape[1]),
                        )
                    )
            except Exception as exc:
                logger.debug(f"ROC-AUC computation skipped: {exc}")
                roc_auc = 0.0

        return {
            "accuracy": round(acc, 4),
            "precision_score": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        }

    def train(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Train risk classifier using an 80/20 chronological split.

        Args:
            df: DataFrame containing fire features and 'risk_encoded'.

        Returns:
            Dict containing fitted model, test data, predictions, metrics, and feature importances,
            or None if data is insufficient.
        """
        if df is None or df.empty or len(df) < 50 or self.TARGET_COL not in df.columns:
            logger.warning(
                f"Cannot train model: need >=50 rows and '{self.TARGET_COL}' column. "
                f"Got {0 if df is None else len(df)} rows."
            )
            return None

        # Sort chronologically if acquisition_date exists
        data = df.copy()
        if "acquisition_date" in data.columns:
            data = data.sort_values(by="acquisition_date").reset_index(drop=True)

        X, y = self.prepare_data(data)
        if X.empty or len(y.unique()) < 2:
            logger.warning("Data lacks required features or multiple target classes.")
            return None

        self.trained_features = list(X.columns)

        # 80/20 Chronological train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = self._instantiate_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = self.evaluate(y_test, y_pred, y_proba)

        # Feature importances
        feature_importance: Dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            for col, imp in zip(self.trained_features, model.feature_importances_):
                feature_importance[col] = round(float(imp), 4)
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
            )

        logger.info(
            f"Trained {self.model_type} model: Acc={metrics['accuracy']} F1={metrics['f1_score']}"
        )

        return {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "model_type": self.model_type,
            "features": self.trained_features,
        }

    def predict(self, model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trained model to new records to predict fire risk categories.

        Args:
            model: Trained classifier.
            X: Input features DataFrame.

        Returns:
            pd.DataFrame with 'risk_encoded', 'risk_level', 'prediction_score'.
        """
        if X is None or X.empty or model is None:
            return pd.DataFrame(columns=["risk_encoded", "risk_level", "prediction_score"])

        # Match columns with model expectations if known
        cols = getattr(model, "feature_names_in_", None) or self.trained_features or list(X.columns)
        X_eval = pd.DataFrame(index=X.index)
        for c in cols:
            if c in X.columns:
                X_eval[c] = pd.to_numeric(X[c], errors="coerce")
                med = float(X_eval[c].median()) if X_eval[c].notna().any() else 0.0
                X_eval[c] = X_eval[c].fillna(med)
            else:
                X_eval[c] = 0.0

        preds = model.predict(X_eval)

        # Confidence / risk score derivation
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)
            # Maximum class probability confidence
            scores = np.max(proba, axis=1)
        else:
            scores = np.ones(len(preds), dtype=float)

        risk_levels = [self.RISK_LABELS.get(int(p), "Moderate") for p in preds]

        out = pd.DataFrame({
            "risk_encoded": preds.astype(int),
            "risk_level": risk_levels,
            "prediction_score": np.round(scores, 4),
        }, index=X.index)

        return out

    def save_experiment_to_db(
        self,
        result: Dict[str, Any],
        session,
        dataset_id: Optional[int] = None,
    ) -> int:
        """
        Record training parameters and evaluation metrics in model_experiments table.

        Args:
            result: Result dictionary returned by train().
            session: SQLAlchemy session.
            dataset_id: Optional Foreign Key to datasets table.

        Returns:
            experiment_id primary key.
        """
        if not result or "metrics" not in result:
            return 0

        metrics = result["metrics"]
        params = json.dumps({
            "model_type": result.get("model_type"),
            "train_size": result.get("train_size"),
            "test_size": result.get("test_size"),
            "features": result.get("features", []),
        })
        feat_imp = json.dumps(result.get("feature_importance", {}))

        experiment = ModelExperiment(
            model_name=result.get("model_type", "unknown").upper(),
            dataset_id=dataset_id,
            accuracy=metrics.get("accuracy"),
            precision_score=metrics.get("precision_score"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            roc_auc=metrics.get("roc_auc"),
            parameters=params,
            feature_importance=feat_imp,
        )
        session.add(experiment)
        session.commit()
        logger.info(f"Saved model experiment {experiment.experiment_id} to database.")
        return experiment.experiment_id

    def compare_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train XGBoost, Random Forest, and Gradient Boosting on identical data splits.

        Args:
            df: DataFrame containing features and risk target.

        Returns:
            Comparison pd.DataFrame with model evaluation metrics.
        """
        models = ["xgboost", "random_forest", "gradient_boosting"]
        rows = []

        for m_name in models:
            m = FireRiskModel(model_type=m_name, random_state=self.random_state)
            res = m.train(df)
            if res and "metrics" in res:
                row = {"model_name": m_name.upper()}
                row.update(res["metrics"])
                rows.append(row)
            else:
                rows.append({
                    "model_name": m_name.upper(),
                    "accuracy": 0.0,
                    "precision_score": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "roc_auc": 0.0,
                })

        return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Standalone helper function to benchmark XGBoost, Random Forest, and Gradient Boosting."""
    return FireRiskModel(random_state=random_state).compare_models(df)

