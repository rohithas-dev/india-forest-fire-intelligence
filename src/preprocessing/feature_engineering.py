"""
Feature engineering pipeline for wildfire detection and risk modeling.

Extracts temporal, spatial, meteorological, and satellite sensor features
and constructs composite multi-factor fire risk indices.
"""
import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEASON_ENCODING = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "winter": 0,
    "spring": 1,
    "summer": 2,
    "fall": 3,
}

ML_FEATURE_COLUMNS: List[str] = [
    "brightness",
    "confidence",
    "fire_radiative_power",
    "month_num",
    "day_of_year",
    "season_encoded",
    "daynight_encoded",
    "is_high_confidence",
    "latitude",
    "longitude",
    "lat_grid",
    "lon_grid",
]


class FeatureEngineer:
    """Computes spatial, temporal, and risk features from fire events and weather."""

    def engineer_fire_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive enriched temporal, intensity, and spatial features from fire observations.

        Features added:
            - day_of_year: Integer (1-366)
            - month_num: Integer (1-12)
            - year: Integer (e.g. 2023)
            - season_encoded: Winter=0, Spring=1, Summer=2, Fall=3
            - daynight_encoded: 1 for Day ('D'), 0 for Night ('N')
            - is_high_confidence: 1 if confidence >= 80 else 0
            - intensity_category: 'Low' (<10 MW), 'Moderate' (10-50), 'High' (50-200), 'Extreme' (>200)
            - lat_grid, lon_grid: 0.1° rounded spatial coordinates for spatial binning

        Args:
            df: Cleaned fire events DataFrame.

        Returns:
            pd.DataFrame with added features.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to engineer_fire_features.")
            return pd.DataFrame()

        out = df.copy()

        # Date parsing
        date_series = pd.to_datetime(out.get("acquisition_date", pd.Series()), errors="coerce")
        out["day_of_year"] = date_series.dt.dayofyear.fillna(180).astype(int)
        out["month_num"] = date_series.dt.month.fillna(7).astype(int)
        out["year"] = date_series.dt.year.fillna(2023).astype(int)

        # Season encoding
        season_col = out.get("season", pd.Series(["Summer"] * len(out)))
        out["season_encoded"] = season_col.map(SEASON_ENCODING).fillna(2).astype(int)

        # Day/Night encoding
        if "daynight" in out.columns:
            out["daynight_encoded"] = (
                out["daynight"].astype(str).str.upper().str.strip().map(lambda x: 1 if x == "D" else 0)
            )
        else:
            out["daynight_encoded"] = 1

        # Confidence thresholding
        if "confidence" in out.columns:
            out["is_high_confidence"] = (pd.to_numeric(out["confidence"], errors="coerce") >= 80).astype(int)
        else:
            out["is_high_confidence"] = 0

        # Fire intensity classification based on Fire Radiative Power (MW)
        frp_col = (
            out["fire_radiative_power"]
            if "fire_radiative_power" in out.columns
            else out.get("frp", pd.Series([0.0] * len(out)))
        )
        frp_numeric = pd.to_numeric(frp_col, errors="coerce").fillna(0.0)

        def categorize_intensity(frp: float) -> str:
            if frp < 10.0:
                return "Low"
            elif frp < 50.0:
                return "Moderate"
            elif frp <= 200.0:
                return "High"
            return "Extreme"

        out["intensity_category"] = frp_numeric.map(categorize_intensity)

        # Spatial 0.1° binning
        if "latitude" in out.columns and "longitude" in out.columns:
            out["lat_grid"] = pd.to_numeric(out["latitude"], errors="coerce").round(1)
            out["lon_grid"] = pd.to_numeric(out["longitude"], errors="coerce").round(1)
        else:
            out["lat_grid"] = 0.0
            out["lon_grid"] = 0.0

        logger.info(f"Engineered features for {len(out)} fire events.")
        return out

    @staticmethod
    def get_feature_columns() -> List[str]:
        """
        Return the list of standard numeric feature column names for ML models.

        Returns:
            List of column names.
        """
        return list(ML_FEATURE_COLUMNS)

    def create_fire_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct multi-factor composite fire risk targets from satellite observations.

        Computes:
            risk_score = 0.4 * norm(brightness) + 0.4 * norm(frp) + 0.2 * norm(confidence)
            risk_level: 'Low' (<0.25), 'Moderate' (0.25-0.50), 'High' (0.50-0.75), 'Critical' (>=0.75)
            risk_encoded: Low=0, Moderate=1, High=2, Critical=3

        Args:
            df: DataFrame containing brightness, frp/fire_radiative_power, and confidence.

        Returns:
            pd.DataFrame with 'risk_score', 'risk_level', and 'risk_encoded' added.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to create_fire_risk_labels.")
            return pd.DataFrame()

        out = df.copy()

        # Extract & clamp components
        b_series = pd.to_numeric(out.get("brightness", 300.0), errors="coerce").fillna(300.0)
        norm_b = np.clip((b_series - 300.0) / (500.0 - 300.0), 0.0, 1.0)

        frp_col = (
            out["fire_radiative_power"]
            if "fire_radiative_power" in out.columns
            else out.get("frp", pd.Series([0.0] * len(out)))
        )
        frp_series = pd.to_numeric(frp_col, errors="coerce").fillna(0.0)
        norm_frp = np.clip(frp_series / 200.0, 0.0, 1.0)

        c_series = pd.to_numeric(out.get("confidence", 50.0), errors="coerce").fillna(50.0)
        norm_c = np.clip(c_series / 100.0, 0.0, 1.0)

        # Composite weighted score
        risk_score = (0.40 * norm_b) + (0.40 * norm_frp) + (0.20 * norm_c)
        risk_score = np.clip(risk_score, 0.0, 1.0)

        out["risk_score"] = risk_score.round(4)

        # Categorical and numerical risk label mapping
        def score_to_level(s: float) -> str:
            if s < 0.25:
                return "Low"
            elif s < 0.50:
                return "Moderate"
            elif s < 0.75:
                return "High"
            return "Critical"

        def level_to_encoded(level: str) -> int:
            return {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}.get(level, 0)

        out["risk_level"] = out["risk_score"].map(score_to_level)
        out["risk_encoded"] = out["risk_level"].map(level_to_encoded).astype(int)

        logger.info(
            f"Created risk labels. Distribution: {out['risk_level'].value_counts().to_dict()}"
        )
        return out


def create_fire_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Module-level helper to generate composite fire risk labels."""
    return FeatureEngineer().create_fire_risk_labels(df)


def engineer_fire_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module-level helper to compute enriched fire features."""
    return FeatureEngineer().engineer_fire_features(df)

