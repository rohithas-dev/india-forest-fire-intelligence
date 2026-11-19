"""
Environmental and meteorological correlation analysis for forest fires.

Calculates linear (Pearson), rank-order (Spearman), and non-linear (XGBoost)
feature associations with wildfire intensity and occurrence.

DISCLAIMER:
    Correlation does not imply causation. High correlation indicates statistical
    association under observed conditions, not definitive physical causality.
"""
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xgboost as xgb

logger = logging.getLogger(__name__)


class EnvironmentalCorrelator:
    """Computes correlation matrices and non-linear feature importances."""

    DISCLAIMER: str = (
        "Statistical Disclaimer: Correlation does not imply causation. "
        "Observed associations should be interpreted in conjunction with domain fire physics."
    )

    def compute_pearson(
        self,
        df: pd.DataFrame,
        target_col: str = "fire_radiative_power",
    ) -> pd.DataFrame:
        """
        Compute Pearson linear correlation matrix for all numeric columns.

        Args:
            df: Input DataFrame.
            target_col: Target column to sort correlations by if present.

        Returns:
            pd.DataFrame correlation matrix.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to compute_pearson.")
            return pd.DataFrame()

        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or len(num_df.columns) < 2:
            return pd.DataFrame()

        corr = num_df.corr(method="pearson").round(4)
        if target_col in corr.columns:
            corr = corr.sort_values(by=target_col, ascending=False)
        return corr

    def compute_spearman(
        self,
        df: pd.DataFrame,
        target_col: str = "fire_radiative_power",
    ) -> pd.DataFrame:
        """
        Compute Spearman rank-order correlation matrix for all numeric columns.

        Args:
            df: Input DataFrame.
            target_col: Target column to sort correlations by if present.

        Returns:
            pd.DataFrame correlation matrix.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to compute_spearman.")
            return pd.DataFrame()

        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or len(num_df.columns) < 2:
            return pd.DataFrame()

        corr = num_df.corr(method="spearman").round(4)
        if target_col in corr.columns:
            corr = corr.sort_values(by=target_col, ascending=False)
        return corr

    def compute_xgboost_importance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> pd.DataFrame:
        """
        Train an XGBoost model to evaluate non-linear feature importance against the target.

        Uses chronological split if temporal columns exist, otherwise 80/20 train/test.

        Args:
            df: Input DataFrame.
            feature_cols: Candidate predictor columns.
            target_col: Target column name.

        Returns:
            pd.DataFrame with 'feature' and 'importance' columns sorted descending.
        """
        if df is None or df.empty or target_col not in df.columns:
            return pd.DataFrame(columns=["feature", "importance"])

        # Filter available feature columns that are numeric
        valid_features = [
            c for c in feature_cols if c in df.columns and c != target_col
        ]
        if not valid_features or len(df) < 20:
            return pd.DataFrame(columns=["feature", "importance"])

        data = df[valid_features + [target_col]].dropna()
        if len(data) < 20:
            return pd.DataFrame(columns=["feature", "importance"])

        # Chronological sort if date column present
        for dt_col in ["acquisition_date", "timestamp", "date"]:
            if dt_col in df.columns:
                data = data.sort_index()
                break

        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        X_train = train_data[valid_features]
        y_train = train_data[target_col]

        # Determine task type: classification or regression
        is_classification = (
            y_train.nunique() <= 5
            or str(y_train.dtype).startswith(("int", "cat", "bool"))
        )

        try:
            if is_classification:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.08,
                    random_state=42,
                    eval_metric="logloss",
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.08,
                    random_state=42,
                )

            model.fit(X_train, y_train)
            importances = model.feature_importances_

            imp_df = pd.DataFrame({
                "feature": valid_features,
                "importance": importances,
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)

            logger.info("Computed XGBoost feature importance. Note: Correlation does not imply causation.")
            return imp_df
        except Exception as exc:
            logger.warning(f"Error computing XGBoost feature importance: {exc}")
            return pd.DataFrame(columns=["feature", "importance"])

    def merge_fire_and_env(
        self,
        fire_df: pd.DataFrame,
        env_df: pd.DataFrame,
        tolerance_deg: float = 0.5,
    ) -> pd.DataFrame:
        """
        Spatial join: matches fire events to the nearest environmental record
        within tolerance_deg degrees using Euclidean KD-Tree lookup on (lat, lon).

        Args:
            fire_df: Cleaned fire detections with latitude and longitude.
            env_df: Meteorological records with latitude, longitude, and weather fields.
            tolerance_deg: Maximum spatial distance in degrees (~55 km at equator).

        Returns:
            pd.DataFrame fire detections merged with nearest meteorological attributes.
        """
        if fire_df is None or fire_df.empty:
            return pd.DataFrame()
        if env_df is None or env_df.empty:
            return fire_df.copy()

        required = ["latitude", "longitude"]
        if not all(c in fire_df.columns for c in required) or not all(c in env_df.columns for c in required):
            logger.warning("Latitude and longitude must be present in both DataFrames.")
            return fire_df.copy()

        fire_coords = fire_df[required].dropna().values
        env_coords = env_df[required].dropna().values

        if len(fire_coords) == 0 or len(env_coords) == 0:
            return fire_df.copy()

        # Build KD-Tree on environmental coordinates
        tree = cKDTree(env_coords)
        distances, indices = tree.query(fire_coords, distance_upper_bound=tolerance_deg)

        env_cols_to_merge = [
            c for c in env_df.columns if c not in ["latitude", "longitude", "dataset_id", "record_id"]
        ]

        merged = fire_df.copy()
        for col in env_cols_to_merge:
            merged[col] = np.nan

        # Map matched records
        valid_match = distances <= tolerance_deg
        for i, (is_valid, idx) in enumerate(zip(valid_match, indices)):
            if is_valid and idx < len(env_df):
                for col in env_cols_to_merge:
                    merged.iloc[i, merged.columns.get_loc(col)] = env_df.iloc[idx][col]

        logger.info(
            f"Merged {len(fire_df)} fire events with {len(env_df)} env records. "
            f"Matched {int(valid_match.sum())} within {tolerance_deg}° tolerance."
        )
        return merged

    def monthly_fire_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate fire detections into monthly time series counts.

        Args:
            df: DataFrame with 'acquisition_date' or date column.

        Returns:
            pd.DataFrame with 'year_month' and 'fire_count' columns.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["year_month", "fire_count"])

        date_col = None
        for candidate in ["acquisition_date", "timestamp", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break

        if not date_col:
            return pd.DataFrame(columns=["year_month", "fire_count"])

        data = df.copy()
        parsed = pd.to_datetime(data[date_col], errors="coerce")
        data = data[parsed.notna()]
        data["year_month"] = parsed.dt.to_period("M").astype(str)

        counts = (
            data.groupby("year_month")
            .size()
            .reset_index(name="fire_count")
            .sort_values(by="year_month")
            .reset_index(drop=True)
        )
        return counts

    def seasonal_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group fire events by season and compute mean brightness, mean FRP, and count.

        Args:
            df: DataFrame containing 'season', 'brightness', and 'fire_radiative_power' (or 'frp').

        Returns:
            pd.DataFrame with columns: season, count, avg_brightness, avg_frp.
        """
        if df is None or df.empty or "season" not in df.columns:
            return pd.DataFrame(columns=["season", "count", "avg_brightness", "avg_frp"])

        data = df.copy()
        bright_col = "brightness" if "brightness" in data.columns else None
        frp_col = (
            "fire_radiative_power"
            if "fire_radiative_power" in data.columns
            else ("frp" if "frp" in data.columns else None)
        )

        rows = []
        for season_name, grp in data.groupby("season"):
            avg_b = float(grp[bright_col].mean()) if bright_col else 0.0
            avg_f = float(grp[frp_col].mean()) if frp_col else 0.0
            rows.append({
                "season": season_name,
                "count": len(grp),
                "avg_brightness": round(avg_b, 2),
                "avg_frp": round(avg_f, 2),
            })

        order = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
        out_df = pd.DataFrame(rows)
        if not out_df.empty:
            out_df["order"] = out_df["season"].map(lambda s: order.get(s, 9))
            out_df = out_df.sort_values(by="order").drop(columns=["order"]).reset_index(drop=True)
        return out_df
