"""Data cleaning and validation module for wildfire and weather data."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def add_season_column(df: pd.DataFrame, date_col: str = "acquisition_date") -> pd.DataFrame:
    """Add season column (Spring, Summer, Fall, Winter) based on calendar month."""
    if df is None or df.empty or date_col not in df.columns:
        return df

    out = df.copy()
    parsed = pd.to_datetime(out[date_col], errors="coerce")

    def m_to_s(m):
        if pd.isna(m):
            return "Summer"
        m = int(m)
        if m in (12, 1, 2):
            return "Winter"
        elif m in (3, 4, 5):
            return "Spring"
        elif m in (6, 7, 8):
            return "Summer"
        else:
            return "Fall"

    out["season"] = parsed.dt.month.apply(m_to_s)
    return out


class DataCleaner:
    """Preprocesses and cleans raw fire observations and environmental records."""

    add_season_column = staticmethod(add_season_column)

    @staticmethod
    def clean_fire_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and normalize raw fire event dataframe.
        Supports NASA FIRMS MODIS & VIIRS headers as well as generic schemas.
        """
        report = {
            "initial_rows": len(df) if df is not None else 0,
            "dropped_invalid_coords": 0,
            "dropped_duplicates": 0,
            "final_rows": 0,
            "columns_mapped": []
        }
        if df is None or df.empty:
            report["final_rows"] = 0
            return pd.DataFrame(), report

        df = df.copy()

        # Standardize column naming
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Column aliases mapping
        col_map = {
            "lat": "latitude",
            "lon": "longitude",
            "long": "longitude",
            "acq_date": "acquisition_date",
            "date": "acquisition_date",
            "acq_time": "acquisition_time",
            "time": "acquisition_time",
            "bright_ti4": "brightness",
            "bright_ti5": "brightness",
            "bright_t31": "brightness",
            "temp": "brightness",
            "frp": "fire_radiative_power",
            "conf": "confidence",
            "satellite": "satellite_source",
            "instrument": "satellite_source",
            "day_night": "daynight"
        }
        for orig, target in col_map.items():
            if orig in df.columns and target not in df.columns:
                df[target] = df[orig]
                report["columns_mapped"].append(f"{orig} -> {target}")

        # Ensure required coordinates
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return pd.DataFrame(), report

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        
        valid_coords = (
            df["latitude"].notna() &
            df["longitude"].notna() &
            df["latitude"].between(-90, 90) &
            df["longitude"].between(-180, 180)
        )
        report["dropped_invalid_coords"] = int((~valid_coords).sum())
        df = df[valid_coords].copy()
        if df.empty:
            return df, report

        # Dates & Times
        if "acquisition_date" in df.columns:
            df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce").dt.date
        else:
            df["acquisition_date"] = pd.Timestamp.now().date()

        if "acquisition_time" not in df.columns:
            df["acquisition_time"] = "1200"
        else:
            df["acquisition_time"] = df["acquisition_time"].astype(str).str.zfill(4).str[:4]

        # Numeric conversions
        for num_col in ["brightness", "confidence", "fire_radiative_power"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            else:
                df[num_col] = np.nan

        # Fill defaults and bounds
        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].fillna(50).clip(0, 100).astype(int)
        if "fire_radiative_power" in df.columns:
            df["fire_radiative_power"] = df["fire_radiative_power"].fillna(10.0).clip(0.0, 2000.0)
            df["frp"] = df["fire_radiative_power"]
        if "brightness" in df.columns:
            med_b = float(df["brightness"].median()) if df["brightness"].notna().any() else 315.0
            df["brightness"] = df["brightness"].fillna(med_b).clip(200.0, 600.0)

        # Categoricals
        if "satellite_source" not in df.columns:
            df["satellite_source"] = "MODIS-Terra"
        df["satellite_source"] = df["satellite_source"].fillna("MODIS-Terra").astype(str)

        if "daynight" not in df.columns:
            df["daynight"] = "D"
        df["daynight"] = df["daynight"].fillna("D").astype(str).str.upper().str[0]

        if "country" not in df.columns:
            df["country"] = "Global"
        df["country"] = df["country"].fillna("Global").astype(str)

        # Season derived from acquisition_date
        if "season" not in df.columns or df["season"].isna().all():
            df = add_season_column(df, "acquisition_date")

        # Drop duplicates
        init_len = len(df)
        df = df.drop_duplicates(subset=["latitude", "longitude", "acquisition_date", "acquisition_time"])
        report["dropped_duplicates"] = init_len - len(df)
        report["final_rows"] = len(df)

        return df, report

    @staticmethod
    def clean_environmental_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean and normalize raw weather/environmental observation dataframe."""
        report = {
            "initial_rows": len(df) if df is not None else 0,
            "dropped_invalid": 0,
            "final_rows": 0
        }
        if df is None or df.empty:
            return pd.DataFrame(), report

        df = df.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Coordinate check
        for c in ["latitude", "longitude"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = 0.0

        # Numeric bounds
        if "temperature" in df.columns:
            df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").fillna(25.0).clip(-80.0, 60.0)
        else:
            df["temperature"] = 25.0

        if "humidity" in df.columns:
            df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce").fillna(45.0).clip(0, 100)
        else:
            df["humidity"] = 45.0

        if "wind_speed" in df.columns:
            df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce").fillna(12.0).clip(0.0, 150.0)
        else:
            df["wind_speed"] = 12.0

        if "wind_direction" in df.columns:
            df["wind_direction"] = pd.to_numeric(df["wind_direction"], errors="coerce").fillna(180.0).clip(0.0, 360.0)
        else:
            df["wind_direction"] = 180.0

        if "precipitation" in df.columns:
            df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0.0).clip(0.0, 500.0)
        else:
            df["precipitation"] = 0.0

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp"] = pd.Timestamp.now()

        report["final_rows"] = len(df)
        return df, report

    @staticmethod
    def get_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a cleaning summary report."""
        orig = len(original_df) if original_df is not None else 0
        cleaned = len(cleaned_df) if cleaned_df is not None else 0
        dropped = max(0, orig - cleaned)
        rate = round((dropped / orig * 100.0) if orig > 0 else 0.0, 2)
        return {
            "initial_rows": orig,
            "final_rows": cleaned,
            "dropped_invalid_coords": 0,
            "dropped_duplicates": dropped,
            "dropped": dropped,
            "original_count": orig,
            "cleaned_count": cleaned,
            "drop_rate": rate,
        }


def clean_fire_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Module-level clean_fire_events wrapper."""
    return DataCleaner.clean_fire_events(df)


def clean_environmental_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Module-level clean_environmental_records wrapper."""
    return DataCleaner.clean_environmental_records(df)


def get_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """Module-level get_cleaning_report wrapper."""
    return DataCleaner.get_cleaning_report(original_df, cleaned_df)

