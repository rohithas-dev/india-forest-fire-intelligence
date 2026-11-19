"""
Unit tests for data cleaning, coordinate validation, imputation, and feature engineering.
"""
from datetime import date
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import (
    DataCleaner,
    add_season_column,
    clean_fire_events,
    get_cleaning_report,
)
from src.preprocessing.feature_engineering import (
    FeatureEngineer,
    create_fire_risk_labels,
    engineer_fire_features,
)


def _unwrap_cleaned(result):
    """Helper to unwrap (df, report) or df return signatures transparently."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _unwrap_report(result, original_df=None):
    """Helper to unwrap report from tuple return or generate it via get_cleaning_report."""
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return get_cleaning_report(original_df or result, result)


def test_clean_fire_events_removes_duplicates() -> None:
    """Provide a DataFrame with 5 duplicated rows and verify duplicates are removed."""
    base_rows = [
        {
            "latitude": 37.5 + i * 0.1,
            "longitude": -120.0 - i * 0.1,
            "acquisition_date": "2021-08-10",
            "acquisition_time": "1430",
            "brightness": 325.0,
            "confidence": 80,
            "fire_radiative_power": 15.0,
        }
        for i in range(10)
    ]
    # Add 5 duplicates of the first 5 rows
    df_raw = pd.DataFrame(base_rows + base_rows[:5])
    assert len(df_raw) == 15

    result = clean_fire_events(df_raw)
    cleaned = _unwrap_cleaned(result)
    report = _unwrap_report(result, df_raw)

    assert len(cleaned) == 10
    assert report.get("dropped_duplicates", report.get("dropped")) == 5


def test_clean_fire_events_removes_invalid_coordinates() -> None:
    """Verify rows with latitude out of [-90, 90] bounds (e.g. lat=200) are dropped."""
    df_raw = pd.DataFrame([
        {
            "latitude": 38.0,
            "longitude": -121.0,
            "acquisition_date": "2022-07-01",
            "brightness": 330.0,
            "confidence": 75,
            "fire_radiative_power": 20.0,
        },
        {
            "latitude": 200.0,  # Invalid latitude
            "longitude": -121.0,
            "acquisition_date": "2022-07-01",
            "brightness": 330.0,
            "confidence": 75,
            "fire_radiative_power": 20.0,
        },
        {
            "latitude": -95.0,  # Invalid latitude
            "longitude": -121.0,
            "acquisition_date": "2022-07-01",
            "brightness": 330.0,
            "confidence": 75,
            "fire_radiative_power": 20.0,
        },
        {
            "latitude": 41.5,
            "longitude": -205.0,  # Invalid longitude
            "acquisition_date": "2022-07-01",
            "brightness": 330.0,
            "confidence": 75,
            "fire_radiative_power": 20.0,
        },
    ])

    result = clean_fire_events(df_raw)
    cleaned = _unwrap_cleaned(result)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["latitude"] == 38.0


def test_clean_fire_events_handles_missing_brightness() -> None:
    """Verify missing/NaN brightness values are filled with the column median."""
    df_raw = pd.DataFrame([
        {
            "latitude": 36.0,
            "longitude": -119.0,
            "acquisition_date": "2021-06-01",
            "acquisition_time": "1000",
            "brightness": 320.0,
            "confidence": 80,
            "fire_radiative_power": 10.0,
        },
        {
            "latitude": 36.1,
            "longitude": -119.1,
            "acquisition_date": "2021-06-01",
            "acquisition_time": "1001",
            "brightness": 340.0,
            "confidence": 80,
            "fire_radiative_power": 15.0,
        },
        {
            "latitude": 36.2,
            "longitude": -119.2,
            "acquisition_date": "2021-06-01",
            "acquisition_time": "1002",
            "brightness": np.nan,  # Missing brightness
            "confidence": 80,
            "fire_radiative_power": 12.0,
        },
    ])

    result = clean_fire_events(df_raw)
    cleaned = _unwrap_cleaned(result)

    assert cleaned["brightness"].isna().sum() == 0
    # Median of 320.0 and 340.0 is 330.0
    imputed_val = cleaned.loc[cleaned["latitude"] == 36.2, "brightness"].values[0]
    assert imputed_val == pytest.approx(330.0, 0.1)


def test_clean_fire_events_clamps_frp() -> None:
    """Verify fire_radiative_power > 2000 MW (e.g. 9999 MW) is clamped to 2000 MW."""
    df_raw = pd.DataFrame([
        {
            "latitude": 39.5,
            "longitude": -122.5,
            "acquisition_date": "2022-09-01",
            "brightness": 380.0,
            "confidence": 99,
            "fire_radiative_power": 9999.0,
        }
    ])

    result = clean_fire_events(df_raw)
    cleaned = _unwrap_cleaned(result)

    assert len(cleaned) == 1
    frp_value = cleaned["fire_radiative_power"].values[0]
    assert frp_value == pytest.approx(2000.0, 0.01)


def test_add_season_column() -> None:
    """Verify add_season_column accurately maps month 1 to Winter and month 7 to Summer."""
    df = pd.DataFrame({
        "acquisition_date": [
            date(2021, 1, 15),   # January (month 1)
            date(2021, 4, 10),   # April (month 4)
            date(2021, 7, 20),   # July (month 7)
            date(2021, 10, 5),   # October (month 10)
        ]
    })

    enriched = add_season_column(df, "acquisition_date")
    assert "season" in enriched.columns

    seasons = list(enriched["season"])
    assert seasons[0] == "Winter"
    assert seasons[1] == "Spring"
    assert seasons[2] == "Summer"
    assert seasons[3] == "Fall"


def test_feature_engineer_adds_columns(sample_fire_df: pd.DataFrame) -> None:
    """Verify FeatureEngineer.engineer_fire_features() populates all expected features."""
    fe = FeatureEngineer()
    enriched = fe.engineer_fire_features(sample_fire_df)

    expected_cols = [
        "day_of_year",
        "month_num",
        "year",
        "season_encoded",
        "daynight_encoded",
        "is_high_confidence",
        "intensity_category",
        "lat_grid",
        "lon_grid",
    ]
    for col in expected_cols:
        assert col in enriched.columns, f"Expected column '{col}' missing from engineered features"

    # Verify type and validity
    assert enriched["season_encoded"].isin([0, 1, 2, 3]).all()
    assert enriched["daynight_encoded"].isin([0, 1]).all()
    assert enriched["is_high_confidence"].isin([0, 1]).all()
    assert enriched["intensity_category"].isin(["Low", "Moderate", "High", "Extreme"]).all()


def test_create_risk_labels() -> None:
    """Verify create_fire_risk_labels outputs all 4 distinct risk tiers."""
    # Construct rows targeted across Low, Moderate, High, and Critical thresholds
    targeted_rows = [
        # Low risk: cool, low FRP, low confidence
        {"brightness": 302.0, "fire_radiative_power": 2.0, "confidence": 50},
        # Moderate risk
        {"brightness": 350.0, "fire_radiative_power": 45.0, "confidence": 60},
        # High risk
        {"brightness": 410.0, "fire_radiative_power": 120.0, "confidence": 80},
        # Critical risk: extremely hot, intense FRP, high confidence
        {"brightness": 490.0, "fire_radiative_power": 220.0, "confidence": 98},
    ]
    df = pd.DataFrame(targeted_rows)

    labeled = create_fire_risk_labels(df)
    assert "risk_score" in labeled.columns
    assert "risk_level" in labeled.columns
    assert "risk_encoded" in labeled.columns

    present_levels = set(labeled["risk_level"].unique())
    expected_levels = {"Low", "Moderate", "High", "Critical"}
    assert expected_levels == present_levels, (
        f"Expected all 4 risk tiers, but found: {present_levels}"
    )


def test_cleaning_report() -> None:
    """Verify cleaning report dictionary contains correct structure, keys, and row counts."""
    orig_df = pd.DataFrame([
        {"latitude": 35.0, "longitude": -118.0, "brightness": 320.0},
        {"latitude": 35.0, "longitude": -118.0, "brightness": 320.0},  # Duplicate
        {"latitude": 190.0, "longitude": -118.0, "brightness": 320.0},  # Invalid
        {"latitude": 36.0, "longitude": -119.0, "brightness": 340.0},
    ])
    cleaned_df = pd.DataFrame([
        {"latitude": 35.0, "longitude": -118.0, "brightness": 320.0},
        {"latitude": 36.0, "longitude": -119.0, "brightness": 340.0},
    ])

    report = get_cleaning_report(orig_df, cleaned_df)
    assert isinstance(report, dict)
    assert "original_count" in report or "initial_rows" in report
    assert "cleaned_count" in report or "final_rows" in report
    assert "dropped" in report
    assert "drop_rate" in report

    assert report.get("original_count", report.get("initial_rows")) == 4
    assert report.get("cleaned_count", report.get("final_rows")) == 2
    assert report["dropped"] == 2
    assert report["drop_rate"] == 50.0


def test_clean_empty_dataframe() -> None:
    """Verify cleaning pipeline handles an empty DataFrame gracefully without raising."""
    empty_df = pd.DataFrame()
    result = clean_fire_events(empty_df)
    cleaned = _unwrap_cleaned(result)

    assert cleaned is not None
    assert cleaned.empty
