"""
Shared pytest fixtures for the AI-Driven Forest Fire Intelligence System test suite.
"""
import os
from datetime import date, datetime, timedelta
from typing import Generator
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from database.db import get_session_factory, reset_engine
from database.schema import drop_all_tables, initialize_schema
from src.preprocessing.feature_engineering import FeatureEngineer, create_fire_risk_labels


@pytest.fixture(scope="function")
def in_memory_db() -> Generator[sessionmaker, None, None]:
    """
    Provide an isolated in-memory SQLite database session factory for unit tests.

    Sets DATABASE_URL to 'sqlite:///:memory:', resets the engine, initializes schema tables,
    and yields the active session factory. Upon fixture teardown, tables are dropped and the
    engine is disposed.
    """
    previous_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

    reset_engine()
    initialize_schema()
    factory = get_session_factory()

    try:
        yield factory
    finally:
        try:
            drop_all_tables()
        except Exception:
            pass
        reset_engine()
        if previous_url is not None:
            os.environ["DATABASE_URL"] = previous_url
        else:
            os.environ.pop("DATABASE_URL", None)


@pytest.fixture(scope="function")
def sample_fire_df() -> pd.DataFrame:
    """
    Generate a reproducible sample DataFrame of 100 satellite fire detection events.

    Realistic ranges:
        - latitude: uniform(32, 49)
        - longitude: uniform(-125, -102)
        - brightness: normal(330, 30) clipped to [300, 450] Kelvin
        - fire_radiative_power: exponential(15) clipped to [0, 300] MW
        - confidence: randint(50, 100)%
        - acquisition_date: random dates between 2020 and 2022
    """
    rng = np.random.RandomState(42)
    n = 100

    latitudes = rng.uniform(32.0, 49.0, size=n)
    longitudes = rng.uniform(-125.0, -102.0, size=n)
    brightness = np.clip(rng.normal(330.0, 30.0, size=n), 300.0, 450.0)
    frp = np.clip(rng.exponential(15.0, size=n), 0.0, 300.0)
    confidence = rng.randint(50, 101, size=n)

    # Random dates across calendar years 2020 through 2022 (1095 days)
    start_date = date(2020, 1, 1)
    day_offsets = rng.randint(0, 1096, size=n)
    acq_dates = [start_date + timedelta(days=int(d)) for d in day_offsets]

    hours = rng.randint(0, 24, size=n)
    minutes = rng.randint(0, 60, size=n)
    acq_times = [f"{h:02d}{m:02d}" for h, m in zip(hours, minutes)]

    satellite_options = ["MODIS-Terra", "MODIS-Aqua", "VIIRS-SNPP", "VIIRS-NOAA20"]
    satellite_sources = [satellite_options[i] for i in rng.randint(0, len(satellite_options), size=n)]

    daynights = ["D" if r > 0.35 else "N" for r in rng.rand(n)]
    countries = ["USA"] * n

    def get_season(d: date) -> str:
        m = d.month
        if m in (12, 1, 2):
            return "Winter"
        elif m in (3, 4, 5):
            return "Spring"
        elif m in (6, 7, 8):
            return "Summer"
        return "Fall"

    seasons = [get_season(d) for d in acq_dates]

    df = pd.DataFrame({
        "latitude": np.round(latitudes, 5),
        "longitude": np.round(longitudes, 5),
        "acquisition_date": acq_dates,
        "acquisition_time": acq_times,
        "brightness": np.round(brightness, 2),
        "confidence": confidence,
        "fire_radiative_power": np.round(frp, 2),
        "satellite_source": satellite_sources,
        "daynight": daynights,
        "country": countries,
        "season": seasons,
    })
    return df


@pytest.fixture(scope="function")
def sample_env_df() -> pd.DataFrame:
    """
    Generate a reproducible sample DataFrame of 100 meteorological / environmental records.

    Realistic ranges:
        - temperature: uniform(-10, 45) °C
        - humidity: uniform(0, 100) %
        - wind_speed: uniform(0, 30) m/s
        - wind_direction: uniform(0, 360) deg
        - precipitation: uniform(0, 50) mm
    """
    rng = np.random.RandomState(42)
    n = 100

    latitudes = rng.uniform(32.0, 49.0, size=n)
    longitudes = rng.uniform(-125.0, -102.0, size=n)

    base_time = datetime(2021, 6, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=int(h)) for h in rng.randint(0, 24 * 90, size=n)]

    temperature = rng.uniform(-10.0, 45.0, size=n)
    humidity = rng.uniform(0.0, 100.0, size=n)
    wind_speed = rng.uniform(0.0, 30.0, size=n)
    wind_direction = rng.uniform(0.0, 360.0, size=n)
    precipitation = rng.uniform(0.0, 50.0, size=n)

    df = pd.DataFrame({
        "latitude": np.round(latitudes, 5),
        "longitude": np.round(longitudes, 5),
        "timestamp": timestamps,
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
        "wind_speed": np.round(wind_speed, 2),
        "wind_direction": np.round(wind_direction, 1),
        "precipitation": np.round(precipitation, 2),
    })
    return df


@pytest.fixture(scope="function")
def sample_df_with_features(sample_fire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich sample fire detection records with engineered features and fire risk target labels.

    Applies:
        1. FeatureEngineer().engineer_fire_features()
        2. create_fire_risk_labels()
    """
    fe = FeatureEngineer()
    df_features = fe.engineer_fire_features(sample_fire_df)
    df_labeled = create_fire_risk_labels(df_features)
    return df_labeled
