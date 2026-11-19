"""
Database initialization and synthetic demonstration data seeding script.

Sets up database schema tables and populates them with 1,200 realistic demonstration
wildfire detections across 4 global regions and 500 synthetic environmental records.

ALL GENERATED DATA IS STRICTLY LABELED WITH source='DEMO_DATA' OR 'SYNTHETIC_DEMO'.
"""
from datetime import date, datetime, timedelta
import logging
import os
import sys
from typing import Dict, Any, List
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.schema import initialize_schema
from database.db import get_db_session
from database.models import Dataset, FireEvent, EnvironmentalRecord
from src.data_collection.copernicus import generate_synthetic_environmental_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("initialize_database")


def _get_season_for_region(month: int, is_south: bool) -> str:
    """Return meteorological season taking hemisphere into account."""
    if not is_south:
        if month in (12, 1, 2):
            return "Winter"
        elif month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        else:
            return "Fall"
    else:
        # Southern hemisphere: seasons reversed
        if month in (12, 1, 2):
            return "Summer"
        elif month in (3, 4, 5):
            return "Fall"
        elif month in (6, 7, 8):
            return "Winter"
        else:
            return "Spring"


def generate_demo_fire_events(dataset_id: int, seed: int = 42) -> List[FireEvent]:
    """
    Generate 1,200 realistic demo fire events across 4 fire-prone regions:
        1. Western USA (300 events) — peak Jun-Sep
        2. Queensland Australia (300 events) — peak Oct-Feb
        3. Mediterranean (300 events) — peak Jul-Sep
        4. British Columbia Canada (300 events) — peak Jul-Sep

    Data provenance label is explicitly 'DEMO_DATA'.

    Args:
        dataset_id: Foreign key to datasets table.
        seed: Random seed for repeatability.

    Returns:
        List of FireEvent ORM instances ready for persistence.
    """
    rng = np.random.default_rng(seed)
    satellites = ["Terra", "Aqua", "NOAA-20", "Suomi-NPP"]

    # Region definitions: (name, lat_min, lat_max, lon_min, lon_max, country_list, is_south, peak_months, month_weights)
    # Months are 1..12
    # Weight distributions normalized to 1.0
    regions_spec = [
        {
            "name": "Western USA",
            "lat": (32.0, 49.0),
            "lon": (-125.0, -102.0),
            "countries": ["United States"],
            "is_south": False,
            "peak_months": {6, 7, 8, 9},
            # Higher in summer months 6-9
            "weights": [0.01, 0.01, 0.02, 0.04, 0.08, 0.20, 0.30, 0.22, 0.09, 0.02, 0.005, 0.005],
        },
        {
            "name": "Queensland Australia",
            "lat": (-28.0, -10.0),
            "lon": (138.0, 154.0),
            "countries": ["Australia"],
            "is_south": True,
            "peak_months": {10, 11, 12, 1, 2},
            # Higher in austral spring/summer
            "weights": [0.18, 0.12, 0.04, 0.02, 0.01, 0.01, 0.01, 0.03, 0.08, 0.16, 0.18, 0.16],
        },
        {
            "name": "Mediterranean",
            "lat": (36.0, 45.0),
            "lon": (-9.0, 30.0),
            "countries": ["Spain", "Greece", "Portugal", "Italy"],
            "is_south": False,
            "peak_months": {7, 8, 9},
            # High in Jul-Aug
            "weights": [0.01, 0.01, 0.02, 0.03, 0.05, 0.15, 0.35, 0.28, 0.07, 0.02, 0.005, 0.005],
        },
        {
            "name": "British Columbia Canada",
            "lat": (49.0, 60.0),
            "lon": (-139.0, -114.0),
            "countries": ["Canada"],
            "is_south": False,
            "peak_months": {7, 8, 9},
            # High in Jul-Aug
            "weights": [0.005, 0.005, 0.01, 0.02, 0.06, 0.18, 0.38, 0.25, 0.07, 0.01, 0.005, 0.005],
        },
    ]

    events: List[FireEvent] = []
    months_arr = np.arange(1, 13)

    for reg in regions_spec:
        w = np.array(reg["weights"])
        w = w / w.sum()  # normalize

        for _ in range(300):
            lat = float(rng.uniform(reg["lat"][0], reg["lat"][1]))
            lon = float(rng.uniform(reg["lon"][0], reg["lon"][1]))
            country = str(rng.choice(reg["countries"]))
            year = int(rng.integers(2010, 2024))
            month = int(rng.choice(months_arr, p=w))
            day = int(rng.integers(1, 29))  # Safe max day

            acq_date = date(year, month, day)

            # Hour and minute for satellite overpass
            hour = int(rng.choice([1, 2, 10, 11, 13, 14, 21, 22]))
            minute = int(rng.integers(0, 60))
            acq_time = f"{hour:02d}{minute:02d}"
            daynight = "D" if 6 <= hour < 18 else "N"

            # Season
            season = _get_season_for_region(month, reg["is_south"])

            # Peak month boost
            is_peak = month in reg["peak_months"]
            peak_boost = 20.0 if is_peak else 0.0

            # Realistic brightness: normal(330, 30) + peak_boost, clamped [300, 500]
            raw_bright = float(rng.normal(330.0, 30.0) + peak_boost)
            brightness = float(np.clip(raw_bright, 300.0, 500.0))

            # FRP: exponential(15) * seasonal_boost clamped [0, 500]
            frp_boost = 1.6 if is_peak else 0.9
            raw_frp = float(rng.exponential(15.0) * frp_boost)
            frp = float(np.clip(raw_frp, 0.0, 500.0))

            # Confidence: 0-100, biased towards high for strong fires
            conf = int(np.clip(rng.normal(75, 18), 30, 100))

            satellite = str(rng.choice(satellites))

            event = FireEvent(
                dataset_id=dataset_id,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                acquisition_date=acq_date,
                acquisition_time=acq_time,
                brightness=round(brightness, 1),
                confidence=conf,
                fire_radiative_power=round(frp, 1),
                satellite_source=satellite,
                daynight=daynight,
                country=country,
                season=season,
            )
            events.append(event)

    return events


def initialize_database() -> Dict[str, int]:
    """
    Initialize database schema and populate with benchmark demonstration datasets.

    Returns:
        Dict reporting record counts: {'fire_events': int, 'env_records': int}.
    """
    logger.info("Initializing database schema...")
    initialize_schema()
    logger.info("Database schema initialized.")

    with get_db_session() as session:
        # Check if demo datasets already exist
        existing_demo = session.query(Dataset).filter_by(is_demo=1).all()
        if existing_demo:
            logger.info(
                f"Found {len(existing_demo)} existing demo dataset(s). Skipping duplicate re-population."
            )
            fire_count = session.query(FireEvent).count()
            env_count = session.query(EnvironmentalRecord).count()
            return {"fire_events": fire_count, "env_records": env_count}

        # 1. Create Fire Dataset Record
        fire_dataset = Dataset(
            dataset_name="Global Wildfire Multi-Region Benchmark (Demo)",
            source="DEMO_DATA",
            upload_date=datetime.utcnow(),
            record_count=1200,
            date_range="2010-01-01 to 2023-12-31",
            description=(
                "Demonstration dataset containing 1,200 simulated satellite wildfire observations "
                "across 4 major fire regimes: Western USA, Queensland Australia, Mediterranean Basin, "
                "and British Columbia Canada. Explicitly labeled DEMO_DATA."
            ),
            is_demo=1,
        )
        session.add(fire_dataset)
        session.flush()

        # 2. Create Environmental Dataset Record
        env_dataset = Dataset(
            dataset_name="Synthetic Environmental Reanalysis Benchmark (Demo)",
            source="SYNTHETIC_DEMO",
            upload_date=datetime.utcnow(),
            record_count=500,
            date_range="2023-01-01 to 2023-12-31",
            description=(
                "Demonstration meteorological dataset containing 500 synthetic weather records "
                "(temperature, humidity, wind, precipitation) aligned with fire hotspots."
            ),
            is_demo=1,
        )
        session.add(env_dataset)
        session.flush()

        # 3. Generate & bulk insert Fire Events
        logger.info("Generating 1,200 demo fire events across 4 regions...")
        fire_events = generate_demo_fire_events(dataset_id=fire_dataset.dataset_id, seed=42)
        session.bulk_save_objects(fire_events)
        session.flush()
        logger.info(f"Successfully inserted {len(fire_events)} demo fire events.")

        # 4. Generate & bulk insert Environmental Records
        logger.info("Generating 500 synthetic environmental records...")
        env_df = generate_synthetic_environmental_data(n=500, seed=42)
        env_records = []
        for _, row in env_df.iterrows():
            env_rec = EnvironmentalRecord(
                dataset_id=env_dataset.dataset_id,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                timestamp=row["timestamp"],
                temperature=float(row["temperature"]),
                humidity=float(row["humidity"]),
                wind_speed=float(row["wind_speed"]),
                wind_direction=float(row["wind_direction"]),
                precipitation=float(row["precipitation"]),
            )
            env_records.append(env_rec)

        session.bulk_save_objects(env_records)
        session.flush()
        logger.info(f"Successfully inserted {len(env_records)} synthetic environmental records.")

        session.commit()

        summary = {
            "fire_events": len(fire_events),
            "env_records": len(env_records),
        }
        print("=" * 65)
        print("DATABASE INITIALIZATION SUMMARY")
        print("=" * 65)
        print(f" Schema:               Initialized successfully")
        print(f" Demo Fire Events:     {summary['fire_events']} records inserted")
        print(f" Synthetic Env Records: {summary['env_records']} records inserted")
        print(" Data Provenance:      Clearly labeled 'DEMO_DATA' / 'SYNTHETIC_DEMO'")
        print("=" * 65)
        return summary


if __name__ == "__main__":
    initialize_database()
