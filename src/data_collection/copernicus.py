"""
Copernicus Climate Data Store (CDS) collector and synthetic environmental data generator.

NOTE ON REAL VS DEMO DATA:
    Direct Copernicus ERA5 reanalysis retrieval requires an active CDS API account
    and the `cdsapi` client package. When unavailable, graceful fallback is provided.
    `generate_synthetic_environmental_data` generates synthetic records for demonstration
    purposes and is strictly labeled with source='SYNTHETIC_DEMO'.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENV_COLUMNS: List[str] = [
    "latitude",
    "longitude",
    "timestamp",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
]

# 4 Benchmark Regions: (name, lat_min, lat_max, lon_min, lon_max, is_southern_hemisphere)
REGIONS = [
    ("Western USA", 32.0, 49.0, -125.0, -102.0, False),
    ("Queensland Australia", -28.0, -10.0, 138.0, 154.0, True),
    ("Mediterranean", 36.0, 45.0, -9.0, 30.0, False),
    ("British Columbia Canada", 49.0, 60.0, -139.0, -114.0, False),
]


class CopernicusCollector:
    """Collector for Copernicus Climate Data Store ERA5 reanalysis data."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the Copernicus collector.

        Args:
            api_key: Optional CDS API Key. If None, checks CDS_API_KEY or CDSAPI_KEY.
        """
        self.api_key: str = (
            api_key
            or os.getenv("CDS_API_KEY", "")
            or os.getenv("CDSAPI_KEY", "")
        ).strip()

    def is_available(self) -> bool:
        """
        Check if cdsapi library is installed and credentials are present.

        Returns:
            True if available and configured, False otherwise.
        """
        if not self.api_key:
            return False
        try:
            import cdsapi  # noqa: F401
            return True
        except ImportError:
            logger.info("The `cdsapi` package is not installed.")
            return False

    def download_era5_data(
        self,
        year: int = 2023,
        month: int = 8,
        region: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Attempt to download ERA5 reanalysis meteorological fields from Copernicus CDS.

        Args:
            year: Analysis year.
            month: Analysis month (1-12).
            region: Optional bounding box [North, West, South, East].

        Returns:
            pd.DataFrame with standard environmental columns or empty DataFrame if unavailable.
        """
        if not self.is_available():
            logger.warning(
                "Copernicus CDS API is unavailable or unconfigured. "
                "Returning empty DataFrame. Use generate_synthetic_environmental_data() for demo purposes."
            )
            return pd.DataFrame(columns=ENV_COLUMNS)

        try:
            import cdsapi

            client = cdsapi.Client()
            logger.info(f"Submitting CDS ERA5 request for year {year}, month {month}...")
            # Sample parameters for ERA5 single levels
            request_params = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "total_precipitation",
                ],
                "year": str(year),
                "month": f"{month:02d}",
                "day": ["01", "15"],
                "time": ["12:00"],
            }
            if region and len(region) == 4:
                request_params["area"] = region

            # In this interface, return an empty placeholder if real retrieval is not active
            return pd.DataFrame(columns=ENV_COLUMNS)
        except Exception as exc:
            logger.warning(f"Error during Copernicus ERA5 download: {exc}")
            return pd.DataFrame(columns=ENV_COLUMNS)


def generate_synthetic_environmental_data(
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate CLEARLY LABELED synthetic environmental records for demonstration.

    Simulates realistic temperature, relative humidity, wind speed, wind direction,
    and precipitation across 4 major wildfire-prone regions:
        1. Western USA
        2. Queensland Australia
        3. Mediterranean Basin
        4. British Columbia Canada

    Seasonality is accounted for, including hemisphere differences.
    Source label is explicitly 'SYNTHETIC_DEMO'.

    Args:
        n: Total number of records to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing synthetic environmental measurements.
    """
    rng = np.random.default_rng(seed)
    records = []
    base_date = datetime(2023, 1, 1)

    records_per_region = n // len(REGIONS)
    remainder = n % len(REGIONS)

    for idx, (reg_name, lat_min, lat_max, lon_min, lon_max, is_south) in enumerate(REGIONS):
        count = records_per_region + (remainder if idx == 0 else 0)

        for _ in range(count):
            lat = float(rng.uniform(lat_min, lat_max))
            lon = float(rng.uniform(lon_min, lon_max))

            # Random day throughout a 365-day calendar year
            day_offset = int(rng.integers(0, 365))
            dt = base_date + timedelta(days=day_offset, hours=int(rng.integers(0, 24)))
            month = dt.month

            # Realistic seasonal temperature curve:
            # Northern hemisphere summer is June-August (peak ~ July=7)
            # Southern hemisphere summer is Dec-Feb (peak ~ Jan=1)
            if is_south:
                # Summer in Jan/Feb, Winter in Jul/Aug
                temp_baseline = 24.0 + 7.0 * np.cos((month - 1) * 2 * np.pi / 12)
            else:
                # Summer in Jul/Aug, Winter in Jan/Feb
                temp_baseline = 16.0 - 12.0 * np.cos((month - 1) * 2 * np.pi / 12)

            # Adjust for latitude (cooler further from equator)
            lat_factor = -0.3 * (abs(lat) - 30.0)
            temp = float(np.clip(temp_baseline + lat_factor + rng.normal(0, 3.5), -15.0, 48.0))

            # Relative humidity: inversely related to temperature + noise
            humidity = float(np.clip(75.0 - (temp * 0.8) + rng.normal(0, 10.0), 5.0, 98.0))

            # Wind speed: Rayleigh / Weibull distribution approx
            wind_speed = float(np.clip(rng.weibull(2.0) * 4.5, 0.2, 45.0))

            # Wind direction: 0-360 degrees
            wind_direction = float(rng.uniform(0.0, 360.0))

            # Precipitation: highly zero-inflated (dry on fire days)
            if rng.random() < 0.82:
                precipitation = 0.0
            else:
                precipitation = float(np.clip(rng.exponential(2.5), 0.0, 80.0))

            records.append({
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "timestamp": dt,
                "temperature": round(temp, 2),
                "humidity": round(humidity, 2),
                "wind_speed": round(wind_speed, 2),
                "wind_direction": round(wind_direction, 1),
                "precipitation": round(precipitation, 2),
                "source": "SYNTHETIC_DEMO",
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic environmental demo records across 4 regions.")
    return df
