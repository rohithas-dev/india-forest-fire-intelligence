"""
NASA FIRMS (Fire Information for Resource Management System) data collector.

Provides access to real-time and historical active fire data from MODIS and VIIRS
satellite sensors via the NASA FIRMS API.

API Registration (free): https://firms.modaps.eosdis.nasa.gov/api/
"""
import io
import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Standard FIRMS CSV columns
FIRMS_COLUMNS = [
    "latitude", "longitude", "brightness", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "version", "bright_t31", "frp", "daynight",
]

# Rename map to our internal schema
_RENAME_MAP = {
    "acq_date": "acquisition_date",
    "acq_time": "acquisition_time",
    "satellite": "satellite_source",
    "frp": "fire_radiative_power",
}

FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api"


class NASAFIRMSCollector:
    """
    Client for the NASA FIRMS active fire API.

    Real-time and near-real-time (NRT) MODIS and VIIRS fire detections.
    Requires a free API key from https://firms.modaps.eosdis.nasa.gov/api/

    If no API key is set, all download methods return an empty DataFrame
    with correct columns — the application continues to function via CSV upload.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialise the FIRMS collector.

        Args:
            api_key: FIRMS MAP_KEY. Falls back to Streamlit secrets, then
                     FIRMS_API_KEY environment variable.
        """
        if api_key:
            self.api_key = api_key
        else:
            # 1. Try Streamlit secrets (Streamlit Cloud)
            try:
                import streamlit as st
                self.api_key = st.secrets.get("FIRMS_API_KEY", "")
            except Exception:
                self.api_key = ""
            # 2. Fall back to env var (local development)
            if not self.api_key:
                self.api_key = os.getenv("FIRMS_API_KEY", "")
        self._timeout: int = 30

    def is_available(self) -> bool:
        """
        Check whether the FIRMS API is reachable and an API key is configured.

        Returns:
            True if key is set and the API health endpoint responds 200.
        """
        if not self.api_key:
            return False
        try:
            url = f"{FIRMS_API_BASE}/country/csv/{self.api_key}/MODIS_NRT/USA/1"
            resp = requests.head(url, timeout=self._timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def download_firms_data(
        self,
        country: str = "USA",
        days: int = 10,
        source: str = "MODIS_NRT",
    ) -> pd.DataFrame:
        """
        Download fire detections for a country over the last N days.

        Args:
            country: ISO-3166 alpha-3 country code (e.g. 'USA', 'AUS').
            days:    Number of days of data to retrieve (1–10 for NRT).
            source:  FIRMS data source (MODIS_NRT, VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT).

        Returns:
            DataFrame with normalised fire event columns, or empty DataFrame on failure.
        """
        if not self.api_key:
            logger.warning(
                "FIRMS_API_KEY not set. Returning empty DataFrame. "
                "Register for a free key at https://firms.modaps.eosdis.nasa.gov/api/"
            )
            return self._empty_df()

        url = f"{FIRMS_API_BASE}/country/csv/{self.api_key}/{source}/{country}/{days}"
        return self._fetch_and_parse(url)

    def get_world_fires(self, days: int = 7) -> pd.DataFrame:
        """
        Download global fire detections for the last N days (MODIS NRT).

        Args:
            days: Number of days (1–7 for world endpoint).

        Returns:
            Normalised DataFrame or empty DataFrame on failure.
        """
        if not self.api_key:
            logger.warning("FIRMS_API_KEY not set. Cannot download world fires.")
            return self._empty_df()

        url = f"{FIRMS_API_BASE}/area/csv/{self.api_key}/MODIS_NRT/world/{days}"
        return self._fetch_and_parse(url)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_and_parse(self, url: str) -> pd.DataFrame:
        """Download a FIRMS CSV URL and return a normalised DataFrame."""
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            return self._normalise(df)
        except requests.exceptions.ConnectionError:
            logger.warning("FIRMS API: connection error — check internet access.")
        except requests.exceptions.Timeout:
            logger.warning("FIRMS API: request timed out after %ds.", self._timeout)
        except requests.exceptions.HTTPError as exc:
            logger.warning("FIRMS API: HTTP %s — %s", exc.response.status_code, exc)
        except Exception as exc:
            logger.warning("FIRMS API: unexpected error — %s", exc)
        return self._empty_df()

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename FIRMS columns to internal schema."""
        df = df.rename(columns=_RENAME_MAP)
        # Ensure acquisition_date is a date
        if "acquisition_date" in df.columns:
            df["acquisition_date"] = pd.to_datetime(
                df["acquisition_date"], errors="coerce"
            ).dt.date
        return df

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the standard fire event columns."""
        return pd.DataFrame(
            columns=[
                "latitude", "longitude", "acquisition_date", "acquisition_time",
                "brightness", "confidence", "fire_radiative_power",
                "satellite_source", "daynight",
            ]
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def load_firms_csv(source: Union[str, io.BytesIO, io.StringIO, Any]) -> pd.DataFrame:
    """
    Parse a NASA FIRMS CSV file (downloaded manually or uploaded by the user).

    Renames FIRMS columns to the internal schema used by this application.

    Args:
        source: File path string, BytesIO, StringIO, or any file-like object.

    Returns:
        Normalised DataFrame.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    try:
        if hasattr(source, "read"):
            source.seek(0)
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source)

        # Apply rename map
        df = df.rename(columns=_RENAME_MAP)
        if "acquisition_date" in df.columns:
            df["acquisition_date"] = pd.to_datetime(
                df["acquisition_date"], errors="coerce"
            ).dt.date
        return df
    except Exception as exc:
        raise ValueError(f"Failed to parse NASA FIRMS data: {exc}") from exc
