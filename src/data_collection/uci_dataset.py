"""
UCI Forest Fires benchmark dataset collector and preprocessor.

Dataset reference:
    Cortez and Morais, 2007. A Data Mining Approach to Predict Forest Fires using Meteorological Data.
    URL: https://archive.ics.uci.edu/dataset/162/forest+fires
"""
import io
import logging
from typing import Dict, List
import pandas as pd
import requests

logger = logging.getLogger(__name__)

UCI_COLUMNS: List[str] = [
    "X",
    "Y",
    "month",
    "day",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain",
    "area",
]

MONTH_MAP: Dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class UCIForestFiresCollector:
    """Collector and processor for the UCI Forest Fires benchmark dataset."""

    UCI_URL: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    )

    def __init__(self, url: str = None) -> None:
        """
        Initialize the UCI Forest Fires collector.

        Args:
            url: Optional override for dataset URL.
        """
        self.url: str = url or self.UCI_URL

    def download(self, timeout: int = 30) -> pd.DataFrame:
        """
        Download the UCI Forest Fires dataset from the repository.

        Args:
            timeout: HTTP request timeout in seconds.

        Returns:
            pd.DataFrame with raw columns or empty DataFrame on failure.
        """
        try:
            logger.info(f"Downloading UCI Forest Fires dataset from {self.url}...")
            response = requests.get(self.url, timeout=timeout)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Successfully downloaded UCI dataset ({len(df)} records).")
            return df
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Network error downloading UCI dataset: {exc}")
        except Exception as exc:
            logger.warning(f"Failed to parse UCI dataset: {exc}")

        return pd.DataFrame(columns=UCI_COLUMNS)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess UCI dataset:
            - Map month abbreviation to integer (1-12)
            - Create binary fire_occurred flag (1 if area > 0 else 0)
            - Compute log-transformed burned area log(area + 1)

        Args:
            df: Raw UCI DataFrame.

        Returns:
            pd.DataFrame with added features.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to UCI preprocess.")
            return df.copy()

        processed = df.copy()

        # Clean column names
        processed.columns = [str(c).strip() for c in processed.columns]

        # Convert month string to month number
        if "month" in processed.columns:
            processed["month_num"] = (
                processed["month"].astype(str).str.lower().str.strip().map(MONTH_MAP)
            )

        # Binary fire_occurred label (area > 0)
        if "area" in processed.columns:
            processed["area"] = pd.to_numeric(processed["area"], errors="coerce").fillna(0.0)
            processed["fire_occurred"] = (processed["area"] > 0).astype(int)

        logger.info(f"Preprocessed UCI dataset: {len(processed)} rows.")
        return processed


def get_uci_dataset() -> pd.DataFrame:
    """
    Download and preprocess the UCI Forest Fires dataset.

    Returns:
        Preprocessed pd.DataFrame ready for correlation and ML analysis.
    """
    collector = UCIForestFiresCollector()
    raw_df = collector.download()
    return collector.preprocess(raw_df)
