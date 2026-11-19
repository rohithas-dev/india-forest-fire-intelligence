"""
Data download pipeline for the Forest Fire Intelligence System.

Fetches active wildfire records from NASA FIRMS API and the UCI Forest Fires benchmark.
Saves raw incoming files to data/raw/ for downstream preprocessing.
"""
import logging
import os
import sys
from typing import Dict, Optional
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_collection.nasa_firms import NASAFIRMSCollector
from src.data_collection.uci_dataset import UCIForestFiresCollector
from src.data_collection.copernicus import CopernicusCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_data")


def download_nasa_firms(
    api_key: Optional[str] = None,
    country: str = "USA",
    days: int = 10,
) -> pd.DataFrame:
    """
    Download Near Real-Time fire detections from NASA FIRMS.

    Args:
        api_key: Optional FIRMS MAP_KEY.
        country: ISO country code (default 'USA').
        days: Days of historical observations (1-10).

    Returns:
        pd.DataFrame containing FIRMS detections.
    """
    collector = NASAFIRMSCollector(api_key=api_key)
    if not collector.api_key:
        logger.warning(
            "No NASA FIRMS API key found in arguments or FIRMS_API_KEY environment variable. "
            "Please register at https://firms.modaps.eosdis.nasa.gov/api/map_key/ to get a free key."
        )
        return pd.DataFrame()

    logger.info(f"Downloading NASA FIRMS fire detections for {country} (past {days} days)...")
    return collector.download_firms_data(country=country, days=days)


def download_uci_dataset() -> pd.DataFrame:
    """
    Download the UCI Forest Fires benchmark meteorological dataset.

    Returns:
        pd.DataFrame containing UCI forest fires data.
    """
    collector = UCIForestFiresCollector()
    logger.info("Downloading UCI Forest Fires dataset from repository...")
    return collector.download()


def download_all(output_dir: str = "data/raw") -> Dict[str, Optional[str]]:
    """
    Execute end-to-end data acquisition for all configured external sources.

    Args:
        output_dir: Target folder to store raw CSV downloads.

    Returns:
        Dict mapping source name to saved CSV file path (or None if unavailable).
    """
    if not os.path.isabs(output_dir):
        target_dir = os.path.join(PROJECT_ROOT, output_dir)
    else:
        target_dir = output_dir

    os.makedirs(target_dir, exist_ok=True)
    results: Dict[str, Optional[str]] = {"nasa_firms": None, "uci": None}

    print("\n" + "=" * 60)
    print("STARTING DATA DOWNLOAD PIPELINE")
    print("=" * 60)

    # 1. UCI Dataset
    print("[1/2] Fetching UCI Forest Fires benchmark dataset...")
    uci_df = download_uci_dataset()
    if not uci_df.empty:
        uci_path = os.path.join(target_dir, "uci_forest_fires.csv")
        uci_df.to_csv(uci_path, index=False)
        results["uci"] = uci_path
        print(f"      -> SUCCESS: Saved {len(uci_df)} records to {uci_path}")
    else:
        print("      -> WARNING: UCI dataset download failed or returned empty.")

    # 2. NASA FIRMS
    print("[2/2] Fetching NASA FIRMS Near Real-Time detections...")
    firms_df = download_nasa_firms(country="USA", days=7)
    if not firms_df.empty:
        firms_path = os.path.join(target_dir, "nasa_firms_usa.csv")
        firms_df.to_csv(firms_path, index=False)
        results["nasa_firms"] = firms_path
        print(f"      -> SUCCESS: Saved {len(firms_df)} records to {firms_path}")
    else:
        print("      -> INFO: FIRMS download skipped or API key not present.")

    return results


def print_download_summary(results: Dict[str, Optional[str]]) -> None:
    """
    Print an executive summary of downloaded artifacts.

    Args:
        results: Dictionary returned by download_all().
    """
    print("\n" + "=" * 60)
    print("DATA DOWNLOAD SUMMARY")
    print("=" * 60)
    for source, path in results.items():
        status = f"Saved: {path}" if path else "Not downloaded / key unavailable"
        print(f" - {source.upper():<15}: {status}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    download_results = download_all()
    print_download_summary(download_results)
