"""
Preprocessing pipeline script for fire events and environmental records.

Applies DataCleaner and FeatureEngineer transformations to raw data files,
generating validated, feature-engineered, and risk-labeled output datasets.
"""
import logging
import os
import sys
from typing import Dict
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preprocess_data")


def preprocess_fire_events(input_path: str, output_path: str) -> int:
    """
    Execute full cleaning, feature engineering, and risk labeling on fire events.

    Args:
        input_path: Path to raw fire events CSV.
        output_path: Destination path for processed CSV.

    Returns:
        Number of processed records saved.
    """
    if not os.path.exists(input_path):
        logger.warning(f"Fire events input file not found: {input_path}")
        return 0

    logger.info(f"Loading raw fire events from {input_path}...")
    try:
        raw_df = pd.read_csv(input_path)
    except Exception as exc:
        logger.error(f"Failed to read {input_path}: {exc}")
        return 0

    cleaner = DataCleaner()
    engineer = FeatureEngineer()

    # 1. Clean
    cleaned_df = cleaner.clean_fire_events(raw_df)
    report = cleaner.get_cleaning_report(raw_df, cleaned_df)
    logger.info(
        f"Cleaning report: {report['original_count']} original, {report['cleaned_count']} cleaned "
        f"({report['dropped']} dropped, drop rate: {report['drop_rate']}%)"
    )

    if cleaned_df.empty:
        logger.warning("No records remaining after cleaning.")
        return 0

    # 2. Engineer features
    feat_df = engineer.engineer_fire_features(cleaned_df)

    # 3. Add risk labels
    labeled_df = engineer.create_fire_risk_labels(feat_df)

    # 4. Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(labeled_df)} processed fire events to {output_path}")
    return len(labeled_df)


def preprocess_environmental(input_path: str, output_path: str) -> int:
    """
    Clean and validate meteorological/environmental records.

    Args:
        input_path: Path to raw environmental CSV.
        output_path: Destination path for processed CSV.

    Returns:
        Number of processed records saved.
    """
    if not os.path.exists(input_path):
        logger.warning(f"Environmental input file not found: {input_path}")
        return 0

    logger.info(f"Loading raw environmental records from {input_path}...")
    try:
        raw_df = pd.read_csv(input_path)
    except Exception as exc:
        logger.error(f"Failed to read {input_path}: {exc}")
        return 0

    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_environmental_records(raw_df)

    if cleaned_df.empty:
        logger.warning("No environmental records remaining after cleaning.")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(cleaned_df)} processed environmental records to {output_path}")
    return len(cleaned_df)


def preprocess_all(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> Dict[str, int]:
    """
    Process all raw files found in raw_dir and store results in processed_dir.

    Args:
        raw_dir: Source folder with raw datasets.
        processed_dir: Output folder for cleaned and feature-engineered datasets.

    Returns:
        Dict mapping processed file name to record count.
    """
    raw_path = os.path.join(PROJECT_ROOT, raw_dir) if not os.path.isabs(raw_dir) else raw_dir
    proc_path = os.path.join(PROJECT_ROOT, processed_dir) if not os.path.isabs(processed_dir) else processed_dir

    os.makedirs(proc_path, exist_ok=True)
    summary: Dict[str, int] = {}

    print("\n" + "=" * 60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Check for FIRMS fire files
    for fname in os.listdir(raw_path) if os.path.exists(raw_path) else []:
        src_file = os.path.join(raw_path, fname)
        if "firms" in fname.lower() or "fire" in fname.lower():
            dest_file = os.path.join(proc_path, f"processed_{fname}")
            print(f"Processing fire events: {fname}...")
            count = preprocess_fire_events(src_file, dest_file)
            summary[f"fire_{fname}"] = count
        elif "env" in fname.lower() or "weather" in fname.lower() or "copernicus" in fname.lower():
            dest_file = os.path.join(proc_path, f"processed_{fname}")
            print(f"Processing environmental records: {fname}...")
            count = preprocess_environmental(src_file, dest_file)
            summary[f"env_{fname}"] = count

    if not summary:
        print("No raw data files found in data/raw/ to preprocess.")
        print("Run scripts/download_data.py or scripts/initialize_database.py first.")

    print("=" * 60)
    print("PREPROCESSING PIPELINE COMPLETE")
    for k, v in summary.items():
        print(f" - {k}: {v} records processed")
    print("=" * 60 + "\n")
    return summary


if __name__ == "__main__":
    preprocess_all()
