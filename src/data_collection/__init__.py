"""
Data collection module for the AI-Driven Forest Fire Intelligence System.

Provides collectors for:
    - NASA FIRMS satellite active fire data (MODIS, VIIRS)
    - UCI Forest Fires benchmark dataset
    - Copernicus Climate Data Store (ERA5) & synthetic environmental demo generator
"""
from src.data_collection.nasa_firms import NASAFIRMSCollector, load_firms_csv
from src.data_collection.uci_dataset import UCIForestFiresCollector, get_uci_dataset
from src.data_collection.copernicus import (
    CopernicusCollector,
    generate_synthetic_environmental_data,
)

__all__ = [
    "NASAFIRMSCollector",
    "load_firms_csv",
    "UCIForestFiresCollector",
    "get_uci_dataset",
    "CopernicusCollector",
    "generate_synthetic_environmental_data",
]
