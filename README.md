# 🔥 AI-Driven Forest Fire Pattern Discovery System

## Overview
The **AI-Driven Forest Fire Pattern Discovery System** is an end-to-end analytical and machine learning platform designed to uncover spatial hotspots, temporal fire regimes, and multi-factor meteorological risk dynamics from satellite observation feeds. By integrating satellite detections from NASA FIRMS (MODIS and VIIRS sensors), Copernicus ERA5 atmospheric reanalysis, and canonical wildfire benchmarks, the platform transforms raw thermal telemetry into actionable intelligence. The application delivers interactive geospatial maps, density-based anomaly investigations, multi-class risk predictions with gradient boosting classifiers, and downloadable audit reports via a production-grade Streamlit web interface.

---

## Features
- **Multi-Source Data Ingestion**: Direct support for NASA FIRMS Near Real-Time (NRT) satellite feeds (MODIS Terra/Aqua, VIIRS SNPP/NOAA-20), Copernicus ERA5 reanalysis data, and the UCI Forest Fires meteorological benchmark.
- **Robust Automated Preprocessing**: Automated column alias resolution, coordinate boundary validation (`[-90, 90]`, `[-180, 180]`), missing value imputation using median statistics, Fire Radiative Power (FRP) clamping, and calendar-based seasonal tagging.
- **Advanced Feature Engineering**: Extraction of day-of-year, cyclic calendar month features, day/night satellite transit flags, 0.1° spatial binning grids, and multi-factor composite risk indices.
- **Centroid-Based Spatial Clustering (K-Means)**: Partitions continental fire outbreaks into regional clusters with automated elbow-curve inertia search and silhouette score evaluations.
- **Density-Based Spatial Clustering (DBSCAN)**: Employs true spherical haversine distance metrics to delineate dense, contiguous wildfire outbreak zones from isolated spatial noise without requiring a pre-set cluster count.
- **Unsupervised Anomaly Detection (Isolation Forest)**: Partitions high-dimensional thermal and geographic feature spaces to flag disproportionately intense megafires and unexpected spatial outliers.
- **Local Outlier Factor (LOF) Analysis**: Measures local density variations against neighboring detections to identify isolated, atypical ignition events.
- **Supervised Wildfire Risk Modeling**: Multi-class fire hazard classification (Low, Moderate, High, Critical) using XGBoost, Random Forest, and Gradient Boosting estimators.
- **Comparative Model Benchmarking**: Side-by-side performance evaluation across models assessing weighted Accuracy, Precision, Recall, F1-Score, and One-vs-Rest ROC-AUC.
- **Rich Interactive Visualizations**: Folium choropleths, heatmap layers, and dual-coordinate spatial maps integrated with Plotly interactive charts for feature importances and ROC curves.
- **Meteorological Correlation Matrix**: Pearson and Spearman correlation heatmaps analyzing pairwise relationships between temperature, relative humidity, wind velocity, precipitation, and fire intensity.
- **SQL Database Storage & Auditing**: SQLAlchemy 2.0 ORM architecture supporting zero-config local SQLite file storage as well as production PostgreSQL instances with foreign-key cascade integrity.
- **Automated Experiment Tracking**: Records hyperparameters, feature contributions, and validation metrics for every training run in persistent relational tables.
- **Exportable PDF & Data Reports**: Summary dashboards, diagnostic confusion heatmaps, and filtered tabular datasets exportable to CSV or PDF.

---

## Architecture

```mermaid
graph TD
    A["Data Sources<br/>(NASA FIRMS / Copernicus / UCI)"] --> B["Data Collection Layer<br/>(API Clients & CSV Parsers)"]
    B --> C["Data Preprocessing & Validation<br/>(Cleaner, Season Tagger, Feature Engineer)"]
    C --> D["Database Persistence<br/>(SQLAlchemy ORM - SQLite / PostgreSQL)"]
    D --> E["Machine Learning Modules"]
    E --> E1["Spatial Clustering<br/>(K-Means & Haversine DBSCAN)"]
    E --> E2["Anomaly Detection<br/>(Isolation Forest & LOF)"]
    E --> E3["Predictive Risk Models<br/>(XGBoost, Random Forest, GBDT)"]
    E1 --> F["Streamlit Multi-Page UI"]
    E2 --> F
    E3 --> F
    F --> G["Interactive Folium & Plotly Analytics"]
    F --> H["PDF Audit Reports & CSV Export"]
```

---

## Project Structure

```
forest-fire-intelligence/
├── .streamlit/
│   └── config.toml                  # Streamlit visual theme and server settings
├── data/
│   ├── processed/
│   │   └── .gitkeep                 # Directory for cleaned and transformed files
│   ├── raw/
│   │   └── .gitkeep                 # Directory for raw downloaded satellite feeds
│   └── sample/
│       ├── .gitkeep
│       ├── sample_environmental.csv # Synthetic meteorological records (DEMO_DATA)
│       └── sample_fire_events.csv   # Synthetic satellite fire detections (DEMO_DATA)
├── database/
│   ├── __init__.py
│   ├── db.py                        # Database engine, connection pool, and session manager
│   ├── models.py                    # SQLAlchemy 2.0 ORM schema definitions (7 tables)
│   └── schema.py                    # Schema creation, migration, and teardown utilities
├── pages/
│   ├── 1_Dataset_Explorer.py        # Dataset upload, preview, and validation page
│   ├── 2_Geographical_Intelligence.py# K-Means and DBSCAN hotspot clustering maps
│   ├── 3_Environmental_Analysis.py  # Weather correlation matrices and temporal trends
│   ├── 4_AI_Laboratory.py          # Supervised risk model training and comparisons
│   └── 5_Anomaly_Investigation.py   # Isolation Forest and LOF anomaly detection
├── scripts/
│   ├── __init__.py
│   ├── download_data.py             # CLI utility for automated data retrieval
│   ├── initialize_database.py       # Seeds database with demo records and schemas
│   └── preprocess_data.py           # Standalone offline ETL cleaning pipeline
├── src/
│   ├── __init__.py
│   ├── anomaly_detection/
│   │   ├── __init__.py
│   │   ├── isolation_forest_detector.py # Tree-based spatial/intensity outlier model
│   │   └── lof_detector.py          # Density-based Local Outlier Factor detector
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── dbscan_clustering.py     # Haversine spatial density clustering
│   │   └── kmeans_clustering.py     # Centroid clustering and optimal-k elbow curves
│   ├── correlation/
│   │   ├── __init__.py
│   │   └── environmental_correlation.py # Statistical weather correlation matrices
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── copernicus.py            # Copernicus CDS client and synthetic weather generator
│   │   ├── nasa_firms.py            # NASA FIRMS API collector and CSV loader
│   │   └── uci_dataset.py           # UCI Forest Fires benchmark dataset downloader
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── model_evaluator.py       # Confusion matrices, ROC curves, feature charts
│   ├── models/
│   │   ├── __init__.py              # Unified machine learning package interface
│   │   └── evaluation.py            # Dark-theme evaluation diagnostics
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── risk_model.py            # XGBoost, Random Forest, and GBDT risk models
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py               # DataCleaner pipeline and bounds validation
│   │   └── feature_engineering.py   # Multi-factor risk indices and feature derivation
│   ├── spatial/
│   │   ├── clustering.py            # Spatial clustering implementation
│   │   └── hotspots.py              # Hotspot density kernel calculators
│   └── utils/
│       ├── __init__.py
│       └── ui.py                    # Shared Streamlit UI components and CSS styling
├── tests/
│   ├── __init__.py
│   ├── .gitkeep
│   ├── conftest.py                  # Pytest fixtures (in-memory SQLite, sample data)
│   ├── test_database.py             # Schema, CRUD, and cascade deletion tests
│   ├── test_models.py               # Clustering, anomaly detection, and ML tests
│   └── test_preprocessing.py        # Data cleaning, bounds, and feature tests
├── app.py                           # Application entry point and Executive Summary
├── pytest.ini                       # Pytest test discovery and execution configuration
├── requirements.txt                 # Pinned dependencies for Python 3.11+
└── README.md                        # Project documentation and deployment guide
```

---

## Technology Stack

| Category | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Core Framework** | Streamlit | `>=1.35.0` | Multi-page reactive web user interface |
| **Language Runtime** | Python | `>=3.11` | Modern Python features, strong typing, pattern matching |
| **Data Processing** | pandas | `>=2.2.0` | High-performance tabular data manipulation and analytics |
| **Numerical Computing** | NumPy / SciPy | `>=1.26.0` | Vectorized math, haversine metrics, statistics |
| **Machine Learning** | scikit-learn | `>=1.4.0` | Clustering (K-Means, DBSCAN), Anomaly (iForest, LOF), Metrics |
| **Gradient Boosting** | XGBoost | `>=2.0.0` | High-speed multi-class classification for wildfire risk |
| **Database ORM** | SQLAlchemy | `>=2.0.0` | Relational ORM supporting SQLite and PostgreSQL dialects |
| **Geospatial Mapping** | Folium / streamlit-folium | `>=0.17.0` | Interactive Leaflet maps, heatmaps, and cluster markers |
| **Data Visualisation** | Plotly / Seaborn / Matplotlib | `>=5.22.0` | Interactive curves, correlation matrices, confusion plots |
| **PDF Reporting** | ReportLab | `>=4.2.0` | Generation of executive summary report documents |
| **Networking** | Requests | `>=2.32.0` | HTTP client for FIRMS and Copernicus REST APIs |
| **Unit Testing** | pytest / pytest-cov | `>=8.2.0` | Automated testing suite with coverage metrics |

---

## Quick Start (Local)

Follow these steps to run the complete system locally:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/forest-fire-intelligence.git
cd forest-fire-intelligence

# 2. Create virtual environment
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialise the database with demonstration data
python scripts/initialize_database.py

# 5. Launch the application
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Data Sources

| Data Source | URL | License | Format | Access Method |
| :--- | :--- | :--- | :--- | :--- |
| **NASA FIRMS** | [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/) | NASA Open Data Policy (Public Domain) | CSV, JSON | REST API (`/area`, `/country`) via free MAP_KEY or manual CSV download |
| **UCI Forest Fires** | [archive.ics.uci.edu/dataset/162/forest+fires](https://archive.ics.uci.edu/dataset/162/forest+fires) | CC BY 4.0 | CSV | Direct HTTP download or preloaded benchmark loader |
| **Copernicus CDS** | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) | Copernicus Open Access License | NetCDF, GRIB | `cdsapi` Python client (requires account registration) or synthetic demo mode |

---

## Acquiring Real Data

### 1. Getting a NASA FIRMS API Key (Free)
1. Visit the NASA FIRMS MAP_KEY registration page: [https://firms.modaps.eosdis.nasa.gov/api/map_key/](https://firms.modaps.eosdis.nasa.gov/api/map_key/).
2. Submit your email address. You will instantly receive a 32-character `MAP_KEY`.
3. Set your key as an environment variable:
   ```bash
   # Windows PowerShell:
   $env:FIRMS_API_KEY="your_nasa_api_key_here"

   # macOS / Linux:
   export FIRMS_API_KEY="your_nasa_api_key_here"
   ```

### 2. Running `download_data.py` with API Key
Retrieve satellite fire observations for a specific region directly via CLI:
```bash
# Download 7 days of MODIS detections for USA
python scripts/download_data.py --source firms --country USA --days 7

# Download bounding box (California: [32, -125, 42, -114])
python scripts/download_data.py --source firms --bbox 32.0 -125.0 42.0 -114.0 --days 5

# Download UCI benchmark dataset
python scripts/download_data.py --source uci
```

### 3. Uploading Custom CSV via Dataset Explorer Page
1. Launch the Streamlit application and navigate to **1. Dataset Explorer** in the sidebar.
2. Expand the **📥 Import New Dataset (CSV / JSON)** container.
3. Browse and select your CSV file. Standard FIRMS columns (`latitude`, `longitude`, `acq_date`, `brightness`, `frp`, `confidence`) are automatically recognized.
4. Click **🚀 Process & Ingest Into Database**. The system validates coordinates, handles missing values, tags seasons, and registers the dataset.

---

## Dataset Format

When importing custom CSV files into the system, the columns below are recognized. Column names are case-insensitive and standard aliases (e.g., `lat` → `latitude`, `frp` → `fire_radiative_power`) are mapped automatically.

| Column Name | Data Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `latitude` | `Float` | **Yes** | Geographic latitude in decimal degrees (`[-90.0, 90.0]`) | `37.7749` |
| `longitude` | `Float` | **Yes** | Geographic longitude in decimal degrees (`[-180.0, 180.0]`) | `-122.4194` |
| `acquisition_date` | `Date / String` | Optional | Date of satellite overpass (`YYYY-MM-DD`). Defaults to current date if missing. | `2023-08-15` |
| `acquisition_time` | `String` | Optional | 4-digit UTC time string (`HHMM`). | `1345` |
| `brightness` | `Float` | Optional | Satellite channel brightness temperature in Kelvin (`[200.0, 600.0]`). | `342.5` |
| `confidence` | `Integer` | Optional | Detection confidence percentage (`[0, 100]`). Defaults to 50 if missing. | `85` |
| `fire_radiative_power` | `Float` | Optional | Radiative thermal power output in Megawatts (MW, `[0.0, 2000.0]`). | `48.2` |
| `satellite_source` | `String` | Optional | Platform sensor identifier (`MODIS-Terra`, `MODIS-Aqua`, `VIIRS-SNPP`, `VIIRS-NOAA20`). | `MODIS-Terra` |
| `daynight` | `String` | Optional | Overpass flag: `D` (Day) or `N` (Night). | `D` |
| `country` | `String` | Optional | Country or territorial region identifier. | `USA` |

---

## Machine Learning Modules

The system provides five dedicated machine learning modules (Modules A through E):

- **Module A: Spatial Hotspot Clustering (K-Means)**
  - Partitions wildfire coordinates into $K$ geographic cluster regions.
  - Generates automated inertia elbow curves across $K \in [2, 10]$ and silhouette scores to determine optimal spatial grouping.
  - Extracts cluster centroids and regional observation density counts.
- **Module B: Density-Based Spatial Outbreak Detection (DBSCAN)**
  - Utilizes spherical haversine distance metric ($\text{km}$) on angular radian coordinates.
  - Identifies contiguous fire perimeter zones of arbitrary geometric shape without predefining cluster numbers.
  - Automatically isolates spatial noise and low-density outliers from active fire fronts.
- **Module C: Unsupervised Thermal Anomaly Detection (Isolation Forest)**
  - Constructs isolation trees on composite multi-variable inputs (brightness, FRP, confidence, latitude, longitude).
  - Isolates intense megafires and unexpected spatial ignition patterns based on shallow path lengths.
- **Module D: Density-Based Outlier Profiling (Local Outlier Factor)**
  - Computes local reachability densities of fire events relative to their $k$-nearest neighbors.
  - Uncovers isolated ignitions that deviate significantly from neighboring density baselines.
- **Module E: Supervised Multi-Class Wildfire Risk Prediction**
  - Trains gradient boosting estimators (XGBoost, Random Forest, Gradient Boosting) using chronological train/test partitioning.
  - Classifies grid risk into 4 tiers: **Low**, **Moderate**, **High**, and **Critical**.
  - Computes weighted Accuracy, Precision, Recall, F1-Score, and Multi-Class One-vs-Rest ROC curves with full feature importance rankings.

---

## Streamlit Deployment

To deploy the application on **Streamlit Community Cloud**:

```
1. Push your code to GitHub (see GitHub Deployment below).
2. Navigate to https://share.streamlit.io and sign in with GitHub.
3. Click "New app".
4. Select your repository: YOUR_USERNAME/forest-fire-intelligence.
5. Set Branch: main.
6. Set Main file path: app.py.
7. Click "Deploy"!
```

---

## GitHub Deployment

To publish this project to GitHub from your local terminal:

```bash
# If using GitHub CLI:
gh repo create forest-fire-intelligence --public --source=. --push

# Or manually:
git init
git add .
git commit -m 'Initial commit: AI-Driven Forest Fire Intelligence System'
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/forest-fire-intelligence.git
git push -u origin main
```

---

## Environment Variables

| Variable | Required | Description | Example / Default |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Optional | Relational database connection URI. If omitted, uses local SQLite database file `forest_fire.db`. | `postgresql://user:pass@ep-cool-db.us-east-2.aws.neon.tech/neondb` |
| `FIRMS_API_KEY` | Optional | 32-character NASA FIRMS MAP_KEY required to download live satellite fire data. | `a1b2c3d4e5f67890123456789abcdef0` |
| `CDS_API_KEY` | Optional | Copernicus Climate Data Store API key for retrieving live ERA5 reanalysis fields. | `12345:abcdef67-8901-2345-6789-abcdef012345` |

---

## Streamlit Cloud Storage Limitation

> [!WARNING]
> **Ephemeral Storage on Streamlit Community Cloud**:
> Streamlit Community Cloud runs inside stateless ephemeral containers. Any local SQLite database file (`forest_fire.db`) stored on the local filesystem will reset whenever the application reboots, sleeps, or redeploys.
> 
> **Solution for Persistent Production Storage**:
> Supply a remote PostgreSQL connection URL via Streamlit Secrets:
> 1. In your Streamlit Cloud app settings, navigate to **Settings > Secrets**.
> 2. Add your database connection string:
>    ```toml
>    DATABASE_URL = "postgresql://user:password@host:5432/dbname"
>    ```
> Compatible free-tier hosted PostgreSQL providers include **Supabase**, **Neon.tech**, and **Railway**. When `DATABASE_URL` is configured, SQLAlchemy automatically connects to PostgreSQL with connection pooling.

---

## Running Tests

Run the comprehensive pytest suite covering database models, cleaning logic, feature engineering, and machine learning models:

```bash
# Run all unit tests with verbose reporting
pytest tests/ -v

# Run tests with code coverage analysis
pytest tests/ -v --cov=src --cov-report=term-missing

# Generate an HTML test coverage report in htmlcov/
pytest tests/ -v --cov=src --cov-report=html
```

---

## API Limitations & Disclaimers

> [!IMPORTANT]
> **Scientific & Emergency Management Disclaimers**:
> - **Historical Pattern Discovery**: This system is designed for historical observational analysis, retrospective hotspot discovery, and statistical risk modeling. It does **NOT** provide real-time tactical fire perimeters.
> - **Not an Emergency Notification System**: Model risk scores represent statistical correlations based on satellite telemetry and meteorological data. They must **NEVER** be used to direct emergency response personnel, alter evacuation routes, or replace official guidance from national wildfire authorities (e.g., USFS, CAL FIRE, NIFC, AFAC).
> - **Satellite Sensor Latency**: Near Real-Time (NRT) satellite observations from NASA FIRMS carry intrinsic orbital transit delays (typically 1 to 3 hours). Cloud coverage and heavy smoke plumes may obscure thermal detections.
> - **API Key Requirements**: Direct programmatic download of live FIRMS telemetry requires a free NASA MAP_KEY. Copernicus ERA5 meteorological retrieval requires a free CDS account and the `cdsapi` package.

---

## Future Enhancements

1. **Topographical GIS Integration**: Incorporate digital elevation models (DEM) to evaluate slope, aspect, and terrain steepness on fire spread.
2. **Satellite Spectral Indices**: Integrate Sentinel-2 and Landsat surface reflectance channels to compute Normalized Burn Ratio (NBR) and delta-NBR burn severity maps.
3. **Deep Learning Sequence Modeling**: Deploy Temporal Convolutional Networks (TCN) or LSTM architectures to forecast fire propagation fronts over 24-48 hour windows.
4. **Automated Webhook Alerts**: Configurable Slack, Telegram, and Email webhook notifications triggered when anomaly scores exceed critical thresholds.
5. **Vegetation Fuel Indexing**: Incorporate MODIS / VIIRS NDVI and live fuel moisture content (LFMC) dynamics into risk calculations.
6. **Live Wind Vector Overlays**: Stream vector wind fields (speed and bearing) into Folium maps to illustrate real-time plume propagation vectors.
7. **Multi-Region Cross-Validation**: Spatial block cross-validation to assess model transferability across diverse biomes (e.g., Mediterranean chaparral vs. boreal taiga).
8. **Automated Ingestion Cron Jobs**: Scheduled background Celery / GitHub Actions workers to ingest FIRMS satellite feeds on a daily automated cadence.
9. **Interactive Scenario Simulator**: Interactive sliders in the Streamlit UI enabling users to simulate the impact of rising temperatures or high winds on regional risk tiers.
10. **RESTful FastAPI Microservice**: Decouple the ML prediction engine into a dedicated high-throughput REST API with OpenAPI documentation and Docker containerization.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

---

## Acknowledgements

- **NASA FIRMS**: Fire Information for Resource Management System (MODIS & VIIRS instruments), operated by NASA's Earth Science Data and Information System (ESDIS).
- **UCI Machine Learning Repository**: Cortez and Morais (2007) Forest Fires Benchmark meteorological dataset.
- **Copernicus Climate Change Service (C3S)**: European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 Atmospheric Reanalysis.
