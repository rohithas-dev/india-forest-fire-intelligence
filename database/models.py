"""
SQLAlchemy ORM Models for the AI-Driven Forest Fire Intelligence System.

Tables:
    - datasets               : Imported dataset metadata
    - fire_events            : Satellite fire observations (NASA FIRMS format)
    - environmental_records  : Weather / environmental measurements
    - geographical_clusters  : Spatial clustering algorithm results
    - anomaly_records        : Detected fire activity anomalies
    - model_experiments      : ML model training experiment results
    - risk_predictions       : Fire risk assessment predictions
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Date,
    Text, ForeignKey, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


# ---------------------------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------------------------
class Dataset(Base):
    """Tracks imported datasets and their provenance metadata."""
    __tablename__ = "datasets"

    dataset_id   = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(255), nullable=False)
    source       = Column(String(255))          # e.g. NASA FIRMS, UCI, Copernicus
    upload_date  = Column(DateTime, default=datetime.utcnow)
    record_count = Column(Integer, default=0)
    date_range   = Column(String(100))          # e.g. "2010-01-01 to 2023-12-31"
    description  = Column(Text)
    is_demo      = Column(Integer, default=0)   # 1 if demonstration data

    # Relationships
    fire_events             = relationship("FireEvent",            back_populates="dataset", cascade="all, delete-orphan")
    environmental_records   = relationship("EnvironmentalRecord",  back_populates="dataset", cascade="all, delete-orphan")
    model_experiments       = relationship("ModelExperiment",      back_populates="dataset")

    def __repr__(self):
        return f"<Dataset id={self.dataset_id} name='{self.dataset_name}' records={self.record_count}>"


# ---------------------------------------------------------------------------
# 2. Fire Events
# ---------------------------------------------------------------------------
class FireEvent(Base):
    """
    Individual satellite fire detections in NASA FIRMS / MODIS / VIIRS format.

    Brightness is the brightness temperature (Kelvin).
    fire_radiative_power (FRP) is in megawatts (MW).
    confidence is 0–100 %.
    """
    __tablename__ = "fire_events"

    event_id             = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id           = Column(Integer, ForeignKey("datasets.dataset_id", ondelete="CASCADE"))
    latitude             = Column(Float, nullable=False)
    longitude            = Column(Float, nullable=False)
    acquisition_date     = Column(Date)
    acquisition_time     = Column(String(10))
    brightness           = Column(Float)         # Kelvin
    confidence           = Column(Integer)       # 0-100
    fire_radiative_power = Column(Float)         # MW
    satellite_source     = Column(String(50))    # Terra, Aqua, NOAA-20, Suomi-NPP
    daynight             = Column(String(1))     # D or N
    country              = Column(String(100))
    season               = Column(String(10))    # Spring, Summer, Fall, Winter

    # Relationships
    dataset   = relationship("Dataset", back_populates="fire_events")
    anomalies = relationship("AnomalyRecord", back_populates="fire_event", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_fire_location", "latitude", "longitude"),
        Index("idx_fire_date",     "acquisition_date"),
        Index("idx_fire_dataset",  "dataset_id"),
        Index("idx_fire_country",  "country"),
    )

    def __repr__(self):
        return (
            f"<FireEvent id={self.event_id} "
            f"lat={self.latitude} lon={self.longitude} "
            f"date={self.acquisition_date}>"
        )


# ---------------------------------------------------------------------------
# 3. Environmental Records
# ---------------------------------------------------------------------------
class EnvironmentalRecord(Base):
    """Historical weather / environmental observations."""
    __tablename__ = "environmental_records"

    record_id       = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id      = Column(Integer, ForeignKey("datasets.dataset_id", ondelete="CASCADE"))
    latitude        = Column(Float)
    longitude       = Column(Float)
    timestamp       = Column(DateTime)
    temperature     = Column(Float)   # Celsius
    humidity        = Column(Float)   # Percentage (0-100)
    wind_speed      = Column(Float)   # m/s
    wind_direction  = Column(Float)   # Degrees (0-360)
    precipitation   = Column(Float)   # mm

    dataset = relationship("Dataset", back_populates="environmental_records")

    __table_args__ = (
        Index("idx_env_location",  "latitude", "longitude"),
        Index("idx_env_timestamp", "timestamp"),
        Index("idx_env_dataset",   "dataset_id"),
    )

    def __repr__(self):
        return f"<EnvironmentalRecord id={self.record_id} temp={self.temperature}°C>"


# ---------------------------------------------------------------------------
# 4. Geographical Clusters
# ---------------------------------------------------------------------------
class GeographicalCluster(Base):
    """
    Results from spatial clustering algorithms (K-Means, DBSCAN).
    Each row represents one cluster centroid from a clustering run.
    """
    __tablename__ = "geographical_clusters"

    cluster_id          = Column(Integer, primary_key=True, autoincrement=True)
    algorithm           = Column(String(50), nullable=False)  # kmeans | dbscan
    centroid_latitude   = Column(Float)
    centroid_longitude  = Column(Float)
    observation_count   = Column(Integer)
    n_clusters          = Column(Integer)
    parameters_json     = Column(Text)   # JSON: algorithm hyperparameters
    created_at          = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_cluster_algo", "algorithm"),
        Index("idx_cluster_date", "created_at"),
    )

    def __repr__(self):
        return (
            f"<GeographicalCluster id={self.cluster_id} "
            f"algo='{self.algorithm}' obs={self.observation_count}>"
        )


# ---------------------------------------------------------------------------
# 5. Anomaly Records
# ---------------------------------------------------------------------------
class AnomalyRecord(Base):
    """Detected anomalies in fire activity from Isolation Forest / LOF."""
    __tablename__ = "anomaly_records"

    anomaly_id    = Column(Integer, primary_key=True, autoincrement=True)
    event_id      = Column(Integer, ForeignKey("fire_events.event_id", ondelete="SET NULL"), nullable=True)
    anomaly_score = Column(Float)       # Negative = more anomalous (Isolation Forest scale)
    anomaly_type  = Column(String(50))  # isolation_forest | lof
    latitude      = Column(Float)
    longitude     = Column(Float)
    brightness    = Column(Float)
    frp           = Column(Float)
    detected_at   = Column(DateTime, default=datetime.utcnow)

    fire_event = relationship("FireEvent", back_populates="anomalies")

    __table_args__ = (
        Index("idx_anomaly_score", "anomaly_score"),
        Index("idx_anomaly_type",  "anomaly_type"),
        Index("idx_anomaly_date",  "detected_at"),
    )

    def __repr__(self):
        return (
            f"<AnomalyRecord id={self.anomaly_id} "
            f"score={self.anomaly_score:.4f} type='{self.anomaly_type}'>"
        )


# ---------------------------------------------------------------------------
# 6. Model Experiments
# ---------------------------------------------------------------------------
class ModelExperiment(Base):
    """
    ML model training run results.
    Stores evaluation metrics for reproducibility and comparison.
    """
    __tablename__ = "model_experiments"

    experiment_id      = Column(Integer, primary_key=True, autoincrement=True)
    model_name         = Column(String(100), nullable=False)
    dataset_id         = Column(Integer, ForeignKey("datasets.dataset_id", ondelete="SET NULL"), nullable=True)
    accuracy           = Column(Float)
    precision_score    = Column(Float)
    recall             = Column(Float)
    f1_score           = Column(Float)
    roc_auc            = Column(Float)
    parameters         = Column(Text)           # JSON: hyperparameters used
    feature_importance = Column(Text)           # JSON: feature name -> importance score
    created_at         = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="model_experiments")

    __table_args__ = (
        Index("idx_exp_model", "model_name"),
        Index("idx_exp_date",  "created_at"),
    )

    def __repr__(self):
        return (
            f"<ModelExperiment id={self.experiment_id} "
            f"model='{self.model_name}' f1={self.f1_score:.4f}>"
        )


# ---------------------------------------------------------------------------
# 7. Risk Predictions
# ---------------------------------------------------------------------------
class RiskPrediction(Base):
    """
    Fire risk estimates for geographical grid cells.

    risk_level: 'Low' | 'Moderate' | 'High' | 'Critical'
    prediction_score: 0.0 (lowest risk) → 1.0 (highest risk)
    """
    __tablename__ = "risk_predictions"

    prediction_id          = Column(Integer, primary_key=True, autoincrement=True)
    latitude               = Column(Float, nullable=False)
    longitude              = Column(Float, nullable=False)
    risk_level             = Column(String(20))  # Low, Moderate, High, Critical
    prediction_score       = Column(Float)        # 0.0 – 1.0
    model_name             = Column(String(100))
    feature_importance_json = Column(Text)        # JSON: contributing features
    created_at             = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_risk_location", "latitude", "longitude"),
        Index("idx_risk_level",    "risk_level"),
        Index("idx_risk_date",     "created_at"),
    )

    def __repr__(self):
        return (
            f"<RiskPrediction id={self.prediction_id} "
            f"risk='{self.risk_level}' score={self.prediction_score:.3f}>"
        )
