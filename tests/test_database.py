"""
Unit tests for database models, schema creation, transactions, and foreign key cascades.
"""
from datetime import date, datetime
import pytest
from sqlalchemy import inspect

from database.db import test_connection
from database.models import (
    AnomalyRecord,
    Dataset,
    EnvironmentalRecord,
    FireEvent,
    GeographicalCluster,
    ModelExperiment,
    RiskPrediction,
)
from database.schema import get_existing_table_names


def test_schema_creation(in_memory_db) -> None:
    """Verify all 7 expected ORM tables are created in the database."""
    expected_tables = {
        "datasets",
        "fire_events",
        "environmental_records",
        "geographical_clusters",
        "anomaly_records",
        "model_experiments",
        "risk_predictions",
    }
    existing_tables = set(get_existing_table_names())
    assert expected_tables.issubset(existing_tables), (
        f"Missing tables: {expected_tables - existing_tables}"
    )


def test_dataset_insert_and_query(in_memory_db) -> None:
    """Insert a Dataset record, query it back, and verify all stored field values."""
    session_factory = in_memory_db
    with session_factory() as session:
        ds = Dataset(
            dataset_name="California Test Fire 2021",
            source="DEMO_DATA",
            upload_date=datetime(2021, 8, 1, 12, 0, 0),
            record_count=150,
            date_range="2021-08-01 to 2021-08-15",
            description="Test synthetic wildfire detections",
            is_demo=1,
        )
        session.add(ds)
        session.commit()
        ds_id = ds.dataset_id

    with session_factory() as session:
        queried = session.query(Dataset).filter_by(dataset_id=ds_id).first()
        assert queried is not None
        assert queried.dataset_name == "California Test Fire 2021"
        assert queried.source == "DEMO_DATA"
        assert queried.record_count == 150
        assert queried.is_demo == 1
        assert "2021-08-01" in queried.date_range


def test_fire_event_insert_and_query(in_memory_db) -> None:
    """Insert a Dataset parent and 10 related FireEvents, query back, and verify count."""
    session_factory = in_memory_db
    with session_factory() as session:
        ds = Dataset(dataset_name="MODIS Feed", source="DEMO_DATA", is_demo=1)
        session.add(ds)
        session.flush()

        events = [
            FireEvent(
                dataset_id=ds.dataset_id,
                latitude=38.5 + i * 0.05,
                longitude=-120.5 - i * 0.05,
                acquisition_date=date(2021, 7, 10),
                acquisition_time=f"{1200 + i:04d}",
                brightness=330.0 + i,
                confidence=85,
                fire_radiative_power=25.0 + i * 2,
                satellite_source="MODIS-Terra",
                daynight="D",
                country="USA",
                season="Summer",
            )
            for i in range(10)
        ]
        session.add_all(events)
        session.commit()
        saved_ds_id = ds.dataset_id

    with session_factory() as session:
        queried_events = (
            session.query(FireEvent).filter_by(dataset_id=saved_ds_id).all()
        )
        assert len(queried_events) == 10
        assert all(e.country == "USA" for e in queried_events)
        assert queried_events[0].satellite_source == "MODIS-Terra"


def test_environmental_record_crud(in_memory_db) -> None:
    """Verify Create, Read, Update, and Delete operations on EnvironmentalRecord and Dataset."""
    session_factory = in_memory_db
    with session_factory() as session:
        ds = Dataset(dataset_name="Weather Station Data", source="Copernicus", record_count=1)
        session.add(ds)
        session.flush()

        rec = EnvironmentalRecord(
            dataset_id=ds.dataset_id,
            latitude=40.123,
            longitude=-105.456,
            timestamp=datetime(2022, 6, 15, 14, 0),
            temperature=32.5,
            humidity=18.0,
            wind_speed=8.5,
            wind_direction=240.0,
            precipitation=0.0,
        )
        session.add(rec)
        session.commit()
        rec_id = rec.record_id
        ds_id = ds.dataset_id

    # Read and Update
    with session_factory() as session:
        queried_rec = session.query(EnvironmentalRecord).filter_by(record_id=rec_id).first()
        assert queried_rec is not None
        assert queried_rec.temperature == 32.5

        # Update parent record_count and record temperature
        queried_rec.temperature = 35.0
        parent_ds = session.query(Dataset).filter_by(dataset_id=ds_id).first()
        parent_ds.record_count = 5
        session.commit()

    # Verify update
    with session_factory() as session:
        updated_rec = session.query(EnvironmentalRecord).filter_by(record_id=rec_id).first()
        parent_ds = session.query(Dataset).filter_by(dataset_id=ds_id).first()
        assert updated_rec.temperature == 35.0
        assert parent_ds.record_count == 5


def test_cluster_insert(in_memory_db) -> None:
    """Insert a GeographicalCluster record and query it back."""
    session_factory = in_memory_db
    with session_factory() as session:
        cluster = GeographicalCluster(
            algorithm="kmeans",
            centroid_latitude=37.7749,
            centroid_longitude=-122.4194,
            observation_count=84,
            n_clusters=5,
            parameters_json='{"n_clusters": 5, "random_state": 42}',
            created_at=datetime.utcnow(),
        )
        session.add(cluster)
        session.commit()
        cid = cluster.cluster_id

    with session_factory() as session:
        saved = session.query(GeographicalCluster).filter_by(cluster_id=cid).first()
        assert saved is not None
        assert saved.algorithm == "kmeans"
        assert saved.observation_count == 84
        assert saved.n_clusters == 5
        assert "random_state" in saved.parameters_json


def test_anomaly_record_insert(in_memory_db) -> None:
    """Insert an AnomalyRecord with null event_id and verify retrieval."""
    session_factory = in_memory_db
    with session_factory() as session:
        anomaly = AnomalyRecord(
            event_id=None,
            anomaly_score=-0.2845,
            anomaly_type="isolation_forest",
            latitude=42.15,
            longitude=-121.35,
            brightness=445.0,
            frp=280.0,
            detected_at=datetime.utcnow(),
        )
        session.add(anomaly)
        session.commit()
        aid = anomaly.anomaly_id

    with session_factory() as session:
        saved = session.query(AnomalyRecord).filter_by(anomaly_id=aid).first()
        assert saved is not None
        assert saved.event_id is None
        assert saved.anomaly_score < 0
        assert saved.anomaly_type == "isolation_forest"
        assert saved.brightness == 445.0


def test_model_experiment_insert(in_memory_db) -> None:
    """Insert a ModelExperiment and verify metrics and hyperparameters."""
    session_factory = in_memory_db
    with session_factory() as session:
        exp = ModelExperiment(
            model_name="XGBOOST",
            dataset_id=None,
            accuracy=0.885,
            precision_score=0.872,
            recall=0.890,
            f1_score=0.881,
            roc_auc=0.945,
            parameters='{"max_depth": 5, "n_estimators": 120}',
            feature_importance='{"brightness": 0.42, "fire_radiative_power": 0.38}',
            created_at=datetime.utcnow(),
        )
        session.add(exp)
        session.commit()
        eid = exp.experiment_id

    with session_factory() as session:
        saved = session.query(ModelExperiment).filter_by(experiment_id=eid).first()
        assert saved is not None
        assert saved.model_name == "XGBOOST"
        assert saved.accuracy == pytest.approx(0.885, 0.001)
        assert saved.f1_score == pytest.approx(0.881, 0.001)
        assert saved.roc_auc == pytest.approx(0.945, 0.001)
        assert "brightness" in saved.feature_importance


def test_risk_prediction_insert(in_memory_db) -> None:
    """Insert a RiskPrediction and verify location and risk attributes."""
    session_factory = in_memory_db
    with session_factory() as session:
        pred = RiskPrediction(
            latitude=39.5,
            longitude=-120.2,
            risk_level="High",
            prediction_score=0.72,
            model_name="XGBOOST",
            feature_importance_json='{"brightness": 0.35, "humidity": 0.25}',
            created_at=datetime.utcnow(),
        )
        session.add(pred)
        session.commit()
        pid = pred.prediction_id

    with session_factory() as session:
        saved = session.query(RiskPrediction).filter_by(prediction_id=pid).first()
        assert saved is not None
        assert saved.risk_level == "High"
        assert saved.prediction_score == pytest.approx(0.72, 0.01)
        assert saved.model_name == "XGBOOST"


def test_cascade_delete(in_memory_db) -> None:
    """Insert a Dataset with FireEvents, delete the Dataset, and verify FireEvents are deleted."""
    session_factory = in_memory_db
    with session_factory() as session:
        ds = Dataset(dataset_name="Cascade Test Set", source="DEMO_DATA")
        session.add(ds)
        session.flush()

        events = [
            FireEvent(
                dataset_id=ds.dataset_id,
                latitude=35.0 + i,
                longitude=-110.0 - i,
                acquisition_date=date(2021, 5, 1),
                confidence=80,
            )
            for i in range(5)
        ]
        session.add_all(events)
        session.commit()
        ds_id = ds.dataset_id

    # Verify rows exist
    with session_factory() as session:
        assert session.query(FireEvent).filter_by(dataset_id=ds_id).count() == 5

        # Delete parent Dataset
        ds_to_delete = session.query(Dataset).filter_by(dataset_id=ds_id).first()
        session.delete(ds_to_delete)
        session.commit()

    # Verify children cascade deleted
    with session_factory() as session:
        remaining_events = session.query(FireEvent).filter_by(dataset_id=ds_id).count()
        assert remaining_events == 0, f"Expected 0 cascade-deleted events, found {remaining_events}"


def test_connection_function(in_memory_db) -> None:
    """Verify test_connection() executes successfully with in-memory database."""
    assert test_connection() is True
