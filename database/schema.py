"""
Database schema management — table creation and initialization.
"""
import logging
from database.db import get_engine_instance
from database.models import Base

logger = logging.getLogger(__name__)


def create_all_tables() -> bool:
    """
    Create all ORM-defined tables in the database if they do not already exist.

    Returns:
        True on success, raises on failure.
    """
    engine = get_engine_instance()
    try:
        Base.metadata.create_all(engine, checkfirst=True)
        logger.info("All database tables created (or already exist).")
        return True
    except Exception as exc:
        logger.error(f"Failed to create tables: {exc}")
        raise


def drop_all_tables() -> None:
    """
    Drop ALL tables from the database.

    .. warning::
        This permanently deletes all data. Use only in tests or development.
    """
    engine = get_engine_instance()
    Base.metadata.drop_all(engine)
    logger.warning("All database tables have been dropped.")


def initialize_schema() -> bool:
    """
    Ensure the database schema exists.
    Safe to call multiple times (idempotent).

    Returns:
        True if schema is ready.
    """
    return create_all_tables()


def get_existing_table_names() -> list[str]:
    """Return names of tables that currently exist in the database."""
    from sqlalchemy import inspect
    engine = get_engine_instance()
    inspector = inspect(engine)
    return inspector.get_table_names()
