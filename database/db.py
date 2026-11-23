"""
Database engine and session management.

Supports:
    - SQLite  (default, zero-config)
    - PostgreSQL  (via DATABASE_URL environment variable)

Usage:
    from database.db import get_db_session

    with get_db_session() as session:
        events = session.query(FireEvent).all()
"""
import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# ─── Lazy singletons ────────────────────────────────────────────────────────
_engine = None
_SessionFactory = None


# ─── URL resolution ─────────────────────────────────────────────────────────
def get_database_url() -> str:
    """
    Determine the database URL.

    Priority order:
        1. Streamlit secrets  st.secrets["DATABASE_URL"]  (Streamlit Cloud deployment)
        2. DATABASE_URL  environment variable  (Docker / local .env)
        3. Default SQLite file  forest_fire.db  in the project root.
    """
    # 1. Try Streamlit secrets (available on Streamlit Cloud)
    try:
        import streamlit as st
        db_url = st.secrets.get("DATABASE_URL", "")
        if db_url:
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.info("Using PostgreSQL from Streamlit secrets.")
            return db_url
    except Exception:
        pass

    # 2. Try environment variable
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        logger.info("Using PostgreSQL from DATABASE_URL environment variable.")
        return db_url

    # 3. Local SQLite fallback
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "forest_fire.db")
    logger.info(f"Using SQLite database at: {db_path}")
    return f"sqlite:///{db_path}"


# ─── Engine factory ─────────────────────────────────────────────────────────
def _create_engine_from_url(url: str):
    """Create a SQLAlchemy engine with settings appropriate for the dialect."""
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        # Enable WAL mode for concurrent reads
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine = create_engine(url, pool_pre_ping=True, echo=False)

    return engine


# ─── Singleton accessors ────────────────────────────────────────────────────
def get_engine_instance():
    """Return the application-wide database engine (created once)."""
    global _engine
    if _engine is None:
        _engine = _create_engine_from_url(get_database_url())
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the application-wide session factory (created once)."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine_instance(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionFactory


# ─── Session context manager ─────────────────────────────────────────────────
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager that provides a transactional database session.

    Commits on successful exit, rolls back on exception.

    Example::

        with get_db_session() as session:
            session.add(some_object)
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── Health check ───────────────────────────────────────────────────────────
def test_connection() -> bool:
    """
    Test database connectivity.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        engine = get_engine_instance()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        return False


def reset_engine():
    """
    Force recreation of the engine and session factory.
    Useful after changing DATABASE_URL at runtime (e.g., in tests).
    """
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
