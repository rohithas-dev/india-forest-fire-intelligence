"""
UI helper utilities, CSS injection, and dark-theme configurations
for the AI-Driven Forest Fire Intelligence System.
"""
from datetime import datetime, date
import os
import streamlit as st
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from sqlalchemy import func, desc

from database.db import get_db_session, get_database_url
from database.models import FireEvent, Dataset, EnvironmentalRecord, GeographicalCluster, AnomalyRecord, ModelExperiment, RiskPrediction


# -----------------------------------------------------------------------------
# Color Palette Constants
# -----------------------------------------------------------------------------
COLOR_BG = "#0A0E1A"
COLOR_SECONDARY = "#141928"
COLOR_ACCENT = "#FF6B35"      # Fire Orange
COLOR_GOLD = "#FFD700"        # Warning / Gold
COLOR_TEXT = "#E8EAF0"
COLOR_MUTED = "#8C9BAE"
COLOR_BORDER = "rgba(255, 107, 53, 0.25)"
COLOR_SUCCESS = "#00C9A7"
COLOR_DANGER = "#FF4B4B"

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLOR_SECONDARY,
    plot_bgcolor=COLOR_SECONDARY,
    font=dict(family="Inter, sans-serif", color=COLOR_TEXT, size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.08)",
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color=COLOR_MUTED),
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.08)",
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color=COLOR_MUTED),
    ),
    legend=dict(
        bgcolor="rgba(20, 25, 40, 0.8)",
        bordercolor=COLOR_BORDER,
        font=dict(color=COLOR_TEXT),
    )
)


def apply_dark_theme(fig):
    """Apply consistent dark fire theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    return fig


# -----------------------------------------------------------------------------
# CSS Injection
# -----------------------------------------------------------------------------
def inject_css() -> None:
    """Inject unified professional dark fire theme CSS into the Streamlit app."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}

    /* Global sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_SECONDARY};
        border-right: 1px solid rgba(255, 107, 53, 0.15);
    }}

    /* Hero Banner */
    .hero-banner {{
        background: linear-gradient(135deg, #141928 0%, #20131B 45%, #2F110B 100%);
        border: 1px solid rgba(255, 107, 53, 0.35);
        border-left: 6px solid {COLOR_ACCENT};
        padding: 26px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }}
    .hero-title {{
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }}
    .hero-subtitle {{
        color: {COLOR_ACCENT};
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 10px;
        letter-spacing: 0.2px;
    }}
    .hero-desc {{
        color: #CBD5E1;
        font-size: 0.95rem;
        line-height: 1.55;
        margin: 0;
    }}

    /* Metric Cards */
    .metric-card {{
        background-color: {COLOR_SECONDARY};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        border-color: {COLOR_ACCENT};
        box-shadow: 0 8px 24px rgba(255, 107, 53, 0.22);
    }}
    .metric-label {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {COLOR_MUTED};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {COLOR_ACCENT};
        line-height: 1.15;
        margin: 2px 0 6px 0;
    }}
    .metric-sub {{
        font-size: 0.82rem;
        color: {COLOR_GOLD};
        font-weight: 500;
    }}

    /* Navigation Card */
    .nav-card {{
        background-color: {COLOR_SECONDARY};
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px 24px;
        height: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }}
    .nav-card:hover {{
        border-color: {COLOR_ACCENT};
        transform: translateY(-4px);
        box-shadow: 0 10px 28px rgba(255, 107, 53, 0.2);
    }}
    .nav-icon {{
        font-size: 2.2rem;
        margin-bottom: 10px;
    }}
    .nav-title {{
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    .nav-desc {{
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.45;
    }}

    /* Notice & Warning Boxes */
    .fire-alert-box {{
        background: rgba(255, 107, 53, 0.08);
        border: 1px solid rgba(255, 107, 53, 0.4);
        border-left: 5px solid {COLOR_ACCENT};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #F1F5F9;
    }}
    .fire-warning-box {{
        background: rgba(255, 215, 0, 0.08);
        border: 1px solid rgba(255, 215, 0, 0.4);
        border-left: 5px solid {COLOR_GOLD};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #FFFBEB;
    }}
    .fire-info-box {{
        background: rgba(0, 201, 167, 0.08);
        border: 1px solid rgba(0, 201, 167, 0.35);
        border-left: 5px solid {COLOR_SUCCESS};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #F1F5F9;
    }}

    /* Streamlit widgets styling enhancements */
    .stButton>button {{
        background: linear-gradient(135deg, #FF6B35 0%, #D34516 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, #FF7F4D 0%, #E65220 100%);
        box-shadow: 0 4px 14px rgba(255, 107, 53, 0.4);
        transform: translateY(-1px);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, description: str) -> None:
    """Render the standard fire intelligence hero banner."""
    html = f"""
    <div class="hero-banner">
        <h1 class="hero-title">{title}</h1>
        <div class="hero-subtitle">{subtitle}</div>
        <p class="hero-desc">{description}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: Any, subtext: Optional[str] = None) -> str:
    """Return HTML markup for a styled KPI metric card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """


# -----------------------------------------------------------------------------
# Global Sidebar Filters
# -----------------------------------------------------------------------------
def render_sidebar_filters() -> Dict[str, Any]:
    """
    Render global sidebar filters common to all pages.
    Returns dictionary with selected filter values.
    """
    st.sidebar.markdown("### 🔍 Global Filters")

    # Load available datasets
    datasets = get_all_datasets()
    dataset_options = {"All Datasets": None}
    for d in datasets:
        dataset_options[f"{d.dataset_name} ({d.record_count:,} records)"] = d.dataset_id

    selected_dataset_label = st.sidebar.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        index=0,
        help="Filter data by imported satellite / simulated dataset"
    )
    selected_dataset_id = dataset_options[selected_dataset_label]

    # Date range filters
    min_date, max_date = get_date_bounds(selected_dataset_id)
    if min_date and max_date and min_date != max_date:
        selected_dates = st.sidebar.date_input(
            "Acquisition Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter fire events by satellite detection date"
        )
        if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
            start_date, end_date = selected_dates[0], selected_dates[1]
        elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
            start_date = end_date = selected_dates[0]
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = min_date
        end_date = max_date

    st.sidebar.markdown("---")
    # System Info in Sidebar
    db_url = get_database_url()
    db_type = "PostgreSQL" if "postgresql" in db_url.lower() else "SQLite"
    st.sidebar.markdown(f"**Database:** `{db_type}`")
    st.sidebar.markdown(f"**Last Sync:** `{datetime.now().strftime('%Y-%m-%d %H:%M')}`")

    return {
        "dataset_id": selected_dataset_id,
        "start_date": start_date,
        "end_date": end_date,
        "dataset_label": selected_dataset_label
    }


# -----------------------------------------------------------------------------
# Database Cached Helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_all_datasets() -> List[Any]:
    """Retrieve all dataset records from the database."""
    try:
        with get_db_session() as session:
            records = session.query(Dataset).order_by(Dataset.upload_date.desc()).all()
            return records
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_database_counts() -> Dict[str, int]:
    """Retrieve total counts for the 4 primary KPI indicators."""
    counts = {
        "fire_events": 0,
        "datasets": 0,
        "anomalies": 0,
        "clusters": 0
    }
    try:
        with get_db_session() as session:
            counts["fire_events"] = session.query(func.count(FireEvent.event_id)).scalar() or 0
            counts["datasets"] = session.query(func.count(Dataset.dataset_id)).scalar() or 0
            counts["anomalies"] = session.query(func.count(AnomalyRecord.anomaly_id)).scalar() or 0
            counts["clusters"] = session.query(func.count(GeographicalCluster.cluster_id)).scalar() or 0
    except Exception:
        pass
    return counts


@st.cache_data(ttl=60)
def get_date_bounds(dataset_id: Optional[int] = None) -> Tuple[Optional[date], Optional[date]]:
    """Get minimum and maximum acquisition dates for fire events."""
    try:
        with get_db_session() as session:
            q = session.query(
                func.min(FireEvent.acquisition_date),
                func.max(FireEvent.acquisition_date)
            )
            if dataset_id is not None:
                q = q.filter(FireEvent.dataset_id == dataset_id)
            min_d, max_d = q.first()
            return min_d, max_d
    except Exception:
        return None, None


@st.cache_data(ttl=60)
def load_fire_events_df(dataset_id: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """Load fire events into a pandas DataFrame."""
    try:
        with get_db_session() as session:
            query = session.query(FireEvent)
            if dataset_id is not None:
                query = query.filter(FireEvent.dataset_id == dataset_id)
            if limit:
                query = query.limit(limit)
            events = query.all()

            if not events:
                return pd.DataFrame()

            data = [{
                "event_id": e.event_id,
                "dataset_id": e.dataset_id,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "acquisition_date": e.acquisition_date,
                "acquisition_time": e.acquisition_time,
                "brightness": e.brightness,
                "confidence": e.confidence,
                "fire_radiative_power": e.fire_radiative_power,
                "satellite_source": e.satellite_source,
                "daynight": e.daynight,
                "country": e.country,
                "season": e.season
            } for e in events]
            df = pd.DataFrame(data)
            if "acquisition_date" in df.columns:
                df["acquisition_date"] = pd.to_datetime(df["acquisition_date"])
            return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_environmental_df(dataset_id: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """Load environmental records into a pandas DataFrame."""
    try:
        with get_db_session() as session:
            query = session.query(EnvironmentalRecord)
            if dataset_id is not None:
                query = query.filter(EnvironmentalRecord.dataset_id == dataset_id)
            if limit:
                query = query.limit(limit)
            records = query.all()

            if not records:
                return pd.DataFrame()

            data = [{
                "record_id": r.record_id,
                "dataset_id": r.dataset_id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "timestamp": r.timestamp,
                "temperature": r.temperature,
                "humidity": r.humidity,
                "wind_speed": r.wind_speed,
                "wind_direction": r.wind_direction,
                "precipitation": r.precipitation
            } for r in records]
            df = pd.DataFrame(data)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    except Exception:
        return pd.DataFrame()
