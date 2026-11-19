"""
Dataset Explorer page for importing, inspecting, profiling, and exporting wildfire datasets.
"""
from datetime import datetime, date
import json
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func, desc

from database.db import get_db_session
from database.models import FireEvent, Dataset, EnvironmentalRecord
from database.schema import initialize_schema
from src.utils.ui import (
    inject_css,
    apply_dark_theme,
    COLOR_ACCENT,
    COLOR_GOLD,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_SECONDARY,
    get_all_datasets
)

try:
    from src.preprocessing.cleaner import DataCleaner
except ImportError:
    class DataCleaner:
        @staticmethod
        def clean_fire_events(df):
            return df, {"initial_rows": len(df), "final_rows": len(df)}
        @staticmethod
        def clean_environmental_records(df):
            return df, {"initial_rows": len(df), "final_rows": len(df)}

try:
    from src.data_collection.nasa_firms import load_firms_csv
except ImportError:
    def load_firms_csv(f):
        return pd.read_csv(f)

# -----------------------------------------------------------------------------
# Page Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dataset Explorer — Forest Fire Intelligence",
    layout="wide",
    page_icon="📊"
)
initialize_schema()
inject_css()

st.title("📊 Dataset Explorer")
st.caption("Import, inspect, clean, and profile satellite thermal data and environmental observations.")

# -----------------------------------------------------------------------------
# Section 1: UPLOAD SECTION
# -----------------------------------------------------------------------------
with st.expander("📥 Import New Dataset (CSV / JSON)", expanded=False):
    col_u1, col_u2 = st.columns([2, 1])

    with col_u1:
        uploaded_file = st.file_uploader(
            "Choose a wildfire or meteorological file",
            type=["csv", "json"],
            help="Supports NASA FIRMS MODIS/VIIRS thermal CSV or JSON catalogs."
        )

    with col_u2:
        dataset_source = st.selectbox(
            "Data Source",
            options=["NASA FIRMS (MODIS)", "NASA FIRMS (VIIRS)", "Copernicus CEMS", "Simulated / DEMO_DATA", "Other"],
            index=0
        )
        custom_name = st.text_input(
            "Dataset Label",
            value="",
            placeholder="e.g., California August 2023 Thermal Feed"
        )
        is_demo_flag = 1 if "DEMO" in dataset_source else 0

    if uploaded_file is not None:
        process_btn = st.button("🚀 Process & Ingest Into Database", type="primary")
        if process_btn:
            try:
                # 1. Parse into DataFrame
                if uploaded_file.name.endswith(".json"):
                    raw_df = pd.read_json(uploaded_file)
                else:
                    raw_df = load_firms_csv(uploaded_file)

                # 2. Detect schema
                cols_lower = [str(c).lower() for c in raw_df.columns]
                is_env = any(k in cols_lower for k in ["temperature", "humidity", "wind_speed"]) and not any(k in cols_lower for k in ["brightness", "frp", "confidence"])

                ds_label = custom_name.strip() if custom_name.strip() else uploaded_file.name

                with st.spinner("Cleaning and validating data records..."):
                    if is_env:
                        clean_df, report = DataCleaner.clean_environmental_records(raw_df)
                        record_type = "environmental"
                    else:
                        clean_df, report = DataCleaner.clean_fire_events(raw_df)
                        record_type = "fire_events"

                if clean_df.empty:
                    st.error("No valid records found after data cleaning. Please verify file schema.")
                else:
                    # 3. Determine date range
                    date_range_str = "Unknown"
                    if "acquisition_date" in clean_df.columns:
                        d_min = clean_df["acquisition_date"].min()
                        d_max = clean_df["acquisition_date"].max()
                        date_range_str = f"{d_min} to {d_max}"
                    elif "timestamp" in clean_df.columns:
                        d_min = clean_df["timestamp"].min()
                        d_max = clean_df["timestamp"].max()
                        date_range_str = f"{d_min.date()} to {d_max.date()}"

                    # 4. Insert into DB
                    with get_db_session() as session:
                        new_ds = Dataset(
                            dataset_name=ds_label,
                            source=dataset_source,
                            upload_date=datetime.utcnow(),
                            record_count=len(clean_df),
                            date_range=date_range_str,
                            description=f"Imported from {uploaded_file.name}. Valid rows: {len(clean_df)}",
                            is_demo=is_demo_flag
                        )
                        session.add(new_ds)
                        session.flush()

                        # Bulk insert
                        if record_type == "fire_events":
                            objects = []
                            for _, row in clean_df.iterrows():
                                objects.append(FireEvent(
                                    dataset_id=new_ds.dataset_id,
                                    latitude=float(row["latitude"]),
                                    longitude=float(row["longitude"]),
                                    acquisition_date=row.get("acquisition_date"),
                                    acquisition_time=str(row.get("acquisition_time", "1200")),
                                    brightness=float(row["brightness"]) if pd.notna(row.get("brightness")) else None,
                                    confidence=int(row["confidence"]) if pd.notna(row.get("confidence")) else None,
                                    fire_radiative_power=float(row["fire_radiative_power"]) if pd.notna(row.get("fire_radiative_power")) else None,
                                    satellite_source=str(row.get("satellite_source", "Unknown")),
                                    daynight=str(row.get("daynight", "D")),
                                    country=str(row.get("country", "Global")),
                                    season=str(row.get("season", "Summer"))
                                ))
                            session.bulk_save_objects(objects)
                        else:
                            objects = []
                            for _, row in clean_df.iterrows():
                                objects.append(EnvironmentalRecord(
                                    dataset_id=new_ds.dataset_id,
                                    latitude=float(row.get("latitude", 0.0)),
                                    longitude=float(row.get("longitude", 0.0)),
                                    timestamp=row.get("timestamp"),
                                    temperature=float(row["temperature"]) if pd.notna(row.get("temperature")) else None,
                                    humidity=float(row["humidity"]) if pd.notna(row.get("humidity")) else None,
                                    wind_speed=float(row["wind_speed"]) if pd.notna(row.get("wind_speed")) else None,
                                    wind_direction=float(row.get("wind_direction", 0.0)),
                                    precipitation=float(row["precipitation"]) if pd.notna(row.get("precipitation")) else None
                                ))
                            session.bulk_save_objects(objects)

                    st.success(f"🎉 Successfully ingested dataset '{ds_label}' with {len(clean_df):,} records!")
                    st.info(
                        f"**Cleaning Report:** Initial rows: `{report.get('initial_rows', 0):,}` | "
                        f"Dropped duplicates: `{report.get('dropped_duplicates', 0):,}` | "
                        f"Invalid coords removed: `{report.get('dropped_invalid_coords', 0):,}` | "
                        f"Final stored rows: `{report.get('final_rows', 0):,}`"
                    )
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"Error importing dataset: {str(e)}")

# -----------------------------------------------------------------------------
# Section 2: DATASETS TABLE
# -----------------------------------------------------------------------------
st.markdown("### 🗄️ Managed Datasets")

all_datasets = get_all_datasets()

if not all_datasets:
    st.info("No datasets currently registered in database. Use the import panel above to add one.")
else:
    table_data = []
    for d in all_datasets:
        table_data.append({
            "ID":           d["dataset_id"],
            "Name":         d["dataset_name"],
            "Source":       d["source"],
            "Upload Date":  d["upload_date"].strftime("%Y-%m-%d %H:%M") if d["upload_date"] else "N/A",
            "Record Count": f"{d['record_count']:,}",
            "Date Range":   d["date_range"] or "N/A",
            "Is Demo":      "Yes (Simulated)" if d["is_demo"] == 1 else "Real Source"
        })
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Dataset Deletion Controller
    with st.expander("🗑️ Dataset Management / Deletion"):
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            ds_to_del_map = {
                f"#{d['dataset_id']} - {d['dataset_name']} ({d['record_count']:,} records)": d["dataset_id"]
                for d in all_datasets
            }
            target_del_label = st.selectbox("Select dataset to delete", options=list(ds_to_del_map.keys()))
            target_del_id = ds_to_del_map[target_del_label]
        with del_col2:
            confirm_del = st.checkbox("Confirm permanent deletion", key="confirm_del")
            if st.button("Delete Dataset", type="secondary", disabled=not confirm_del):
                try:
                    with get_db_session() as session:
                        ds_obj = session.query(Dataset).filter(Dataset.dataset_id == target_del_id).first()
                        if ds_obj:
                            session.delete(ds_obj)
                    st.success(f"Dataset #{target_del_id} deleted successfully.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as ex:
                    st.error(f"Deletion failed: {ex}")

# -----------------------------------------------------------------------------
# Section 3 & 4: DATA PREVIEW & QUALITY PANEL
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Dataset Profiling & Quality Inspection")

if not all_datasets:
    st.warning("Please upload or initialize a dataset to inspect rows and quality metrics.")
else:
    ds_select_map = {
        f"#{d['dataset_id']}: {d['dataset_name']}": d["dataset_id"]
        for d in all_datasets
    }
    selected_inspect_name = st.selectbox("Select Dataset to Profile", options=list(ds_select_map.keys()))
    selected_inspect_id = ds_select_map[selected_inspect_name]

    # Load fire events or environmental records
    with get_db_session() as session:
        events = session.query(FireEvent).filter(FireEvent.dataset_id == selected_inspect_id).limit(2000).all()
        if events:
            inspect_df = pd.DataFrame([{
                "event_id": e.event_id,
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
            } for e in events])
            if "acquisition_date" in inspect_df.columns:
                inspect_df["acquisition_date"] = pd.to_datetime(inspect_df["acquisition_date"])
        else:
            # Check environmental records
            envs = session.query(EnvironmentalRecord).filter(EnvironmentalRecord.dataset_id == selected_inspect_id).limit(2000).all()
            inspect_df = pd.DataFrame([{
                "record_id": env.record_id,
                "latitude": env.latitude,
                "longitude": env.longitude,
                "timestamp": env.timestamp,
                "temperature": env.temperature,
                "humidity": env.humidity,
                "wind_speed": env.wind_speed,
                "wind_direction": env.wind_direction,
                "precipitation": env.precipitation
            } for env in envs])

    if inspect_df.empty:
        st.info("The selected dataset currently contains 0 records.")
    else:
        st.markdown(f"**First 100 Records Preview** *(Showing sample of {len(inspect_df):,} loaded rows)*:")
        st.dataframe(inspect_df.head(100), use_container_width=True)

        st.markdown("#### 📊 Data Quality & Distribution Profiling")
        q_tab1, q_tab2, q_tab3 = st.tabs(["Missing Values", "Numerical Statistics", "Feature Distributions"])

        with q_tab1:
            missing_counts = inspect_df.isnull().sum().reset_index()
            missing_counts.columns = ["Column", "Missing Count"]
            missing_counts["Missing Pct"] = (missing_counts["Missing Count"] / len(inspect_df)) * 100

            fig_missing = px.bar(
                missing_counts,
                x="Missing Pct",
                y="Column",
                orientation="h",
                text="Missing Pct",
                labels={"Missing Pct": "Missing Percentage (%)", "Column": "Attribute"},
                title="Missing Data Proportion Per Field"
            )
            fig_missing.update_traces(
                marker_color=COLOR_ACCENT,
                texttemplate="%{x:.1f}%",
                textposition="outside"
            )
            apply_dark_theme(fig_missing)
            fig_missing.update_layout(height=350)
            st.plotly_chart(fig_missing, use_container_width=True)

        with q_tab2:
            st.markdown("Numerical Summaries (Descriptive Statistics):")
            st.dataframe(inspect_df.describe().T, use_container_width=True)

        with q_tab3:
            dist_cols = st.columns(2)
            if "brightness" in inspect_df.columns and inspect_df["brightness"].notna().sum() > 0:
                with dist_cols[0]:
                    fig_b = px.histogram(
                        inspect_df,
                        x="brightness",
                        nbins=30,
                        title="Thermal Brightness Temperature (Kelvin) Distribution",
                        color_discrete_sequence=[COLOR_ACCENT]
                    )
                    apply_dark_theme(fig_b)
                    fig_b.update_layout(height=320)
                    st.plotly_chart(fig_b, use_container_width=True)

            if "fire_radiative_power" in inspect_df.columns and inspect_df["fire_radiative_power"].notna().sum() > 0:
                with dist_cols[1]:
                    fig_frp = px.histogram(
                        inspect_df,
                        x="fire_radiative_power",
                        nbins=30,
                        title="Fire Radiative Power (MW) Distribution",
                        color_discrete_sequence=[COLOR_GOLD]
                    )
                    apply_dark_theme(fig_frp)
                    fig_frp.update_layout(height=320)
                    st.plotly_chart(fig_frp, use_container_width=True)

            if "temperature" in inspect_df.columns and inspect_df["temperature"].notna().sum() > 0:
                with dist_cols[0]:
                    fig_t = px.histogram(
                        inspect_df,
                        x="temperature",
                        nbins=30,
                        title="Ambient Temperature (°C) Distribution",
                        color_discrete_sequence=[COLOR_ACCENT]
                    )
                    apply_dark_theme(fig_t)
                    fig_t.update_layout(height=320)
                    st.plotly_chart(fig_t, use_container_width=True)

# -----------------------------------------------------------------------------
# Section 5: FILTER & EXPORT
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Dataset Explorer Filters")

if all_datasets:
    # Sidebar Filters
    all_countries = ["All"]
    all_seasons = ["All", "Spring", "Summer", "Fall", "Winter"]

    try:
        with get_db_session() as session:
            countries_db = [c[0] for c in session.query(FireEvent.country).distinct().all() if c[0]]
            if countries_db:
                all_countries.extend(sorted(countries_db))
    except Exception:
        pass

    selected_countries = st.sidebar.multiselect("Filter Countries", options=all_countries, default=["All"])
    selected_seasons = st.sidebar.multiselect("Filter Seasons", options=all_seasons, default=["All"])
    conf_range = st.sidebar.slider("Confidence Score Range", min_value=0, max_value=100, value=(0, 100))

    st.markdown("---")
    st.markdown("### 💾 Filtered Data Export")

    if not inspect_df.empty:
        filtered_df = inspect_df.copy()

        if "All" not in selected_countries and "country" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

        if "All" not in selected_seasons and "season" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]

        if "confidence" in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df["confidence"] >= conf_range[0]) &
                (filtered_df["confidence"] <= conf_range[1])
            ]

        exp_c1, exp_c2 = st.columns([3, 1])
        with exp_c1:
            st.write(f"Matched **{len(filtered_df):,}** records matching current sidebar filter criteria.")
        with exp_c2:
            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered CSV",
                data=csv_bytes,
                file_name=f"wildfire_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )
