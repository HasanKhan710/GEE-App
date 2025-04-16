import streamlit as st
import ee
import geemap
import pandas as pd
import geopandas as gpd
import nest_asyncio
from shapely.geometry import LineString
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import asyncio
import os

# Initialize session state
if 'map_obj' not in st.session_state:
    st.session_state.map_obj = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Earth Engine authentication - for Streamlit Cloud deployment
@st.cache_resource
def initialize_ee():
    if 'EE_SERVICE_ACCOUNT_JSON' in st.secrets:
        service_account_info = st.secrets["EE_SERVICE_ACCOUNT_JSON"]
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'], 
            key_data=service_account_info['private_key']
        )
        ee.Initialize(credentials)
    else:
        try:
            ee.Initialize(project='ee-hasan710')
        except Exception as e:
            st.error(f"Error initializing Earth Engine: {e}")
            st.error("Please make sure you have authenticated with Earth Engine locally or configured secrets for deployment.")

# Initialize Earth Engine
initialize_ee()

# Apply nest_asyncio for async support in Streamlit
nest_asyncio.apply()

# Configure async event loop
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Add EE Layer to Folium Map
def add_ee_layer(self, ee_object, vis_params, name):
    try:
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
    except Exception as e:
        st.error(f" Dolanot display {name}: {str(e)}")

folium.Map.add_ee_layer = add_ee_layer

def process_temperature(uploaded_file, intensity, time_period):
    try:
        # Temperature thresholds for intensity levels
        thresholds = {"Low": 35, "Mid": 38, "High": 41}
        thresholds_p = {"Low": 50, "Mid": 100, "High": 150}
        thresholds_w = {"Low": 10, "Mid": 15, "High": 20}

        if intensity not in thresholds or time_period not in ["Monthly", "Weekly"]:
            raise ValueError("Invalid intensity or time period")

        # Load transmission line data from uploaded file
        df = pd.read_excel(uploaded_file, sheet_name="Line Parameters")
        df["geodata"] = df["geodata"].apply(
            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
        )
        line_geometries = [LineString(coords) for coords in df["geodata"]]

        # Create map
        m = folium.Map(location=[30, 70], zoom_start=5)

        # Convert transmission lines to FeatureCollection
        features = [ee.Feature(ee.Geometry.LineString(row["geodata"])) for _, row in df.iterrows()]
        line_fc = ee.FeatureCollection(features)
        bounding_box = line_fc.geometry().bounds()

        # Define date range (last 10 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)

        # Select dataset based on time period
        dataset_name = "ECMWF/ERA5/MONTHLY" if time_period == "Monthly" else "ECMWF/ERA5_LAND/DAILY_AGGR"
        dataset = ee.ImageCollection(dataset_name).filterDate(
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )

        dataset_forecast = ee.ImageCollection("NOAA/GFS0P25")

        # Band selection
        temp_band = "temperature_2m" if time_period == "Weekly" else "mean_2m_air_temperature"
        precip_band = "total_precipitation_sum" if time_period == "Weekly" else "total_precipitation"
        u_wind_band = "u_component_of_wind_10m"
        v_wind_band = "v_component_of_wind_10m"

        temp_forecast = "temperature_2m_above_ground"
        u_forecast = "u_component_of_wind_10m_above_ground"
        v_forecast = "v_component_of_wind_10m_above_ground"

        # Validate bands
        first_img = dataset.first()
        band_names = first_img.bandNames().getInfo()
        if temp_band not in band_names or precip_band not in band_names:
            raise ValueError(f"Dataset missing required bands. Available: {band_names}")

        # Data processing
        dataset1 = dataset.map(lambda img: img.select(temp_band).subtract(273.15).rename("temp_C"))
        filtered_dataset1 = dataset1.map(lambda img: img.gt(thresholds[intensity]))

        dataset2 = dataset.map(lambda img: img.select(precip_band).multiply(1000).rename("preci_mm"))
        filtered_dataset2 = dataset2.map(lambda img: img.gt(thresholds_p[intensity]))

        wind_magnitude = dataset.map(lambda img: img.expression(
            "sqrt(pow(u, 2) + pow(v, 2))",
            {"u": img.select(u_wind_band), "v": img.select(v_wind_band)}
        ).rename("wind_magnitude"))
        filtered_wind = wind_magnitude.map(lambda img: img.gt(thresholds_w[intensity]))

        # Process occurrences
        occurrence_count_t = filtered_dataset1.sum().clip(bounding_box)
        occurrence_count_p = filtered_dataset2.sum().clip(bounding_box)
        occurrence_count_w = filtered_wind.sum().clip(bounding_box)

        stats_t = occurrence_count_t.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=bounding_box,
            scale=1000,
            crs='EPSG:4326',
            bestEffort=True,
            maxPixels=1e13
        ).getInfo()

        stats_p = occurrence_count_p.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=bounding_box,
            scale=1000,
            crs='EPSG:4326',
            bestEffort=True,
            maxPixels=1e13
        ).getInfo()

        stats_w = occurrence_count_w.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=bounding_box,
            scale=1000,
            crs='EPSG:4326',
            bestEffort=True,
            maxPixels=1e13
        ).getInfo()

        # Display print outputs in Streamlit
        st.write("ReduceRegion Output for temperature:", stats_t)
        st.write("ReduceRegion Output for precipitation:", stats_p)
        st.write("ReduceRegion Output for wind:", stats_w)

        max_occurrence_t = stats_t.get(list(stats_t.keys())[0], 1)
        max_occurrence_p = stats_p.get(list(stats_p.keys())[0], 1)
        max_occurrence_w = stats_w.get(list(stats_w.keys())[0], 1)

        st.write(f"Max Occurrences of temperature: {max_occurrence_t}")
        st.write(f"Max Occurrences of precipitation: {max_occurrence_p}")
        st.write(f"Max Occurrences of wind: {max_occurrence_w}")

        # Classification parameters
        mid1_t, mid2_t = max_occurrence_t/3, 2*max_occurrence_t/3
        mid1_p, mid2_p = max_occurrence_p/3, 2*max_occurrence_p/3
        mid1_w, mid2_w = max_occurrence_w/3, 2*max_occurrence_w/3

        # Forecast calculations
        now = datetime.utcnow()
        nearest_gfs_time = now.replace(hour=(now.hour // 6) * 6)
        future = nearest_gfs_time + timedelta(hours=24)
        latest_image = dataset_forecast.sort("system:time_start", False).first()
        latest_timestamp = latest_image.date().format().getInfo()

        forecast_24h = dataset_forecast.filterDate(latest_timestamp, future.strftime('%Y-%m-%dT%H:%M:%S'))
        forecast_celsius = forecast_24h.select(temp_forecast).max().rename("forecast_temp_C")
        forecast_u = forecast_24h.select(u_forecast).max()
        forecast_v = forecast_24h.select(v_forecast).max()

        forecast_wind_magnitude = forecast_u.expression(
            "sqrt(pow(u, 2) + pow(v, 2))",
            {"u": forecast_u, "v": forecast_v}
        ).rename("forecast_wind_magnitude")

        # Classification expressions
        classified_t = occurrence_count_t.expression(
            "(VAL <= MID1) ? 1 : (VAL <= MID2) ? 2 : 3",
            {"VAL": occurrence_count_t, "MID1": mid1_t, "MID2": mid2_t}
        )
        classified_p = occurrence_count_p.expression(
            "(VAL <= MID1) ? 1 : (VAL <= MID2) ? 2 : 3",
            {"VAL": occurrence_count_p, "MID1": mid1_p, "MID2": mid2_p}
        )
        classified_w = occurrence_count_w.expression(
            "(VAL <= MID1) ? 1 : (VAL <= MID2) ? 2 : 3",
            {"VAL": occurrence_count_w, "MID1": mid1_w, "MID2": mid2_w}
        )

        # Forecast classifications
        mid1_ft = thresholds[intensity] * 0.9
        mid2_ft = thresholds[intensity] * 0.8
        mid1_fw = thresholds_w[intensity] * 0.9
        mid2_fw = thresholds_w[intensity] * 0.8

        classified_forecast_t = forecast_celsius.expression(
            "(VAL <= MID1) ? 1 : (VAL <= MID2) ? 2 : 3",
            {"VAL": forecast_celsius, "MID1": mid1_ft, "MID2": mid2_ft}
        )
        classified_forecast_w = forecast_wind_magnitude.expression(
            "(VAL <= MID1) ? 1 : (VAL <= MID2) ? 2 : 3",
            {"VAL": forecast_wind_magnitude, "MID1": mid1_fw, "MID2": mid2_fw}
        )

        combined_layer = (
            classified_t
            .add(classified_p)
            .add(classified_w)
            .add(classified_forecast_t)
            .add(classified_forecast_w)
        )

        # Visualization parameters
        classified_viz = {'min': 1, 'max': 3, 'palette': ['green', 'yellow', 'red']}
        combined_viz = {
            'min': 1,
            'max': 15,
            'palette': [
                '#00FF00', '#28D728', '#2ECC40', '#50C878', '#66CC66',
                '#7FC97F', '#99CC99', '#CCCC00', '#E6B800', '#FFD700',
                '#FFCC00', '#FFA500', '#FF9933', '#FF6600', '#FF0000'
            ]
        }

        # Add all EE raster layers first
        m.add_ee_layer(classified_t.clip(bounding_box), classified_viz, f"Temperature ({time_period})")
        m.add_ee_layer(classified_p.clip(bounding_box), classified_viz, f"Precipitation ({time_period})")
        m.add_ee_layer(classified_w.clip(bounding_box), classified_viz, f"Wind ({time_period})")
        m.add_ee_layer(classified_forecast_t.clip(bounding_box), classified_viz, "Forecast Temperature")
        m.add_ee_layer(classified_forecast_w.clip(bounding_box), classified_viz, "Forecast Wind")
        m.add_ee_layer(combined_layer.clip(bounding_box), combined_viz, "Combined")

        # Add transmission lines as a regular Folium layer
        for line in line_geometries:
            coords = [(y, x) for x, y in line.coords]  # Folium uses (lat, lon) order
            folium.PolyLine(
                locations=coords,
                weight=4,
                color='blue',
                opacity=1,
                name="Transmission Lines"
            ).add_to(m)

        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)

        return m

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI
st.title("Transmission Line Weather Risk Analyzer")
st.write("View sample results below or upload your own IEEE 9-Bus Parameters file to analyze weather risks.")

# File upload
uploaded_file = st.file_uploader("Upload IEEE 9-Bus Parameters", type="xlsx")

# User inputs
intensity = st.selectbox("Risk Intensity", ["Low", "Mid", "High"], index=1)  # Default to Mid
time_period = st.selectbox("Analysis Period", ["Weekly", "Monthly"], index=0)  # Default to Weekly

# Sample file path for Streamlit Cloud deployment
SAMPLE_FILE_PATH = "IEEE_9BUS_Parameters_only.xlsx"

# Check if sample file exists
if not os.path.exists(SAMPLE_FILE_PATH):
    st.warning(f"Sample file not found at: {SAMPLE_FILE_PATH}. Please ensure 'IEEE_9BUS_Parameters_only.xlsx' is in the correct directory.")
else:
    # Run analysis with sample file on page load if no user file is uploaded
    if not st.session_state.analysis_run and not uploaded_file:
        with st.spinner("Loading sample results..."):
            with open(SAMPLE_FILE_PATH, "rb") as f:
                st.session_state.map_obj = process_temperature(f, intensity, time_period)
                st.session_state.analysis_run = True
                st.session_state.uploaded_file = None
        st.info("Showing results for sample IEEE 9-Bus data. Upload your own file to run a new analysis.")
        
# Handle user-uploaded file
if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        st.session_state.map_obj = process_temperature(uploaded_file, intensity, time_period)
        st.session_state.analysis_run = True
        st.session_state.uploaded_file = uploaded_file
    st.success("Analysis complete for uploaded file.")

# Display map and results
if st.session_state.map_obj:
    st.subheader("Weather Risk Visualization")
    st_folium(st.session_state.map_obj, width=700, height=500, key="main_map")

    # CSS to ensure layer control is visible
    st.markdown("""
    <style>
    .leaflet-control-layers {
        z-index: 9999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Custom CSS for styling
    st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 18px;
        color: #2c3e50;
    }
    .stMap {
        width: 100%;
        height: 800px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Layer guide
    st.markdown("""
    **Layer Guide:**
    - Transmission Lines: Blue lines
    - Temperature/Precipitation/Wind: Green/Yellow/Red gradients
    - Forecast Layers: Future predictions
    - Combined: Multi-factor risk assessment
    """)

# Run analysis button (optional, for re-running with same file)
if st.session_state.uploaded_file and st.button("Re-run Analysis"):
    with st.spinner("Re-running analysis..."):
        st.session_state.map_obj = process_temperature(st.session_state.uploaded_file, intensity, time_period)
    st.success("Analysis re-run complete.")

# Reset button
if st.button("Clear Results"):
    st.session_state.map_obj = None
    st.session_state.analysis_run = False
    st.session_state.uploaded_file = None
    st.experimental_rerun()
