import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import pandapower as pp
import re
import ast
import nest_asyncio
import ee
from streamlit_folium import st_folium
from shapely.geometry import LineString
from datetime import datetime, timedelta
import random
import geemap
import numpy as np
import math
import traceback
from shapely.geometry import LineString, Point


# Set page configuration
st.set_page_config(
    page_title="Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Network Initialization", "Weather Risk Visualisation Using GEE", "Business As Usual", "Weather Aware System", "Data Analytics"]
selection = st.sidebar.radio("Go to", pages)

# Shared session state initialization
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "network_data" not in st.session_state:
    st.session_state.network_data = None
if "map_obj" not in st.session_state:
    st.session_state.map_obj = None
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None
if "weather_map_obj" not in st.session_state:
    st.session_state.weather_map_obj = None

# Apply nest_asyncio for async support in Streamlit (used in Network Initialization)
nest_asyncio.apply()

# Shared function: Add EE Layer to Folium Map (used in both pages)
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
        st.error(f"Could not display {name}: {str(e)}")

# Attach the method to folium.Map
folium.Map.add_ee_layer = add_ee_layer


def transform_loading(a):
    if a is None:
        return a
    is_single = False
    if isinstance(a, (int, float)):
        a = [a]
        is_single = True
    flag = True
    for item in a:
        if isinstance(item, (int, float)) and item >= 2.5:
            flag = False
    if flag:
        a = [item * 100 if isinstance(item, (int, float)) else item for item in a]
    return a[0] if is_single else a

def all_real_numbers(lst):
    invalid_count = 0
    for x in lst:
        if not isinstance(x, (int, float)):
            invalid_count += 1
        elif not math.isfinite(x):
            invalid_count += 1
    if invalid_count > len(lst):
        return False
    return True

def check_bus_pair(df_line, df_trafo, bus_pair):
    from_bus, to_bus = bus_pair
    if df_trafo is not None:
        transformer_match = (
            ((df_trafo['hv_bus'] == from_bus) & (df_trafo['lv_bus'] == to_bus)) |
            ((df_trafo['hv_bus'] == to_bus) & (df_trafo['lv_bus'] == from_bus))
        ).any()
        if transformer_match:
            return True
    line_match = (
        ((df_line['from_bus'] == from_bus) & (df_line['to_bus'] == to_bus)) |
        ((df_line['from_bus'] == to_bus) & (df_line['to_bus'] == from_bus))
    ).any()
    if line_match:
        return False
    st.error(f"Line or Transformer {from_bus}-{to_bus} not present in network.")
    return None

def generate_line_outages(outage_hours, line_down, risk_scores, capped_contingency_mode=False, df_line=None):
    if not outage_hours or not line_down or not risk_scores or df_line is None:
        return []
    no_of_lines_in_network = len(df_line) - 1
    capped_limit = math.floor(0.2 * no_of_lines_in_network)
    # # Debug: Log risk_scores structure
    # st.write("Debug: risk_scores =", risk_scores)
    # Extract numeric risk scores
    def extract_risk(rs):
        if isinstance(rs, (int, float)):
            return float(rs)
        elif isinstance(rs, dict):
            for key in ['score', 'risk', 'value']:  # Common keys
                if key in rs and isinstance(rs[key], (int, float)):
                    return float(rs[key])
                elif key in rs and isinstance(rs[key], str) and rs[key].replace('.', '', 1).isdigit():
                    return float(rs[key])
        elif isinstance(rs, str) and rs.replace('.', '', 1).isdigit():
            return float(rs)
        return 0.0  # Default for invalid entries
    numeric_risk_scores = [extract_risk(rs) for rs in risk_scores]
    combined = [(line[0], line[1], hour, risk) for line, hour, risk in zip(line_down, outage_hours, numeric_risk_scores)]
    sorted_combined = sorted(combined, key=lambda x: x[-1], reverse=True)
    line_outages = [(line[0], line[1], line[2]) for line in sorted_combined]
    if capped_contingency_mode and len(line_outages) > capped_limit:
        line_outages = line_outages[:capped_limit]
    return line_outages

def overloaded_lines(net, max_loading_capacity):
    overloaded = []
    for idx, res in net.res_line.iterrows():
        val = transform_loading(res["loading_percent"])
        if all_real_numbers(net.res_line['loading_percent'].tolist()) == False:
            if not isinstance(val, (int, float)) or math.isnan(val) or val >= max_loading_capacity:
                overloaded.append(idx)
        else:
            if val is not None and not (isinstance(val, float) and math.isnan(val)) and val > max_loading_capacity:
                overloaded.append(idx)
    return overloaded

def overloaded_transformer(net, max_loading_capacity_transformer):
    overloaded = []
    if 'trafo' in net and net.trafo is not None:
        for idx, res in net.res_trafo.iterrows():
            val = transform_loading(res["loading_percent"])
            if all_real_numbers(net.res_trafo['loading_percent'].tolist()) == False:
                if val is not None and not (isinstance(val, float) and math.isnan(val)) and val > max_loading_capacity_transformer:
                    overloaded.append(idx)
            else:
                if val >= max_loading_capacity_transformer:
                    overloaded.append(idx)
    return overloaded

def initialize_network(df_bus, df_load, df_gen, df_line, df_trafo, df_load_profile, df_gen_profile):
    net = pp.create_empty_network()
    for idx, row in df_bus.iterrows():
        pp.create_bus(net,
                      name=row["name"],
                      vn_kv=row["vn_kv"],
                      zone=row["zone"],
                      in_service=row["in_service"],
                      max_vm_pu=row["max_vm_pu"],
                      min_vm_pu=row["min_vm_pu"])
    for idx, row in df_load.iterrows():
        pp.create_load(net,
                       bus=row["bus"],
                       p_mw=row["p_mw"],
                       q_mvar=row["q_mvar"],
                       in_service=row["in_service"])
    for idx, row in df_gen.iterrows():
        if row["slack_weight"] == 1:
            ext_grid = pp.create_ext_grid(net,
                                          bus=row["bus"],
                                          vm_pu=row["vm_pu"],
                                          va_degree=0)
            pp.create_poly_cost(net, element=ext_grid, et="ext_grid",
                                cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])
        else:
            gen_idx = pp.create_gen(net,
                                    bus=row["bus"],
                                    p_mw=row["p_mw"],
                                    vm_pu=row["vm_pu"],
                                    min_q_mvar=row["min_q_mvar"],
                                    max_q_mvar=row["max_q_mvar"],
                                    scaling=row["scaling"],
                                    in_service=row["in_service"],
                                    slack_weight=row["slack_weight"],
                                    controllable=row["controllable"],
                                    max_p_mw=row["max_p_mw"],
                                    min_p_mw=row["min_p_mw"])
            pp.create_poly_cost(net, element=gen_idx, et="gen",
                                cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])
    for idx, row in df_line.iterrows():
        if pd.isna(row["parallel"]):
            continue
        if isinstance(row["geodata"], str):
            geodata = ast.literal_eval(row["geodata"])
        else:
            geodata = row["geodata"]
        pp.create_line_from_parameters(net,
                                       from_bus=row["from_bus"],
                                       to_bus=row["to_bus"],
                                       length_km=row["length_km"],
                                       r_ohm_per_km=row["r_ohm_per_km"],
                                       x_ohm_per_km=row["x_ohm_per_km"],
                                       c_nf_per_km=row["c_nf_per_km"],
                                       max_i_ka=row["max_i_ka"],
                                       in_service=row["in_service"],
                                       max_loading_percent=row["max_loading_percent"],
                                       geodata=geodata)
    if df_trafo is not None:
        for idx, row in df_trafo.iterrows():
            pp.create_transformer_from_parameters(net,
                                                  hv_bus=row["hv_bus"],
                                                  lv_bus=row["lv_bus"],
                                                  sn_mva=row["sn_mva"],
                                                  vn_hv_kv=row["vn_hv_kv"],
                                                  vn_lv_kv=row["vn_lv_kv"],
                                                  vk_percent=row["vk_percent"],
                                                  vkr_percent=row["vkr_percent"],
                                                  pfe_kw=row["pfe_kw"],
                                                  i0_percent=row["i0_percent"],
                                                  in_service=row["in_service"],
                                                  max_loading_percent=row["max_loading_percent"])
    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus = int(m.group(1))
            q_col = f"q_mvar_bus_{bus}"
            if q_col in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": q_col}
    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            numbers = re.findall(r'\d+', col)
            if numbers:
                bus = int(numbers[-1])
                gen_dynamic[bus] = col
    return net, load_dynamic, gen_dynamic

def calculate_hourly_cost(net, load_dynamic, gen_dynamic, num_hours, df_load_profile, df_gen_profile):
    hourly_cost_list = []
    for hour in range(num_hours):
        for bus_id, cols in load_dynamic.items():
            p_val = float(df_load_profile.at[hour, cols["p"]])
            q_val = float(df_load_profile.at[hour, cols["q"]])
            mask = net.load.bus == bus_id
            net.load.loc[mask, "p_mw"] = p_val
            net.load.loc[mask, "q_mvar"] = q_val
        for bus_id, col in gen_dynamic.items():
            p_val = float(df_gen_profile.at[hour, col])
            if bus_id in net.ext_grid.bus.values:
                mask = net.ext_grid.bus == bus_id
                net.ext_grid.loc[mask, "p_mw"] = p_val
            else:
                mask = net.gen.bus == bus_id
                net.gen.loc[mask, "p_mw"] = p_val
        try:
            pp.runopp(net)
            hourly_cost_list.append(net.res_cost)
        except:
            hourly_cost_list.append(0)
    return hourly_cost_list

# Shared function: Create and display the map (used in Network Initialization)
def create_map(df_line):
    try:
        # Process geodata
        df_line["geodata"] = df_line["geodata"].apply(
            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
        )
        line_geometries = [LineString(coords) for coords in df_line["geodata"]]
        gdf = gpd.GeoDataFrame(df_line, geometry=line_geometries, crs="EPSG:4326")

        # Create Folium map
        m = folium.Map(location=[30, 70], zoom_start=5, width=700, height=500)

        # Convert GeoDataFrame to EE FeatureCollection
        features = [ee.Feature(ee.Geometry.LineString(row["geodata"])) for _, row in df_line.iterrows()]
        line_fc = ee.FeatureCollection(features)

        # Add transmission lines to the map
        m.add_ee_layer(line_fc.style(**{'color': 'black', 'width': 2}), {}, "Transmission Lines")

        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)

        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None



# Page 1: Network Initialization
if selection == "Network Initialization":
    # Primary project title
    st.title("Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine")

    # Secondary page-specific title
    st.header("Network Initialization")

    # File uploader for the Excel file
    uploaded_file = st.file_uploader("Upload your network Excel file (e.g., Final_IEEE_9Bus_Parameters_only.xlsx)", type=["xlsx"], key="file_uploader")

    # Check if a new file was uploaded
    if uploaded_file is not None and st.session_state.uploaded_file_key != uploaded_file.name:
        st.session_state.show_results = False
        st.session_state.network_data = None
        st.session_state.map_obj = None
        st.session_state.uploaded_file_key = uploaded_file.name
        st.session_state.uploaded_file = uploaded_file  # Store the file object

    if uploaded_file is not None and not st.session_state.show_results:
        # Create an empty pandapower network
        net = pp.create_empty_network()

        # --- Create Buses ---
        df_bus = pd.read_excel(uploaded_file, sheet_name="Bus Parameters", index_col=0)
        for idx, row in df_bus.iterrows():
            pp.create_bus(net,
                          name=row["name"],
                          vn_kv=row["vn_kv"],
                          zone=row["zone"],
                          in_service=row["in_service"],
                          max_vm_pu=row["max_vm_pu"],
                          min_vm_pu=row["min_vm_pu"])

        # --- Create Loads ---
        df_load = pd.read_excel(uploaded_file, sheet_name="Load Parameters", index_col=0)
        for idx, row in df_load.iterrows():
            pp.create_load(net,
                           bus=row["bus"],
                           p_mw=row["p_mw"],
                           q_mvar=row["q_mvar"],
                           in_service=row["in_service"])

        # --- Create Transformers (if sheet exists) ---
        df_trafo = None
        if "Transformer Parameters" in pd.ExcelFile(uploaded_file).sheet_names:
            df_trafo = pd.read_excel(uploaded_file, sheet_name="Transformer Parameters", index_col=0)
            for idx, row in df_trafo.iterrows():
                pp.create_transformer_from_parameters(net,
                                                      hv_bus=row["hv_bus"],
                                                      lv_bus=row["lv_bus"],
                                                      sn_mva=row["sn_mva"],
                                                      vn_hv_kv=row["vn_hv_kv"],
                                                      vn_lv_kv=row["vn_lv_kv"],
                                                      vk_percent=row["vk_percent"],
                                                      vkr_percent=row["vkr_percent"],
                                                      pfe_kw=row["pfe_kw"],
                                                      i0_percent=row["i0_percent"],
                                                      in_service=row["in_service"],
                                                      max_loading_percent=row["max_loading_percent"])

        # --- Create Generators/External Grid ---
        df_gen = pd.read_excel(uploaded_file, sheet_name="Generator Parameters", index_col=0)
        df_gen['in_service'] = df_gen['in_service'].astype(str).str.strip().str.upper().map({'TRUE': True, 'FALSE': False}).fillna(True)
        df_gen['controllable'] = df_gen['controllable'].astype(str).str.strip().str.upper().map({'TRUE': True, 'FALSE': False})
        for idx, row in df_gen.iterrows():
            if row["slack_weight"] == 1:
                ext_idx = pp.create_ext_grid(net,
                                             bus=row["bus"],
                                             vm_pu=row["vm_pu"],
                                             va_degree=0)
                pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                    cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                    cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                    cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                    cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                    cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                    cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])
            else:
                gen_idx = pp.create_gen(net,
                                        bus=row["bus"],
                                        p_mw=row["p_mw"],
                                        vm_pu=row["vm_pu"],
                                        min_q_mvar=row["min_q_mvar"],
                                        max_q_mvar=row["max_q_mvar"],
                                        scaling=row["scaling"],
                                        in_service=row["in_service"],
                                        slack_weight=row["slack_weight"],
                                        controllable=row["controllable"],
                                        max_p_mw=row["max_p_mw"],
                                        min_p_mw=row["min_p_mw"])
                pp.create_poly_cost(net, element=gen_idx, et="gen",
                                    cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                    cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                    cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                    cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                    cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                    cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])

        # --- Create Lines ---
        df_line = pd.read_excel(uploaded_file, sheet_name="Line Parameters", index_col=0)
        for idx, row in df_line.iterrows():
            if isinstance(row["geodata"], str):
                geodata = ast.literal_eval(row["geodata"])
            else:
                geodata = row["geodata"]
            pp.create_line_from_parameters(net,
                                           from_bus=row["from_bus"],
                                           to_bus=row["to_bus"],
                                           length_km=row["length_km"],
                                           r_ohm_per_km=row["r_ohm_per_km"],
                                           x_ohm_per_km=row["x_ohm_per_km"],
                                           c_nf_per_km=row["c_nf_per_km"],
                                           max_i_ka=row["max_i_ka"],
                                           in_service=row["in_service"],
                                           max_loading_percent=row["max_loading_percent"],
                                           geodata=geodata)

        # --- Read Dynamic Profiles ---
        df_load_profile = pd.read_excel(uploaded_file, sheet_name="Load Profile")
        df_load_profile.columns = df_load_profile.columns.str.strip()

        df_gen_profile = pd.read_excel(uploaded_file, sheet_name="Generator Profile")
        df_gen_profile.columns = df_gen_profile.columns.str.strip()

        # --- Build Dictionaries for Dynamic Column Mapping ---
        load_dynamic = {}
        for col in df_load_profile.columns:
            m = re.match(r"p_mw_bus_(\d+)", col)
            if m:
                bus = int(m.group(1))
                q_col = f"q_mvar_bus_{bus}"
                if q_col in df_load_profile.columns:
                    load_dynamic[bus] = {"p": col, "q": q_col}

        gen_dynamic = {}
        for col in df_gen_profile.columns:
            if col.startswith("p_mw"):
                numbers = re.findall(r'\d+', col)
                if numbers:
                    bus = int(numbers[-1])
                    gen_dynamic[bus] = col

        # Store network data in session state
        st.session_state.network_data = {
            'df_bus': df_bus,
            'df_load': df_load,
            'df_gen': df_gen,
            'df_line': df_line,
            'df_load_profile': df_load_profile,
            'df_gen_profile': df_gen_profile,
            'df_trafo': df_trafo  # Add transformer data to session state
        }

    # --- Button to Display Results ---
    if st.button("Show Excel Network Parameters") and uploaded_file is not None:
        st.session_state.show_results = True
        # Generate map if not already generated
        if st.session_state.map_obj is None and st.session_state.network_data is not None:
            with st.spinner("Generating map..."):
                st.session_state.map_obj = create_map(st.session_state.network_data['df_line'])

    # --- Display Results ---
    if st.session_state.show_results and st.session_state.network_data is not None:
        st.subheader("Network Parameters")

        # Display Bus Parameters
        st.write("### Bus Parameters")
        st.dataframe(st.session_state.network_data['df_bus'])

        # Display Load Parameters
        st.write("### Load Parameters")
        st.dataframe(st.session_state.network_data['df_load'])

        # Display Generator Parameters
        st.write("### Generator Parameters")
        st.dataframe(st.session_state.network_data['df_gen'])

        # Display Transformer Parameters (if exists)
        if st.session_state.network_data['df_trafo'] is not None:
            st.write("### Transformer Parameters")
            st.dataframe(st.session_state.network_data['df_trafo'])

        # Display Line Parameters
        st.write("### Line Parameters")
        st.dataframe(st.session_state.network_data['df_line'])

        # Display Load Profile
        st.write("### Load Profile")
        st.dataframe(st.session_state.network_data['df_load_profile'])

        # Display Generator Profile
        st.write("### Generator Profile")
        st.dataframe(st.session_state.network_data['df_gen_profile'])

        # Display Map
        st.subheader("Transmission Network Map")
        if st.session_state.map_obj is not None:
            st_folium(st.session_state.map_obj, width=800, height=600, key="network_map")
            st.success("Network uploaded successfully!")
        else:
            st.warning("Map could not be generated.")

    # --- Clear Results Button ---
    if st.session_state.show_results and st.button("Clear Results"):
        st.session_state.show_results = False
        st.session_state.network_data = None
        st.session_state.map_obj = None
        st.session_state.uploaded_file_key = None
        st.experimental_rerun()

    if uploaded_file is None and not st.session_state.show_results:
        st.info("Please upload an Excel file to proceed.")
        
# Page 2: Weather Risk Visualisation Using GEE
elif selection == "Weather Risk Visualisation Using GEE":
    st.title("Weather Risk Visualisation Using GEE")

    # Create columns for dropdown menus
    col1, col2, col3 = st.columns(3)

    with col1:
        # Risk Tolerance dropdown
        intensity_options = ["Low", "Medium", "High"]
        intensity = st.selectbox(
            "Risk Tolerance",
            options=intensity_options,
            help="Low: Temperature > 35°C, Precipitation > 50mm, Wind > 10m/s, Medium: Temperature > 38°C, Precipitation > 100mm, Wind > 15m/s, High: Temperature > 41°C, Precipitation > 150mm, Wind > 20m/s"
        )

    with col2:
        # Study Period dropdown
        period_options = ["Weekly", "Monthly"]
        study_period = st.selectbox(
            "Study Period",
            options=period_options,
            help="Weekly: Daily aggregated data, Monthly: Monthly aggregated data"
        )

    with col3:
        # Risk Score Threshold slider
        risk_score = st.slider(
            "Risk Score Threshold",
            min_value=6,
            max_value=18,
            value=14,
            help="Higher threshold means higher risk tolerance. Range: 6-18"
        )

    # Check if network data is available
    if "network_data" not in st.session_state or st.session_state.network_data is None:
        st.warning("Please upload and initialize network data on the Network Initialization page first.")
    else:
        # Button to process and show results
        if st.button("Process Weather Risk Data"):
            with st.spinner("Processing weather risk data..."):
                try:
                    # Initialize Earth Engine if not already done
                    try:
                        initialize_ee()
                    except Exception as e:
                        st.error(f"Error initializing Earth Engine: {str(e)}")

                    # Process temperature and generate results
                    def process_temperature(intensity, time_period, risk_score_threshold, df_line):
                        # Temperature thresholds for intensity levels
                        thresholds = {"Low": 35, "Medium": 38, "High": 41}
                        thresholds_p = {"Low": 50, "Medium": 100, "High": 150}
                        thresholds_w = {"Low": 10, "Medium": 15, "High": 20}

                        if intensity not in thresholds or time_period not in ["Monthly", "Weekly"]:
                            raise ValueError("Invalid intensity or time period")

                        # Use the transmission line data from session state
                        df = df_line.copy()

                        from_buses = df["from_bus"].tolist()
                        to_buses = df["to_bus"].tolist()
                        all_lines = list(df[["from_bus", "to_bus"]].itertuples(index=False, name=None))

                        df["geodata"] = df["geodata"].apply(lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x)
                        line_geometries = [LineString(coords) for coords in df["geodata"]]
                        gdf = gpd.GeoDataFrame(df, geometry=line_geometries, crs="EPSG:4326")

                        # Create Folium map (instead of geemap.Map)
                        m = folium.Map(location=[30, 70], zoom_start=5, width=800, height=600)

                        # Define date range (last 10 years)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365 * 10)

                        # Select dataset based on time period
                        dataset_name = "ECMWF/ERA5/MONTHLY" if time_period == "Monthly" else "ECMWF/ERA5_LAND/DAILY_AGGR"
                        dataset = ee.ImageCollection(dataset_name).filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                        dataset_forecast = ee.ImageCollection("NOAA/GFS0P25")
                        d = dataset_forecast.first()

                        # Create land mask
                        land_mask = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                        land_mask = land_mask.map(lambda feature: feature.set("dummy", 1))
                        land_image = land_mask.reduceToImage(["dummy"], ee.Reducer.first()).gt(0)

                        # Select the correct band
                        temp_band = "temperature_2m" if time_period == "Weekly" else "mean_2m_air_temperature"
                        precip_band = "total_precipitation_sum" if time_period == "Weekly" else "total_precipitation"
                        u_wind_band = "u_component_of_wind_10m" if time_period == "Weekly" else "u_component_of_wind_10m"
                        v_wind_band = "v_component_of_wind_10m" if time_period == "Weekly" else "v_component_of_wind_10m"

                        temp_forecast = "temperature_2m_above_ground"
                        u_forecast = "u_component_of_wind_10m_above_ground"
                        v_forecast = "v_component_of_wind_10m_above_ground"
                        precip_forecast = "precipitation_rate"

                        # Ensure dataset contains required bands
                        first_img = dataset.first()
                        band_names = first_img.bandNames().getInfo()
                        required_bands = [temp_band, precip_band, u_wind_band, v_wind_band]
                        for band in required_bands:
                            if band not in band_names:
                                raise ValueError(f"Dataset does not contain band: {band}. Available bands: {band_names}")

                        # Convert temperature from Kelvin to Celsius and filter occurrences above threshold
                        dataset1 = dataset.map(lambda img: img.select(temp_band).subtract(273.15).rename("temp_C"))
                        filtered_dataset1 = dataset1.map(lambda img: img.gt(thresholds[intensity]))

                        dataset2 = dataset.map(lambda img: img.select(precip_band).multiply(1000).rename("preci_mm"))
                        filtered_dataset2 = dataset2.map(lambda img: img.gt(thresholds_p[intensity]))

                        dataset3 = dataset.map(lambda img: img.select(u_wind_band).rename("u_wind"))
                        dataset4 = dataset.map(lambda img: img.select(v_wind_band).rename("v_wind"))

                        wind_magnitude = dataset.map(lambda img: img.expression(
                            "sqrt(pow(u, 2) + pow(v, 2))",
                            {
                                "u": img.select(u_wind_band),
                                "v": img.select(v_wind_band)
                            }
                        ).rename("wind_magnitude"))

                        filtered_wind = wind_magnitude.map(lambda img: img.gt(thresholds_w[intensity]))

                        # Sum occurrences where thresholds were exceeded
                        occurrence_count_t = filtered_dataset1.sum()
                        occurrence_count_p = filtered_dataset2.sum()
                        occurrence_count_w = filtered_wind.sum()

                        # Convert transmission lines to FeatureCollection
                        features = [
                            ee.Feature(ee.Geometry.LineString(row["geodata"]), {
                                "line_id": i,
                                "geodata": str(row["geodata"])
                            }) for i, row in df.iterrows()
                        ]
                        line_fc = ee.FeatureCollection(features)

                        bounding_box = line_fc.geometry().bounds()

                        masked_occurrences_t = occurrence_count_t.clip(bounding_box)
                        masked_occurrences_p = occurrence_count_p.clip(bounding_box)
                        masked_occurrences_w = occurrence_count_w.clip(bounding_box)

                        masked_occurrences_t = masked_occurrences_t.updateMask(land_image)
                        masked_occurrences_p = masked_occurrences_p.updateMask(land_image)
                        masked_occurrences_w = masked_occurrences_w.updateMask(land_image)

                        # Computing occurrence statistics
                        stats_t = masked_occurrences_t.reduceRegion(
                            reducer=ee.Reducer.max(),
                            geometry=bounding_box,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e13
                        )

                        stats_p = masked_occurrences_p.reduceRegion(
                            reducer=ee.Reducer.max(),
                            geometry=bounding_box,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e13
                        )

                        stats_w = masked_occurrences_w.reduceRegion(
                            reducer=ee.Reducer.max(),
                            geometry=bounding_box,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e13
                        )

                        stats_dict_t = stats_t.getInfo()
                        stats_dict_p = stats_p.getInfo()
                        stats_dict_w = stats_w.getInfo()

                        if stats_dict_t:
                            max_occurrence_key_t = list(stats_dict_t.keys())[0]
                            max_occurrence_t = ee.Number(stats_t.get(max_occurrence_key_t)).getInfo()
                        else:
                            max_occurrence_t = 1

                        if stats_dict_p:
                            max_occurrence_key_p = list(stats_dict_p.keys())[0]
                            max_occurrence_p = ee.Number(stats_p.get(max_occurrence_key_p)).getInfo()
                        else:
                            max_occurrence_p = 1

                        if stats_dict_w:
                            max_occurrence_key_w = list(stats_dict_w.keys())[0]
                            max_occurrence_w = ee.Number(stats_w.get(max_occurrence_key_w)).getInfo()
                        else:
                            max_occurrence_w = 1

                        mid1_t = max_occurrence_t / 3
                        mid2_t = 2 * (max_occurrence_t / 3)

                        mid1_p = max_occurrence_p / 3
                        mid2_p = 2 * (max_occurrence_p / 3)

                        mid1_w = max_occurrence_w / 3
                        mid2_w = 2 * (max_occurrence_w / 3)

                        mid1_ft = thresholds[intensity] * (1 - 10/100)
                        mid2_ft = thresholds[intensity] * (1 - 20/100)

                        mid1_fw = thresholds_w[intensity] * (1 - 10/100)
                        mid2_fw = thresholds_w[intensity] * (1 - 20/100)

                        # Get current time and forecast time
                        now = datetime.utcnow()
                        nearest_gfs_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
                        future = nearest_gfs_time + timedelta(hours=24)
                        now_str = nearest_gfs_time.strftime('%Y-%m-%dT%H:%M:%S')
                        future_str = future.strftime('%Y-%m-%dT%H:%M:%S')

                        latest_image = dataset_forecast.sort("system:time_start", False).first()
                        latest_timestamp = latest_image.date().format().getInfo()

                        # Classify occurrences
                        classified_occurrences_t = masked_occurrences_t.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": masked_occurrences_t,
                                "mid1": mid1_t,
                                "mid2": mid2_t,
                            }
                        )

                        classified_occurrences_w = masked_occurrences_w.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": masked_occurrences_w,
                                "mid1": mid1_w,
                                "mid2": mid2_w,
                            }
                        )

                        classified_occurrences_p = masked_occurrences_p.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": masked_occurrences_p,
                                "mid1": mid1_p,
                                "mid2": mid2_p,
                            }
                        )

                        classified_viz = {
                            'min': 1,
                            'max': 3,
                            'palette': ['green', 'yellow', 'red']
                        }

                        classified_t = classified_occurrences_t.clip(bounding_box)
                        classified_p = classified_occurrences_p.clip(bounding_box)
                        classified_w = classified_occurrences_w.clip(bounding_box)

                        classified_t = classified_t.updateMask(land_image)
                        classified_p = classified_p.updateMask(land_image)
                        classified_w = classified_w.updateMask(land_image)

                        combined_layer = classified_t.add(classified_p).add(classified_w)
                        combined_layer = combined_layer.clip(bounding_box)
                        combined_layer = combined_layer.updateMask(land_image)

                        vis_params = {
                            'min': 3,
                            'max': 9,
                            'palette': ['lightgreen', 'green', 'yellow', 'orange', 'red', 'crimson', 'darkred']
                        }

                        combined_viz = {
                            'min': 6,
                            'max': 18,
                            'palette': [
                                '#32CD32',  # 6 - Lime Green
                                '#50C878',  # 7 - Medium Sea Green
                                '#66CC66',  # 8 - Soft Green
                                '#B2B200',  # 9 - Olive Green
                                '#CCCC00',  # 10 - Yellow-Green
                                '#E6B800',  # 11 - Mustard Yellow
                                '#FFD700',  # 12 - Golden Yellow
                                '#FFCC00',  # 13 - Deep Yellow
                                '#FFA500',  # 14 - Orange
                                '#FF9933',  # 15 - Dark Orange
                                '#FF6600',  # 16 - Reddish Orange
                                '#FF0000'   # 18 - Red
                            ]
                        }

                        # Add weather layers
                        m.add_ee_layer(classified_t, classified_viz, f"Temperature Occurrence Classification ({time_period})")
                        m.add_ee_layer(classified_p, classified_viz, f"Precipitation Occurrence Classification ({time_period})")
                        m.add_ee_layer(classified_w, classified_viz, f"Wind Occurrence Classification ({time_period})")
                        m.add_ee_layer(combined_layer, vis_params, "Combined Historic Classification")

                        fut = [latest_timestamp]
                        daily_dfs = {}
                        results_per_day = []
                        max_times = []
                        risk_scores = []  # Add this line to initialize risk_scores

                        # Process forecast for next 24 hours
                        future = nearest_gfs_time + timedelta(hours=24)
                        future_str = future.strftime('%Y-%m-%dT%H:%M:%S')
                        fut.append(future_str)

                        forecast_24h = dataset_forecast.filterDate(latest_timestamp, future_str)

                        forecast_temp = forecast_24h.select(temp_forecast).max().rename("forecast_temp_C_day_1")
                        forecast_u = forecast_24h.select(u_forecast).max().rename("forecast_u_day_1")
                        forecast_v = forecast_24h.select(v_forecast).max().rename("forecast_v_day_1")
                        forecast_pre = forecast_24h.select(precip_forecast).max().multiply(86400).rename("forecast_prec_day_1")

                        forecast_wind_magnitude = forecast_u.expression(
                            "sqrt(pow(u, 2) + pow(v, 2))",
                            {"u": forecast_u, "v": forecast_v}
                        ).rename("forecast_wind_magnitude_day_1")

                        mid1_ft = thresholds[intensity] * 0.90
                        mid2_ft = thresholds[intensity] * 0.80
                        mid1_fw = thresholds_w[intensity] * 0.90
                        mid2_fw = thresholds_w[intensity] * 0.80
                        mid1_fp = thresholds_p[intensity] * 0.90
                        mid2_fp = thresholds_p[intensity] * 0.80

                        classified_forecast_t = forecast_temp.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": forecast_temp,
                                "mid1": mid1_ft,
                                "mid2": mid2_ft
                            }
                        ).clip(bounding_box)

                        classified_forecast_w = forecast_wind_magnitude.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": forecast_wind_magnitude,
                                "mid1": mid1_fw,
                                "mid2": mid2_fw
                            }
                        ).clip(bounding_box)

                        classified_forecast_p = forecast_pre.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": forecast_pre,
                                "mid1": mid1_fp,
                                "mid2": mid2_fp
                            }
                        ).clip(bounding_box)

                        combined_forecast = classified_forecast_t.add(classified_forecast_w).add(classified_forecast_p).clip(bounding_box)
                        combined_forecast = combined_forecast.add(combined_layer)
                        combined_forecast = combined_forecast.updateMask(land_image)

                        # Add combined forecast layer
                        m.add_ee_layer(combined_forecast, combined_viz, "Day Ahead - Risk Score")

                        # Add transmission lines last with thicker styling
                        m.add_ee_layer(line_fc.style(**{'color': 'blue', 'width': 4}), {}, "Transmission Lines")

                        # Add layer control
                        folium.LayerControl(collapsed=False).add_to(m)

                        # Reduce regions to get risk scores per line
                        reduced = combined_forecast.reduceRegions(
                            collection=line_fc,
                            reducer=ee.Reducer.max(),
                            scale=1000
                        )

                        results = reduced.getInfo()

                        data = []
                        daily_results = []
                        risk_scores = []

                        for feature in results["features"]:
                            props = feature["properties"]
                            line_id = props["line_id"]
                            max_risk = props.get("max", 0)
                            from_bus = df.loc[line_id, "from_bus"]
                            to_bus = df.loc[line_id, "to_bus"]
                            daily_results.append((int(from_bus), int(to_bus), int(max_risk)))
                            risk_scores.append(int(max_risk))  # Add this line to collect risk scores
                    
                            data.append({
                                "line_id": props["line_id"],
                                "from_bus": int(from_bus),
                                "to_bus": int(to_bus),
                                "risk_score": int(max_risk)
                            })
                    
                            risk_scores.append({
                                "line_id": int(line_id),
                                "from_bus": int(from_bus),
                                "to_bus": int(to_bus),
                                "risk_score": int(max_risk)
                            })
                        results_per_day.append(daily_results)
                        daily_dfs["Day_1"] = pd.DataFrame(data)

                        # Filter lines with risk score >= threshold
                        day_1_results = results_per_day[0]
                        filtered_lines_day1 = [(from_bus, to_bus) for from_bus, to_bus, score in day_1_results if score >= risk_score_threshold]
                        length_lines = len(filtered_lines_day1)
                        outage_hour_day = [random.randint(11, 15) for _ in range(length_lines)]

                        # Create structured output for lines and outage hours
                        line_outages = [{"from_bus": from_bus, "to_bus": to_bus} for from_bus, to_bus in filtered_lines_day1]
                        outage_data = [{"line": f"From Bus {line[0]} to Bus {line[1]}", "outage_hours": hours, "risk_score": score}
                                      for line, hours, score in zip(filtered_lines_day1, outage_hour_day, [score for _, _, score in day_1_results if score >= risk_score_threshold])]

                        # Store in a format that can be used by other pages
                        line_outage_data = {
                            "lines": filtered_lines_day1,
                            "hours": outage_hour_day,
                            "risk_scores": risk_scores
                        }

                        return m, daily_dfs["Day_1"], line_outage_data, outage_data, max_occurrence_t, max_occurrence_p, max_occurrence_w, risk_scores  # Update this line

                    # Call the function with selected parameters
                    weather_map, risk_df, line_outage_data, outage_data, max_occurrence_t, max_occurrence_p, max_occurrence_w, risk_scores = process_temperature(
                    intensity,
                    study_period,
                    risk_score,
                    st.session_state.network_data['df_line']
                    )
                    # Store the map and data in session state
                    st.session_state.weather_map_obj = weather_map
                    st.session_state.line_outage_data = line_outage_data
                    st.session_state.risk_df = risk_df
                    st.session_state.outage_data = outage_data
                    st.session_state.risk_score = risk_score
                    st.session_state.max_occurrences = {
                        "temperature": max_occurrence_t,
                        "precipitation": max_occurrence_p,
                        "wind": max_occurrence_w
                    }

                    # Display the results
                    st.subheader("Day Ahead Risk Assessment")

                    # Display the map
                    st.write("### Geographic Risk Visualization")
                    if st.session_state.weather_map_obj:
                        st_folium(st.session_state.weather_map_obj, width=800, height=600, key="weather_map")

                    # Display legends
                    st.write("### Risk Visualization Legends")
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Patch

                    # Define legend data
                    final_risk_score = {
                        "title": "Final Risk Score (6-18)",
                        "colors": [('#32CD32', '6'), ('#50C878', '7'), ('#66CC66', '8'), ('#B2B200', '9'),
                                   ('#CCCC00', '10'), ('#E6B800', '11'), ('#FFD700', '12'), ('#FFCC00', '13'),
                                   ('#FFA500', '14'), ('#FF9933', '15'), ('#FF6600', '16'), ('#FF0000', '18')]
                    }

                    historical_classification = {
                        "title": "Historical Risk Classification (1-3)",
                        "colors": [('green', '1'), ('yellow', '2'), ('red', '3')]
                    }

                    historical_score = {
                        "title": "Historical Risk Score (3-9)",
                        "colors": [('lightgreen', '3'), ('green', '4'), ('yellow', '5'), ('orange', '6'),
                                   ('red', '7'), ('crimson', '8'), ('darkred', '9')]
                    }

                    # Create figure and grid layout
                    fig = plt.figure(figsize=(5, 3))
                    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.8])

                    # Final Risk Score - vertical on left
                    ax1 = fig.add_subplot(gs[:, 0])
                    ax1.axis('off')
                    ax1.set_title(final_risk_score["title"], fontsize=9, fontweight='bold', color='white', loc='left')
                    handles1 = [Patch(color=color, label=label) for color, label in final_risk_score["colors"]]
                    ax1.legend(handles=handles1, loc='center left', frameon=False,
                               handleheight=1.2, handlelength=8, fontsize=9, labelcolor='white')

                    # Historical Risk Classification - top right
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.axis('off')
                    ax2.set_title(historical_classification["title"], fontsize=9, fontweight='bold', color='white')
                    handles2 = [Patch(color=color, label=label) for color, label in historical_classification["colors"]]
                    ax2.legend(handles=handles2, loc='center', ncol=3, frameon=False,
                               handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                    # Historical Risk Score - bottom right
                    ax3 = fig.add_subplot(gs[1, 1])
                    ax3.axis('off')
                    ax3.set_title(historical_score["title"], fontsize=9, fontweight='bold', color='white')
                    handles3 = [Patch(color=color, label=label) for color, label in historical_score["colors"]]
                    ax3.legend(handles=handles3, loc='center', ncol=3, frameon=False,
                               handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                    fig.patch.set_facecolor('black')
                    plt.tight_layout(pad=1)
                    st.pyplot(fig)

                    # Display risk scores for all lines
                    st.write("### Risk Scores for All Transmission Lines")
                    risk_df_display = risk_df[["line_id", "from_bus", "to_bus", "risk_score"]].sort_values(by="risk_score", ascending=False)
                    risk_df_display.columns = ["Line ID", "From Bus", "To Bus", "Risk Score"]
                    st.dataframe(risk_df_display, use_container_width=True)

                    # Display lines expected to face outage based on threshold
                    if outage_data:
                        st.write(f"### Lines Expected to Face Outage (Risk Score ≥ {risk_score})")
                        outage_df = pd.DataFrame(outage_data)
                        outage_df.columns = ["Transmission Line", "Expected Outage Hours", "Risk Score"]
                        st.dataframe(outage_df, use_container_width=True)

                        # Summary statistics
                        st.write("### Outage Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Number of Lines at Risk", len(outage_data))
                        with col2:
                            st.metric("Max Temperature Occurrences", int(max_occurrence_t))
                        with col3:
                            st.metric("Max Precipitation Occurrences", int(max_occurrence_p))
                        with col4:
                            st.metric("Max Wind Occurrences", int(max_occurrence_w))
                    else:
                        st.success(f"No transmission lines are expected to face outage at the selected risk threshold ({risk_score}).")
                        # Still display max occurrences even if no outages
                        st.write("### Historical Max Occurrences")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Temperature Occurrences", int(max_occurrence_t))
                        with col2:
                            st.metric("Max Precipitation Occurrences", int(max_occurrence_p))
                        with col3:
                            st.metric("Max Wind Occurrences", int(max_occurrence_w))

                except Exception as e:
                    st.error(f"Error processing weather risk data: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        else:
            # Display cached results if available
            if st.session_state.weather_map_obj and "risk_df" in st.session_state and "outage_data" in st.session_state:
                st.subheader("Day Ahead Risk Assessment")

                # Display the map
                st.write("### Geographic Risk Visualization")
                st_folium(st.session_state.weather_map_obj, width=800, height=600, key="weather_map_cached")

                # Display legends
                st.write("### Risk Visualization Legends")
                import matplotlib.pyplot as plt
                from matplotlib.patches import Patch

                # Define legend data
                final_risk_score = {
                    "title": "Final Risk Score (6-18)",
                    "colors": [('#32CD32', '6'), ('#50C878', '7'), ('#66CC66', '8'), ('#B2B200', '9'),
                               ('#CCCC00', '10'), ('#E6B800', '11'), ('#FFD700', '12'), ('#FFCC00', '13'),
                               ('#FFA500', '14'), ('#FF9933', '15'), ('#FF6600', '16'), ('#FF0000', '18')]
                }

                historical_classification = {
                    "title": "Historical Risk Classification (1-3)",
                    "colors": [('green', '1'), ('yellow', '2'), ('red', '3')]
                }

                historical_score = {
                    "title": "Historical Risk Score (3-9)",
                    "colors": [('lightgreen', '3'), ('green', '4'), ('yellow', '5'), ('orange', '6'),
                               ('red', '7'), ('crimson', '8'), ('darkred', '9')]
                }

                # Create figure and grid layout
                fig = plt.figure(figsize=(5, 3))
                gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.8])

                # Final Risk Score - vertical on left
                ax1 = fig.add_subplot(gs[:, 0])
                ax1.axis('off')
                ax1.set_title(final_risk_score["title"], fontsize=9, fontweight='bold', color='white', loc='left')
                handles1 = [Patch(color=color, label=label) for color, label in final_risk_score["colors"]]
                ax1.legend(handles=handles1, loc='center left', frameon=False,
                           handleheight=1.2, handlelength=8, fontsize=9, labelcolor='white')

                # Historical Risk Classification - top right
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.axis('off')
                ax2.set_title(historical_classification["title"], fontsize=9, fontweight='bold', color='white')
                handles2 = [Patch(color=color, label=label) for color, label in historical_classification["colors"]]
                ax2.legend(handles=handles2, loc='center', ncol=3, frameon=False,
                           handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                # Historical Risk Score - bottom right
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.axis('off')
                ax3.set_title(historical_score["title"], fontsize=9, fontweight='bold', color='white')
                handles3 = [Patch(color=color, label=label) for color, label in historical_score["colors"]]
                ax3.legend(handles=handles3, loc='center', ncol=3, frameon=False,
                           handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                fig.patch.set_facecolor('black')
                plt.tight_layout(pad=1)
                st.pyplot(fig)

                # Display risk scores for all lines
                st.write("### Risk Scores for All Transmission Lines")
                risk_df_display = st.session_state.risk_df[["line_id", "from_bus", "to_bus", "risk_score"]].sort_values(by="risk_score", ascending=False)
                risk_df_display.columns = ["Line ID", "From Bus", "To Bus", "Risk Score"]
                st.dataframe(risk_df_display, use_container_width=True)

                # Display lines expected to face outage based on threshold
                if st.session_state.outage_data:
                    st.write(f"### Lines Expected to Face Outage (Risk Score ≥ {st.session_state.risk_score})")
                    outage_df = pd.DataFrame(st.session_state.outage_data)
                    outage_df.columns = ["Transmission Line", "Expected Outage Hours", "Risk Score"]
                    st.dataframe(outage_df, use_container_width=True)

                    # Summary statistics
                    st.write("### Outage Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Number of Lines at Risk", len(st.session_state.outage_data))
                    with col2:
                        st.metric("Max Temperature Occurrences", int(st.session_state.max_occurrences["temperature"]))
                    with col3:
                        st.metric("Max Precipitation Occurrences", int(st.session_state.max_occurrences["precipitation"]))
                    with col4:
                        st.metric("Max Wind Occurrences", int(st.session_state.max_occurrences["wind"]))
                else:
                    st.success(f"No transmission lines are expected to face outage at the selected risk threshold ({st.session_state.risk_score}).")
                    # Still display max occurrences even if no outages
                    st.write("### Historical Max Occurrences")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Temperature Occurrences", int(st.session_state.max_occurrences["temperature"]))
                    with col2:
                        st.metric("Max Precipitation Occurrences", int(st.session_state.max_occurrences["precipitation"]))
                    with col3:
                        st.metric("Max Wind Occurrences", int(st.session_state.max_occurrences["wind"]))
            else:
                st.info("Select parameters and click 'Process Weather Risk Data' to analyze weather risks to the electricity grid.")
                

# Page 3: Business As Usual
elif selection == "Business As Usual":
    st.title("Business As Usual")
    
    # Validate required data
    required_keys = ['df_bus', 'df_load', 'df_gen', 'df_line', 'df_load_profile', 'df_gen_profile']
    required_load_cols = ['bus', 'p_mw', 'q_mvar', 'in_service', 'criticality', 'load_coordinates']
    if "network_data" not in st.session_state or st.session_state.network_data is None:
        st.warning("Please upload and initialize network data on the Network Initialization page.")
    elif not all(key in st.session_state.network_data for key in required_keys):
         st.warning("Network data is incomplete. Ensure all required sheets are loaded.")
    elif not all(col in st.session_state.network_data['df_load'].columns for col in required_load_cols):
         st.warning("Load Parameters missing required columns (e.g., criticality, load_coordinates).")
    elif "line_outage_data" not in st.session_state or st.session_state.line_outage_data is None:
        st.warning("Please process weather risk data on the Weather Risk Visualisation page.")
    else:
        # Initialize session state
        if "bau_results" not in st.session_state:
            st.session_state.bau_results = None
        if "bau_map_obj" not in st.session_state:
            st.session_state.bau_map_obj = None
        if "selected_hour" not in st.session_state:
            st.session_state.selected_hour = None

        # Dropdown for contingency mode
        contingency_mode = st.selectbox(
            "Select Contingency Mode",
            options=["Capped Contingency Mode", "Maximum Contingency Mode"],
            help="Capped: Limits outages to 20% of network lines. Maximum: Includes all outages."
        )
        capped_contingency = contingency_mode == "Capped Contingency Mode"
        
        # Button to run analysis
        if st.button("Run Business As Usual Analysis"):
            with st.spinner("Running Business As Usual analysis..."):
                try:
                    # Extract data
                    network_data = st.session_state.network_data
                    df_bus = network_data['df_bus']
                    df_load = network_data['df_load']
                    df_gen = network_data['df_gen']
                    df_line = network_data['df_line']
                    df_load_profile = network_data['df_load_profile']
                    df_gen_profile = network_data['df_gen_profile']
                    df_trafo = network_data.get('df_trafo')
                    line_outage_data = st.session_state.line_outage_data
                    outage_hours = line_outage_data['hours']
                    line_down = line_outage_data['lines']
                    risk_scores = line_outage_data['risk_scores']
                            
        
                    def run_bau_simulation(net, load_dynamic, gen_dynamic, num_hours, line_outages, max_loading_capacity, max_loading_capacity_transformer):
                        business_as_usual_cost = calculate_hourly_cost(net, load_dynamic, gen_dynamic, num_hours, df_load_profile, df_gen_profile)
                        cumulative_load_shedding = {bus: {"p_mw": 0.0, "q_mvar": 0.0} for bus in net.load["bus"].unique()}
                        total_demand_per_bus = {}
                        p_cols = [c for c in df_load_profile.columns if c.startswith("p_mw_bus_")]
                        q_cols = [c for c in df_load_profile.columns if c.startswith("q_mvar_bus_")]
                        bus_ids = set(int(col.rsplit("_", 1)[1]) for col in p_cols)
                        for bus in bus_ids:
                            p_col = f"p_mw_bus_{bus}"
                            q_col = f"q_mvar_bus_{bus}"
                            total_p = df_load_profile[p_col].sum()
                            total_q = df_load_profile[q_col].sum()
                            total_demand_per_bus[bus] = {"p_mw": float(total_p), "q_mvar": float(total_q)}
                        hourly_shed_bau = [0] * num_hours
                        served_load_per_hour = []
                        gen_per_hour_bau = []
                        slack_per_hour_bau = []
                        loading_percent_bau = []
                        shedding_buses = []
                        
                        for hour in range(num_hours):
                            # Reset network state
                            net.line["in_service"] = True
                            if df_trafo is not None:
                                net.trafo["in_service"] = True
                            
                            # Apply outages
                            for (fbus, tbus, start_hr) in line_outages:
                                if hour < start_hr:
                                    continue
                                is_trafo = check_bus_pair(df_line, df_trafo, (fbus, tbus))
                                if is_trafo:
                                    mask_tf = (
                                        ((net.trafo.hv_bus == fbus) & (net.trafo.lv_bus == tbus)) |
                                        ((net.trafo.hv_bus == tbus) & (net.trafo.lv_bus == fbus))
                                    )
                                    if mask_tf.any():
                                        for tf_idx in net.trafo[mask_tf].index:
                                            net.trafo.at[tf_idx, "in_service"] = False
                                elif is_trafo == False:
                                    idx = line_idx_map.get((fbus, tbus))
                                    if idx is not None:
                                        net.line.at[idx, "in_service"] = False
                            
                            # Update profiles
                            for idx in net.load.index:
                                bus = net.load.at[idx, "bus"]
                                if bus in load_dynamic:
                                    p = df_load_profile.at[hour, load_dynamic[bus]["p"]]
                                    q = df_load_profile.at[hour, load_dynamic[bus]["q"]]
                                    net.load.at[idx, "p_mw"] = p
                                    net.load.at[idx, "q_mvar"] = q
                            for idx in net.gen.index:
                                bus = net.gen.at[idx, "bus"]
                                if bus in gen_dynamic:
                                    p = df_gen_profile.at[hour, gen_dynamic[bus]]
                                    net.gen.at[idx, "p_mw"] = p
                            
                            # Update criticality
                            criticality_map = dict(zip(df_load["bus"], df_load["criticality"]))
                            net.load["bus"] = net.load["bus"].astype(int)
                            net.load["criticality"] = net.load["bus"].map(criticality_map)
                            
                            # Run power flow
                            try:
                                pp.runpp(net)
                            except:
                                business_as_usual_cost[hour] = 0
                                served_load_per_hour.append([None] * len(net.load))
                                gen_per_hour_bau.append([None] * len(net.res_gen))
                                slack_per_hour_bau.append(None)
                                loading_percent_bau.append([None] * (len(net.line) + (len(net.trafo) if df_trafo is not None else 0)))
                                continue
                            
                            # Record loadings
                            intermediate_var = transform_loading(net.res_line["loading_percent"]).copy()
                            if df_trafo is not None:
                                intermediate_var.extend(transform_loading(net.res_trafo["loading_percent"].tolist()))
                            loading_percent_bau.append(intermediate_var)
                            
                            # Check overloads
                            overloads = overloaded_lines(net, max_loading_capacity)
                            overloads_trafo = overloaded_transformer(net, max_loading_capacity_transformer)
                            all_loads_zero_flag = False
                            if not overloads and not overloads_trafo and all_real_numbers(loading_percent_bau[-1]):
                                slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))
                                served_load_per_hour.append(net.load["p_mw"].tolist() if not net.load["p_mw"].isnull().any() else [None] * len(net.load))
                                gen_per_hour_bau.append(net.res_gen["p_mw"].tolist() if not net.res_gen["p_mw"].isnull().any() else [None] * len(net.res_gen))
                                continue
                            
                            # Load shedding loop
                            hour_shed = 0.0
                            while (overloads or overloads_trafo) and not all_loads_zero_flag:
                                for crit in sorted(net.load['criticality'].dropna().unique(), reverse=True):
                                    for ld_idx in net.load[net.load['criticality'] == crit].index:
                                        if not overloads and not overloads_trafo:
                                            break
                                        value = max_loading_capacity_transformer if df_trafo is not None else max_loading_capacity
                                        factor = ((1/500) * value - 0.1)/2
                                        bus = net.load.at[ld_idx, 'bus']
                                        dp = factor * net.load.at[ld_idx, 'p_mw']
                                        hour_shed += dp
                                        dq = factor * net.load.at[ld_idx, 'q_mvar']
                                        net.load.at[ld_idx, 'p_mw'] -= dp
                                        net.load.at[ld_idx, 'q_mvar'] -= dq
                                        cumulative_load_shedding[bus]['p_mw'] += dp
                                        cumulative_load_shedding[bus]['q_mvar'] += dq
                                        hourly_shed_bau[hour] += dp
                                        shedding_buses.append((hour, int(bus)))
                                        try:
                                            try:
                                                pp.runopp(net)
                                                if net.OPF_converged:
                                                    business_as_usual_cost[hour] = net.res_cost
                                            except:
                                                pp.runpp(net)
                                        except:
                                            business_as_usual_cost[hour] = 0
                                            overloads.clear()
                                            if df_trafo is not None:
                                                overloads_trafo.clear()
                                            break
                                        if dp < 0.01:
                                            all_loads_zero_flag = True
                                            business_as_usual_cost[hour] = 0
                                            remaining_p = net.load.loc[net.load["bus"] == bus, "p_mw"].sum()
                                            remaining_q = net.load.loc[net.load["bus"] == bus, "q_mvar"].sum()
                                            cumulative_load_shedding[bus]["p_mw"] += remaining_p
                                            cumulative_load_shedding[bus]["q_mvar"] += remaining_q
                                            hourly_shed_bau[hour] += sum(net.load['p_mw'])
                                            for i in range(len(net.load)):
                                                net.load.at[i, 'p_mw'] = 0
                                                net.load.at[i, 'q_mvar'] = 0
                                            break
                                    if not overloads and not overloads_trafo:
                                        break
                                overloads = overloaded_lines(net, max_loading_capacity)
                                overloads_trafo = overloaded_transformer(net, max_loading_capacity_transformer)
                            
                            # Record final state
                            served_load_per_hour.append(net.load["p_mw"].tolist() if not net.load["p_mw"].isnull().any() else [None] * len(net.load))
                            gen_per_hour_bau.append(net.res_gen["p_mw"].tolist() if not net.res_gen["p_mw"].isnull().any() else [None] * len(net.res_gen))
                            slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]) if not net.res_ext_grid["p_mw"].isnull().any() else None)
                        
                        return (business_as_usual_cost, cumulative_load_shedding, total_demand_per_bus,
                                hourly_shed_bau, served_load_per_hour, gen_per_hour_bau, slack_per_hour_bau,
                                loading_percent_bau, shedding_buses)
                    
                    # CHANGED: Pass required arguments to initialize_network
                    net, load_dynamic, gen_dynamic = initialize_network(df_bus, df_load, df_gen, df_line, df_trafo, df_load_profile, df_gen_profile)
                    num_hours = len(df_load_profile)
                    
                    # Create index maps
                    line_idx_map = {
                        (row["from_bus"], row["to_bus"]): idx for idx, row in net.line.iterrows()
                    }
                    line_idx_map.update({
                        (row["to_bus"], row["from_bus"]): idx for idx, row in net.line.iterrows()
                    })
                    st.session_state.line_idx_map = line_idx_map  # Store in session state
                    trafo_idx_map = {}
                    if df_trafo is not None:
                        trafo_idx_map = {
                            (row["hv_bus"], row["lv_bus"]): idx for idx, row in net.trafo.iterrows()
                        }
                        trafo_idx_map.update({
                            (row["lv_bus"], row["hv_bus"]): idx for idx, row in net.trafo.iterrows()
                        })
                    st.session_state.trafo_idx_map = trafo_idx_map  # Store in session state
                    
                    # Get max loading capacities
                    max_loading_capacity = max(df_line['max_loading_percent'].dropna().tolist())
                    max_loading_capacity_transformer = max(df_trafo['max_loading_percent'].dropna().tolist()) if df_trafo is not None else max_loading_capacity
                    st.session_state.max_loading_capacity = max_loading_capacity  # Store in session state
                    st.session_state.max_loading_capacity_transformer = max_loading_capacity_transformer  # Store in session state
                    
                    # Generate outages
                    line_outages = generate_line_outages(outage_hours, line_down, risk_scores, capped_contingency, df_line=df_line)
                    st.session_state.line_outages = line_outages  # Store in session state
                    
                    # Run simulation
                    (business_as_usual_cost, cumulative_load_shedding, total_demand_per_bus,
                     hourly_shed_bau, served_load_per_hour, gen_per_hour_bau,
                     slack_per_hour_bau, loading_percent_bau, shedding_buses) = run_bau_simulation(
                        net, load_dynamic, gen_dynamic, num_hours, line_outages,
                        max_loading_capacity, max_loading_capacity_transformer
                    )
                    
                    # Store results
                    st.session_state.bau_results = {
                        'business_as_usual_cost': business_as_usual_cost,
                        'cumulative_load_shedding': cumulative_load_shedding,
                        'total_demand_per_bus': total_demand_per_bus,
                        'hourly_shed_bau': hourly_shed_bau,
                        'served_load_per_hour': served_load_per_hour,
                        'gen_per_hour_bau': gen_per_hour_bau,
                        'slack_per_hour_bau': slack_per_hour_bau,
                        'loading_percent_bau': loading_percent_bau,
                        'shedding_buses': shedding_buses
                    }
                    
                except Exception as e:
                    st.error(f"Error running Business As Usual analysis: {str(e)}")
                    st.error(traceback.format_exc())
        
        if st.session_state.bau_results is not None:
            st.subheader("Day End Summary")
            cumulative_load_shedding = st.session_state.bau_results['cumulative_load_shedding']
            total_demand_per_bus = st.session_state.bau_results['total_demand_per_bus']
            if any(v["p_mw"] > 0 or v["q_mvar"] > 0 for v in cumulative_load_shedding.values()):
                summary_data = []
                for bus, shed in cumulative_load_shedding.items():
                    total = total_demand_per_bus.get(bus, {"p_mw": 0.0, "q_mvar": 0.0})
                    summary_data.append({
                        "Bus": bus,
                        "Load Shedding (MWh)": round(shed['p_mw'], 2),
                        "Load Shedding (MVARh)": round(shed['q_mvar'], 2),
                        "Total Demand (MWh)": round(total['p_mw'], 2),
                        "Total Demand (MVARh)": round(total['q_mvar'], 2)
                    })
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.success("No load shedding occurred today.")
            
            st.write("### Hourly Generation Costs")
            business_as_usual_cost = st.session_state.bau_results['business_as_usual_cost']
            cost_data = [{"Hour": i, "Cost (PKR)": round(cost, 2)} for i, cost in enumerate(business_as_usual_cost)]
            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df, use_container_width=True)

        # Visualization cc
    
        # ────────────────────────────────────────────────────────────────────────────
        # Visualisation – Business As Usual (final fix)
        # ────────────────────────────────────────────────────────────────────────────
        st.subheader("Visualize Business As Usual")
        
        # initialise session key that remembers which hour to show
        if "visualize_hour" not in st.session_state:
            st.session_state.visualize_hour = None
        
        if st.session_state.bau_results is None:
            st.info("Please run the Business As Usual analysis first.")
        else:
            num_hours   = len(st.session_state.network_data['df_load_profile'])
            hour_labels = [f"Hour {i}" for i in range(num_hours)]
            picked_hour_label = st.selectbox("Select Hour to Visualize", hour_labels)
            picked_hour       = int(picked_hour_label.split()[-1])
        
            # button only sets the hour to show; the actual drawing happens *outside*
            if st.button("Generate Visualization", key="generate_vis"):
                st.session_state.visualize_hour = picked_hour
        
            # nothing to draw yet
            if st.session_state.visualize_hour is None:
                st.stop()
        
            # ── build / display the map every run ───────────────────────────────────
            hour_idx = st.session_state.visualize_hour
            try:
                # data prep ----------------------------------------------------------
                df_line  = st.session_state.network_data['df_line'].copy()
                df_load  = st.session_state.network_data['df_load'].copy()
                df_trafo = st.session_state.network_data.get('df_trafo')
        
                loading_percent  = st.session_state.bau_results['loading_percent_bau'][hour_idx]
                shedding_buses   = st.session_state.bau_results['shedding_buses']
        
                no_of_lines = len(df_line) if df_trafo is None else len(df_line) - len(df_trafo)
        
                line_idx_map  = st.session_state.get('line_idx_map', {})
                trafo_idx_map = st.session_state.get('trafo_idx_map', {})
        
                # convert geodata to lon/lat tuples & GeoDataFrame
                df_line["geodata"] = df_line["geodata"].apply(
                    lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
                )
                gdf = gpd.GeoDataFrame(
                    df_line,
                    geometry=[LineString(coords) for coords in df_line["geodata"]],
                    crs="EPSG:4326"
                )
                gdf["idx"]     = gdf.index
                gdf["loading"] = gdf["idx"].map(lambda i: loading_percent[i] if i < len(loading_percent) else 0.0)
        
                # lines down because of weather
                weather_down = set()
                if "line_outages" in st.session_state:
                    for (fbus, tbus, start_hr) in st.session_state.line_outages:
                        if hour_idx >= start_hr:
                            is_tf = check_bus_pair(df_line, df_trafo, (fbus, tbus))
                            idx   = trafo_idx_map.get((fbus, tbus)) if is_tf else line_idx_map.get((fbus, tbus))
                            if idx is not None:
                                weather_down.add(idx)
                gdf["down_weather"] = gdf["idx"].isin(weather_down)
        
                # Folium map ---------------------------------------------------------
                m = folium.Map(location=[27.0, 66.5], zoom_start=7, width=800, height=600)
        
                max_line_cap = st.session_state.get('max_loading_capacity',               100.0)
                max_trf_cap  = st.session_state.get('max_loading_capacity_transformer',   max_line_cap)
        
                def col_line(p):   # for lines
                    if p is None or p == 0:           return '#000000'
                    if p <= 0.75 * max_line_cap:      return '#00FF00'
                    if p <= 0.90 * max_line_cap:      return '#FFFF00'
                    if p <  max_line_cap:             return '#FFA500'
                    return '#FF0000'
        
                def col_trf(p):    # for transformers
                    if p is None or p == 0:           return '#000000'
                    if p <= 0.75 * max_trf_cap:       return '#00FF00'
                    if p <= 0.90 * max_trf_cap:       return '#FFFF00'
                    if p <  max_trf_cap:              return '#FFA500'
                    return '#FF0000'
        
                def style_feat(feat):
                    p = feat["properties"]
                    if p.get("down_weather", False):
                        return {"color": "#000000", "weight": 3}
                    pct   = p.get("loading", 0.0)
                    colour = col_trf(pct) if p["idx"] >= no_of_lines and df_trafo is not None else col_line(pct)
                    return {"color": colour, "weight": 3}
        
                folium.GeoJson(
                    gdf.__geo_interface__, name=f"Transmission Net at Hour {hour_idx}",
                    style_function=style_feat
                ).add_to(m)
        
                # load‑bus circles
                shed_now = [b for (h,b) in shedding_buses if h == hour_idx]
                for _, row in df_load.iterrows():
                    bus = row["bus"]
                    lat, lon = ast.literal_eval(row["load_coordinates"])
                    col = "red" if bus in shed_now else "green"
                    folium.Circle((lat, lon), radius=20000,
                                  color=col, fill_color=col, fill_opacity=0.5).add_to(m)
        
                # ---------- legend (replace the whole legend_html string) ------------------
                legend_html = """
                <style>
                  .legend-box,* .legend-box { color:#000 !important; }
                </style>
                
                <div class="legend-box leaflet-control leaflet-bar"
                     style="position:absolute; top:150px; left:10px; z-index:9999;
                            background:#ffffff; padding:8px; border:1px solid #ccc;
                            font-size:14px; max-width:210px;">
                  <strong>Line Load Level&nbsp;(&#37; of Max)</strong><br>
                  <span style='display:inline-block;width:12px;height:12px;background:#00FF00;'></span>&nbsp;Below&nbsp;75&nbsp;%<br>
                  <span style='display:inline-block;width:12px;height:12px;background:#FFFF00;'></span>&nbsp;75–90&nbsp;%<br>
                  <span style='display:inline-block;width:12px;height:12px;background:#FFA500;'></span>&nbsp;90–100&nbsp;%<br>
                  <span style='display:inline-block;width:12px;height:12px;background:#FF0000;'></span>&nbsp;Overloaded&nbsp;>&nbsp;100&nbsp;%<br>
                  <span style='display:inline-block;width:12px;height:12px;background:#000000;'></span>&nbsp;Weather‑Impacted<br><br>
                
                  <strong>Load Status</strong><br>
                  <span style='display:inline-block;width:12px;height:12px;background:#008000;border-radius:50%;'></span>&nbsp;Fully Served<br>
                  <span style='display:inline-block;width:12px;height:12px;background:#FF0000;border-radius:50%;'></span>&nbsp;Not Fully Served
                </div>
                """
                m.get_root().html.add_child(folium.Element(legend_html))

                
                # ---------------- title (overwrite your title_html string) -----------------
                title_html = f"""
                <style>
                  .map-title {{ color:#000 !important; }}
                </style>
                
                <div class="map-title leaflet-control leaflet-bar"
                     style="position:absolute; top:90px; left:10px; z-index:9999;
                            background:rgba(255,255,255,0.9); padding:4px;
                            font-size:18px; font-weight:bold;">
                  Business As Usual – Hour {hour_idx}
                </div>
                """
                m.get_root().html.add_child(folium.Element(title_html))

                folium.LayerControl(collapsed=False).add_to(m)
        
                # display
                st.write(f"### Network Loading Visualization – Hour {hour_idx}")
                st_folium(m, width=800, height=600, key=f"bau_map_{hour_idx}")
        
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
                st.error(traceback.format_exc())


# ────────────────────────────────────────────────────────────────────────────
# Page 4 :  Weather‑Aware System
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Weather Aware System":
    st.title("Weather Aware System")

    # --- sanity checks ------------------------------------------------------
    req_keys = ["network_data", "line_outage_data", "bau_results"]
    if any(k not in st.session_state or st.session_state[k] is None for k in req_keys):
        st.warning(
            "Run **Network Initialization**, **Weather Risk Visualisation**, "
            "and **Business As Usual** first."
        )
        st.stop()

    net_data          = st.session_state.network_data
    df_bus            = net_data["df_bus"]
    df_load           = net_data["df_load"]
    df_gen            = net_data["df_gen"]
    df_line           = net_data["df_line"]
    df_load_profile   = net_data["df_load_profile"]
    df_gen_profile    = net_data["df_gen_profile"]
    df_trafo          = net_data.get("df_trafo")
    num_hours         = len(df_load_profile)

    # contingency selection – same logic as BAU -----------------------------
    cont_mode = st.selectbox(
        "Select Contingency Mode",
        ["Capped Contingency Mode", "Maximum Contingency Mode"],
        help="Capped: ≤ 20 % of lines;  Maximum: all forecast outages."
    )
    capped = cont_mode == "Capped Contingency Mode"

    # build outages with the helper you already have
    line_out_data  = st.session_state.line_outage_data
    line_outages   = generate_line_outages(
        line_out_data["hours"],
        line_out_data["lines"],
        line_out_data["risk_scores"],
        capped_contingency_mode=capped,
        df_line=df_line,
    )

    # run button -------------------------------------------------------------
    if st.button("Run Weather‑Aware Analysis"):
        # -------------------------------------------------------------------- #
        # 1.  rebuild the pandapower network so we start from a fresh copy
        # -------------------------------------------------------------------- #
        net, load_dyn, gen_dyn = initialize_network(
            df_bus, df_load, df_gen,
            df_line, df_trafo,
            df_load_profile, df_gen_profile,
        )

        # maps for quick look‑up (already computed once – reuse)
        line_idx_map  = st.session_state.get("line_idx_map", {})
        trafo_idx_map = st.session_state.get("trafo_idx_map", {})   # ← add this

        # figure out how many plain lines the network has
        no_of_lines = len(df_line) - (len(df_trafo) if df_trafo is not None else 0)

        
        max_line_cap = st.session_state.get("max_loading_capacity", 100.0)
        max_trf_cap  = st.session_state.get("max_loading_capacity_transformer", max_line_cap)
        # storage  -----------------------------------------------------------
        wa_cost                = calculate_hourly_cost(
            net, load_dyn, gen_dyn, num_hours,
            df_load_profile, df_gen_profile
        )
        cumulative_shed        = {b: {"p_mw":0., "q_mvar":0.}
                                  for b in net.load.bus.unique()}
        hourly_shed            = [0.]*num_hours
        loading_percent_wa     = []
        serving_per_hour       = []
        gen_per_hour           = []
        slack_per_hour         = []
        shedding_buses         = []

        # ------------------------------------------------------------------- #
        # 2.  hourly simulation
        # ------------------------------------------------------------------- #
        for hr in range(num_hours):
            # reset all branches each hour (they’ll be taken out again below)
            net.line["in_service"] = True
            if df_trafo is not None:
                net.trafo["in_service"] = True

            # take forecasted lines/trafos out of service
            for fbus, tbus, start_hr in line_outages:
                if hr < start_hr:
                    continue
                is_tf = check_bus_pair(df_line, df_trafo, (fbus, tbus))
                if is_tf:
                    mask = (
                        ((net.trafo.hv_bus == fbus) & (net.trafo.lv_bus == tbus)) |
                        ((net.trafo.hv_bus == tbus) & (net.trafo.lv_bus == fbus))
                    )
                    net.trafo.loc[mask, "in_service"] = False
                else:
                    idx = line_idx_map.get((fbus, tbus))
                    if idx is not None:
                        net.line.at[idx, "in_service"] = False

            # update load & gen profiles for this hour -----------------------
            for idx in net.load.index:
                bus           = net.load.at[idx, "bus"]
                if bus in load_dyn:
                    net.load.at[idx, "p_mw"]   = df_load_profile.at[hr, load_dyn[bus]["p"]]
                    net.load.at[idx, "q_mvar"] = df_load_profile.at[hr, load_dyn[bus]["q"]]
            for idx in net.gen.index:
                bus = net.gen.at[idx, "bus"]
                if bus in gen_dyn:
                    net.gen.at[idx, "p_mw"] = df_gen_profile.at[hr, gen_dyn[bus]]

            # run PF first ---------------------------------------------------
            try:
                pp.runpp(net)
            except Exception:
                # network unsolvable even before optimisation
                loading_percent_wa.append([None]* (len(net.line)+len(net.trafo or [])))
                serving_per_hour.append([None]*len(net.load))
                gen_per_hour.append([None]*len(net.res_gen))
                slack_per_hour.append(None)
                continue

            # helper to push line+trafo loading list -------------------------
            def record_loadings():
                vals = transform_loading(net.res_line.loading_percent)
                if df_trafo is not None:
                    vals.extend(transform_loading(net.res_trafo.loading_percent))
                loading_percent_wa.append(vals)

            # if no overload – keep PF result
            if (overloaded_lines(net, max_line_cap)==[] and
                overloaded_transformer(net, max_trf_cap)==[] and
                all_real_numbers(transform_loading(net.res_line.loading_percent))):
                record_loadings()
                serving_per_hour.append(net.load.p_mw.tolist())
                gen_per_hour.append(net.res_gen.p_mw.tolist())
                slack_per_hour.append(float(net.res_ext_grid.at[0,"p_mw"]))
                continue

            # attempt OPF first ---------------------------------------------
            try:
                pp.runopp(net)
                if net.OPF_converged:
                    record_loadings()
                    wa_cost[hr] = net.res_cost
                    serving_per_hour.append(net.load.p_mw.tolist())
                    gen_per_hour.append(net.res_gen.p_mw.tolist())
                    slack_per_hour.append(float(net.res_ext_grid.at[0,"p_mw"]))
                    continue
            except Exception:
                pass  # fall‑through to load‑shedding loop

            #  load‑shedding loop -------------------------------------------
            while (overloaded_lines(net, max_line_cap) or
                   overloaded_transformer(net, max_trf_cap)):
                for crit in sorted(net.load.criticality.dropna().unique(), reverse=True):
                    for ld_idx in net.load[net.load.criticality==crit].index:
                        if not overloaded_lines(net, max_line_cap) and \
                           not overloaded_transformer(net, max_trf_cap):
                            break
                        val   = max_trf_cap if df_trafo is not None else max_line_cap
                        red_f = ((1/500)*val - .1)/2
                        dp    = red_f * net.load.at[ld_idx, "p_mw"]
                        dq    = red_f * net.load.at[ld_idx, "q_mvar"]
                        bus   = int(net.load.at[ld_idx, "bus"])
                        net.load.at[ld_idx, "p_mw"] -= dp
                        net.load.at[ld_idx, "q_mvar"] -= dq
                        cumulative_shed[bus]["p_mw"] += dp
                        cumulative_shed[bus]["q_mvar"] += dq
                        hourly_shed[hr]                += dp
                        shedding_buses.append((hr, bus))
                    # try OPF / PF again
                    try:
                        pp.runopp(net)
                    except Exception:
                        pp.runpp(net)

            # record final state after shedding -----------------------------
            record_loadings()
            wa_cost[hr] = net.res_cost if net.OPF_converged else 0
            serving_per_hour.append(net.load.p_mw.tolist())
            gen_per_hour.append(net.res_gen.p_mw.tolist())
            slack_per_hour.append(float(net.res_ext_grid.at[0,"p_mw"]))

        # ------------------------------------------------------------------ #
        # 3.   store results in session‑state
        # ------------------------------------------------------------------ #
        st.session_state.weather_aware_results = dict(
            cost               = wa_cost,
            cumulative_shed    = cumulative_shed,
            hourly_shed        = hourly_shed,
            loading_percent    = loading_percent_wa,
            served_load        = serving_per_hour,
            gen_per_hour       = gen_per_hour,
            slack_per_hour     = slack_per_hour,
            shedding_buses     = shedding_buses,
        )
        st.success("Weather‑Aware simulation finished.")


    # ---------------------------------------------------------------------- #
    # 4.  show summary tables if we have results
    # ---------------------------------------------------------------------- #
    if "weather_aware_results" in st.session_state:
        wa_res   = st.session_state.weather_aware_results
        bau_cost = st.session_state.bau_results["business_as_usual_cost"]

        st.subheader("Day‑End Summary (Weather‑Aware)")

        # load‑shedding per bus --------------------------------------------
        shed_tbl = []
        for bus, shed in wa_res["cumulative_shed"].items():
            total_d = st.session_state.bau_results["total_demand_per_bus"].get(bus, {"p_mw":0,"q_mvar":0})
            shed_tbl.append(dict(
                Bus                 = bus,
                **{ "Load Shed (MWh)": round(shed["p_mw"],2),
                   "Load Shed (MVARh)": round(shed["q_mvar"],2),
                   "Total Demand (MWh)": round(total_d["p_mw"],2),
                   "Total Demand (MVARh)": round(total_d["q_mvar"],2)}
            ))
        st.dataframe(pd.DataFrame(shed_tbl), use_container_width=True)

        # hourly gen cost ---------------------------------------------------
        cost_tbl = pd.DataFrame({
            "Hour"                      : list(range(num_hours)),
            "Weather‑Aware Cost (PKR)"  : [round(c,2) for c in wa_res["cost"]],
            "BAU Cost (PKR)"            : [round(c,2) for c in bau_cost],
            "Δ Cost (WA – BAU)"         : [round(w-b,2) for w,b in zip(wa_res["cost"], bau_cost)],
        })
        st.write("### Hourly Generation Cost Comparison")
        st.dataframe(cost_tbl, use_container_width=True)

        # ------------------------------------------------------------------ #
        # 5.  interactive map – structure parallels BAU visualiser
        # ------------------------------------------------------------------ #
        if "wa_vis_hour" not in st.session_state:
            st.session_state.wa_vis_hour = None

        hr_label = st.selectbox(
            "Select Hour to Visualize (Weather‑Aware)",
            [f"Hour {i}" for i in range(num_hours)]
        )
        want_hr  = int(hr_label.split()[-1])

        if st.button("Generate WA Visualization"):
            st.session_state.wa_vis_hour = want_hr

        if st.session_state.wa_vis_hour is not None:
            h = st.session_state.wa_vis_hour
            # --- pull branch‑index maps from session‑state ------------------------
            line_idx_map  = st.session_state.get("line_idx_map", {})
            trafo_idx_map = st.session_state.get("trafo_idx_map", {})
            loadings     = wa_res["loading_percent"][h]
            shed_buses_h = [b for t,b in wa_res["shedding_buses"] if t==h]

            # -----------------------------------------------------------------
            # make sure the capacity limits are in scope *before* helpers exist
            # -----------------------------------------------------------------
            max_line_cap = st.session_state.get("max_loading_capacity", 100.0)
            max_trf_cap  = st.session_state.get(
                               "max_loading_capacity_transformer", max_line_cap)
            # rebuild gdf like we did in BAU visualiser ---------------------
            df_line["geodata"] = df_line.geodata.apply(
                lambda x: [(lo,la) for la,lo in eval(x)] if isinstance(x,str) else x
            )
            gdf = gpd.GeoDataFrame(
                df_line,
                geometry=[LineString(c) for c in df_line.geodata],
                crs="EPSG:4326"
            )
            gdf["idx"]     = gdf.index
            gdf["loading"] = gdf["idx"].map(lambda i: loadings[i] if i < len(loadings) else 0.)
            weather_down = {
            ( trafo_idx_map.get((f, t)) if check_bus_pair(df_line, df_trafo, (f, t))
              else line_idx_map.get((f, t)) )
            for f, t, s in line_outages
            if h >= s
            }
            weather_down.discard(None)
            gdf["down_weather"] = gdf.idx.isin(weather_down)

            # Folium map ----------------------------------------------------
            m = folium.Map(location=[27,66.5], zoom_start=7, width=800, height=600)

            # ---------------------------------------------------------------------------
            # helper: always work with a *scalar* float (or None)
            # ---------------------------------------------------------------------------
            def _first(x):
                """Return a plain float/None even if x is list‑like."""
                if isinstance(x, (list, tuple)) and len(x):
                    return x[0]
                try:                       # NumPy scalar → Python float
                    import numpy as np
                    if isinstance(x, np.generic):
                        return float(x)
                except ImportError:
                    pass
                return x                   # already scalar or None
            
            
            # colour helpers that match the BAU page but are list‑safe
            def col_line(p):
                p = _first(p)
                if p is None or p == 0:            return '#000000'
                if p <= 0.75 * max_line_cap:       return '#00FF00'
                if p <= 0.90 * max_line_cap:       return '#FFFF00'
                if p <  max_line_cap:              return '#FFA500'
                return '#FF0000'
            
            def col_trf(p):
                p = _first(p)
                if p is None or p == 0:            return '#000000'
                if p <= 0.75 * max_trf_cap:        return '#00FF00'
                if p <= 0.90 * max_trf_cap:        return '#FFFF00'
                if p <  max_trf_cap:               return '#FFA500'
                return '#FF0000'

            # ---------------------------------------------------------------------------
            # Weather‑Aware map‑drawing (unchanged except for the safe helpers above)
            # ---------------------------------------------------------------------------
            n_trafos    = len(df_trafo) if df_trafo is not None else 0
            line_cutoff = len(df_line) - n_trafos                 # idx ≥ cut‑off ⇒ trafo
            
            def style_ft(feat):
                prop = feat["properties"]
            
                # black for weather‑impacted
                if prop.get("down_weather", False):
                    return {"color": "#000000", "weight": 3}
            
                pct   = prop.get("loading", 0.0)
                use_t = (df_trafo is not None) and (prop["idx"] >= line_cutoff)
                colour = col_trf(pct) if use_t else col_line(pct)
                return {"color": colour, "weight": 3}
    
            folium.GeoJson(gdf.__geo_interface__,
                           name=f"Transmission Net at Hour {h}",
                           style_function=style_ft).add_to(m)

            # load circles
            for _, r in df_load.iterrows():
                lat, lon = ast.literal_eval(r.load_coordinates)
                col = "red" if r.bus in shed_buses_h else "green"
                folium.Circle((lat,lon), radius=20000,
                              color=col, fill_color=col, fill_opacity=0.5).add_to(m)

            legend_html = """
            <style>
              .legend-box,* .legend-box { color:#000 !important; }
            </style>
            
            <div class="legend-box leaflet-control leaflet-bar"
                 style="position:absolute; top:150px; left:10px; z-index:9999;
                        background:#ffffff; padding:8px; border:1px solid #ccc;
                        font-size:14px; max-width:210px;">
              <strong>Line Load Level&nbsp;(&#37; of Max)</strong><br>
              <span style='display:inline-block;width:12px;height:12px;background:#00FF00;'></span>&nbsp;Below&nbsp;75&nbsp;%<br>
              <span style='display:inline-block;width:12px;height:12px;background:#FFFF00;'></span>&nbsp;75–90&nbsp;%<br>
              <span style='display:inline-block;width:12px;height:12px;background:#FFA500;'></span>&nbsp;90–100&nbsp;%<br>
              <span style='display:inline-block;width:12px;height:12px;background:#FF0000;'></span>&nbsp;Overloaded&nbsp;>&nbsp;100&nbsp;%<br>
              <span style='display:inline-block;width:12px;height:12px;background:#000000;'></span>&nbsp;Weather‑Impacted<br><br>
            
              <strong>Load Status</strong><br>
              <span style='display:inline-block;width:12px;height:12px;background:#008000;border-radius:50%;'></span>&nbsp;Fully Served<br>
              <span style='display:inline-block;width:12px;height:12px;background:#FF0000;border-radius:50%;'></span>&nbsp;Not Fully Served
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            
            # ---------------- title (overwrite your title_html string) -----------------
            title_html = f"""
            <style>
              .map-title {{ color:#000 !important; }}
            </style>
            
            <div class="map-title leaflet-control leaflet-bar"
                 style="position:absolute; top:90px; left:10px; z-index:9999;
                        background:rgba(255,255,255,0.9); padding:4px;
                        font-size:18px; font-weight:bold;">
              Weather Aware – Hour {h}
            </div>
            """
            m.get_root().html.add_child(folium.Element(title_html))

            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, width=800, height=600, key=f"wa_map_{h}")

# ────────────────────────────────────────────────────────────────────────────
# Page 5 :  Data Analytics
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Data Analytics":
    import plotly.graph_objects as go
    import plotly.express        as px
    import numpy                 as np
    st.title("Data Analytics")

    # --------------------------------------------------------------------- #
    # sanity‑check that prerequisites exist
    # --------------------------------------------------------------------- #
    if ("bau_results"           not in st.session_state or
        "weather_aware_results" not in st.session_state or
        st.session_state.bau_results            is None or
        st.session_state.weather_aware_results  is None):
        st.info("Run **Business As Usual** and **Weather‑Aware System** first.")
        st.stop()

    # common data ----------------------------------------------------------
    num_hours        = len(st.session_state.network_data["df_load_profile"])
    hours            = list(range(num_hours))

    load_shed_bau    = st.session_state.bau_results["hourly_shed_bau"]
    load_shed_wa     = st.session_state.weather_aware_results["hourly_shed"]

    cost_bau_raw     = st.session_state.bau_results["business_as_usual_cost"]
    cost_wa_raw      = st.session_state.weather_aware_results["cost"]
    cost_bau_M       = [c/1e6 for c in cost_bau_raw]      # scale to millions
    cost_wa_M        = [c/1e6 for c in cost_wa_raw]

    df_line          = st.session_state.network_data["df_line"]
    loading_bau      = np.array(st.session_state.bau_results["loading_percent_bau"])
    loading_wa       = np.array(st.session_state.weather_aware_results["loading_percent"])

    # line legend helpers
    line_legends     = [f"Line {r['from_bus']}-{r['to_bus']}" for _, r in df_line.iterrows()]
    palette          = px.colors.qualitative.Plotly
    colour_list      = palette * (loading_bau.shape[1] // len(palette) + 1)

    # --------------------------------------------------------------------- #
    # persistent UI flags
    # --------------------------------------------------------------------- #
    for k in ("show_comp", "show_diff", "show_lines"):
        if k not in st.session_state:
            st.session_state[k] = False

    # --------------------------------------------------------------------- #
    # three always‑visible buttons
    # --------------------------------------------------------------------- #
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Plot 1  •  Hourly Comparison"):
            st.session_state.show_comp  = True
    with col2:
        if st.button("Plot 2  •  Cost Difference & Lost Savings"):
            st.session_state.show_diff  = True
    with col3:
        if st.button("Plot 3  •  Line‑Loading‑over‑Time"):
            st.session_state.show_lines = True

    # ===================================================================== #
    # PLOT 1  ─ Hourly load‑shedding + grouped cost bars
    # ===================================================================== #
    if st.session_state.show_comp:
        # — Load‑shedding comparison —
        fig_ls = go.Figure()
        fig_ls.add_trace(go.Scatter(
            x=hours, y=load_shed_bau,
            mode="lines+markers", name="Current OPF Load Shedding",
            line=dict(color="rgba(99,110,250,1)", width=3), marker=dict(size=6)))
        fig_ls.add_trace(go.Scatter(
            x=hours, y=load_shed_wa,
            mode="lines+markers", name="Weather‑Aware Load Shedding",
            line=dict(color="rgba(239,85,59,1)",  width=3), marker=dict(size=6)))
        fig_ls.update_layout(
            title="Hourly Load‑Shedding Comparison",
            xaxis=dict(title="Hour (0–23)", tickmode="linear", dtick=1),
            yaxis_title="Load Shedding [MWh]",
            template="plotly_dark", legend=dict(x=0.01, y=0.99),
            width=1000, height=500, margin=dict(l=60,r=40,t=60,b=50))
        st.plotly_chart(fig_ls, use_container_width=True)

        # — Grouped generation‑cost bars —
        fig_cost = go.Figure()
        fig_cost.add_bar(x=hours, y=cost_bau_M, name="BAU Cost",
                         marker=dict(color="rgba(99,110,250,0.7)"))
        fig_cost.add_bar(x=hours, y=cost_wa_M,  name="Weather‑Aware Cost",
                         marker=dict(color="rgba(239,85,59,0.7)"))
        fig_cost.update_layout(
            barmode="group",
            title="Hourly Generation Cost Comparison",
            xaxis=dict(title="Hour (0–23)", tickmode="linear", dtick=1),
            yaxis_title="Cost [Million PKR]",
            template="plotly_dark", legend=dict(x=0.01,y=0.99),
            width=1000, height=500, margin=dict(l=60,r=40,t=60,b=50))
        st.plotly_chart(fig_cost, use_container_width=True)

    # ===================================================================== #
    # PLOT 2  ─ Cost‑difference *and* Lost‑Savings area
    # ===================================================================== #
    if st.session_state.show_diff:
        # --- cost‑difference shaded area ----------------------------------
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=cost_bau_M + cost_wa_M[::-1],
            fill="toself", fillcolor="rgba(255,140,0,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Loss of Potential Revenue (Cost Difference)",
            hoverinfo="skip"))
        fig_diff.add_trace(go.Scatter(
            x=hours, y=cost_bau_M,
            mode="lines+markers", name="Current OPF Cost",
            line=dict(color="rgba(0,204,150,1)", width=3)))
        fig_diff.add_trace(go.Scatter(
            x=hours, y=cost_wa_M,
            mode="lines+markers", name="Weather‑Aware Cost",
            line=dict(color="rgba(171,99,250,1)", width=3)))
        fig_diff.update_layout(
            title="Potential Lost Revenue – Hourly Cost Difference",
            xaxis=dict(title="Hour (0–23)", tickmode="linear", dtick=1, range=[0,max(hours)]),
            yaxis_title="Cost [Million PKR]",
            template="plotly_dark", legend=dict(x=0.01,y=0.99),
            width=1200, height=500, margin=dict(l=60,r=40,t=60,b=50))
        st.plotly_chart(fig_diff, use_container_width=True)

        # --- lost‑savings‑only area plot ----------------------------------
        lost_sav = [wa - bau if wa > bau else 0 for wa, bau in zip(cost_wa_M, cost_bau_M)]
        fig_lsav = go.Figure()
        fig_lsav.add_trace(go.Scatter(
            x=hours, y=lost_sav, fill="tozeroy",
            fillcolor="rgba(255,99,71,0.6)", mode="none",
            name="Lost Savings Region",
            hovertemplate="Hour %{x}: %{y:.2f} M PKR<extra></extra>"))
        fig_lsav.update_layout(
            title="Potential Lost Revenue (When Weather‑Aware > BAU)",
            xaxis=dict(title="Hour (0–23)", tickmode="linear", dtick=1),
            yaxis_title="Lost Revenue [Million PKR]",
            template="plotly_dark", width=1000, height=500)
        st.plotly_chart(fig_lsav, use_container_width=True)

    # ===================================================================== #
    # PLOT 3  ─ Line‑loading evolution (BAU & WA)
    # ===================================================================== #
    if st.session_state.show_lines:
        x_axis = np.arange(loading_bau.shape[0])

        # — BAU line‑loading —
        fig_bau = go.Figure()
        for idx in range(loading_bau.shape[1]):
            fig_bau.add_trace(go.Scatter(
                x=x_axis, y=loading_bau[:, idx],
                mode="lines",
                name=line_legends[idx],
                line=dict(width=3, color=colour_list[idx], dash="solid")))
        fig_bau.update_layout(
            title="Current OPF Line Loading Over Time",
            template="plotly_dark",
            xaxis_title="Hour",
            yaxis_title="Line Loading [%]",
            xaxis=dict(tickmode="linear", dtick=1),
            plot_bgcolor="rgb(20,20,20)",
            showlegend=True)
        st.plotly_chart(fig_bau, use_container_width=True)

        # — WA line‑loading —
        fig_wa = go.Figure()
        for idx in range(loading_wa.shape[1]):
            fig_wa.add_trace(go.Scatter(
                x=x_axis, y=loading_wa[:, idx],
                mode="lines",
                name=line_legends[idx],
                line=dict(width=3, color=colour_list[idx], dash="dash")))
        fig_wa.update_layout(
            title="Weather‑Aware OPF Line Loading Over Time",
            template="plotly_dark",
            xaxis_title="Hour",
            yaxis_title="Line Loading [%]",
            xaxis=dict(tickmode="linear", dtick=1),
            plot_bgcolor="rgb(20,20,20)",
            showlegend=True)
        st.plotly_chart(fig_wa, use_container_width=True)
