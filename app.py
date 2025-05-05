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
pages = ["Network Initialization", "Weather Risk Visualisation Using GEE", "Business As Usual", "Weather Aware System"]
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

# Helper function to check if all elements in a list are real numbers
def all_real_numbers(lst):
    return all(isinstance(x, (int, float)) and not math.isnan(x) for x in lst)

# Business As Usual Backend Functions
def overloaded_lines(net):
    overloaded = []
    for idx, res in net.res_line.iterrows():
        val = transform_loading(res["loading_percent"])
        if all_real_numbers((net.res_line['loading_percent']).tolist()) == False:
            if not isinstance(val, (int, float)) or math.isnan(val) or val >= max_loading_capacity:
                overloaded.append(idx)
        else:
            if val is not None and not (isinstance(val, float) and math.isnan(val)) and val > max_loading_capacity:
                overloaded.append(idx)
    return overloaded

def overloaded_transformer(net):
    overloaded = []
    if "Transformer Parameters" in pd.ExcelFile(path).sheet_names:
        for idx, res in net.res_trafo.iterrows():
            val = transform_loading(res["loading_percent"])
            if all_real_numbers((net.res_trafo['loading_percent']).tolist()) == False:
                if val is not None and not (isinstance(val, float) and math.isnan(val)) and val > max_loading_capacity_transformer:
                    overloaded.append(idx)
            else:
                if val >= max_loading_capacity_transformer:
                    overloaded.append(idx)
    return overloaded

def Network_initialize(uploaded_file):
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
        pp.create_load(net, bus=row["bus"],
                       p_mw=row["p_mw"],
                       q_mvar=row["q_mvar"],
                       in_service=row["in_service"])

    # --- Create Generators/External Grid ---
    df_slack = pd.read_excel(uploaded_file, sheet_name="Generator Parameters", index_col=0)
    for idx, row in df_slack.iterrows():
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

    # --- Create Lines ---
    df_line = pd.read_excel(uploaded_file, sheet_name="Line Parameters", index_col=0)
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

    xls = pd.ExcelFile(uploaded_file)
    if "Transformer Parameters" in xls.sheet_names:
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

    df_load_profile = pd.read_excel(uploaded_file, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()
    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus = int(m.group(1))
            q_col = f"q_mvar_bus_{bus}"
            if q_col in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": q_col}

    df_gen_profile = pd.read_excel(uploaded_file, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()
    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            numbers = re.findall(r'\d+', col)
            if numbers:
                bus = int(numbers[-1])
                gen_dynamic[bus] = col

    num_hours = len(df_load_profile)

    if "Transformer Parameters" in xls.sheet_names:
        return [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile, df_trafo]
    return [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile]

def calculating_hourly_cost(path):
    xls = pd.ExcelFile(path)
    hourly_cost_list = []
    net = pp.create_empty_network()
    df_bus = pd.read_excel(path, sheet_name="Bus Parameters", index_col=0)
    for idx, row in df_bus.iterrows():
        pp.create_bus(net,
                      name=row["name"],
                      vn_kv=row["vn_kv"],
                      zone=row["zone"],
                      in_service=row["in_service"],
                      max_vm_pu=row["max_vm_pu"],
                      min_vm_pu=row["min_vm_pu"])

    df_load = pd.read_excel(path, sheet_name="Load Parameters", index_col=0)
    for idx, row in df_load.iterrows():
        pp.create_load(net, bus=row["bus"],
                       p_mw=row["p_mw"],
                       q_mvar=row["q_mvar"],
                       in_service=row["in_service"])

    df_slack = pd.read_excel(path, sheet_name="Generator Parameters", index_col=0)
    for idx, row in df_slack.iterrows():
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

    df_line = pd.read_excel(path, sheet_name="Line Parameters", index_col=0)
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

    if "Transformer Parameters" in xls.sheet_names:
        df_trafo = pd.read_excel(path, sheet_name="Transformer Parameters", index_col=0)
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

    df_load_profile = pd.read_excel(path, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()
    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus = int(m.group(1))
            q_col = f"q_mvar_bus_{bus}"
            if q_col in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": q_col}

    df_gen_profile = pd.read_excel(path, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()
    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            numbers = re.findall(r'\d+', col)
            if numbers:
                bus = int(numbers[-1])
                gen_dynamic[bus] = col

    num_hours = len(df_load_profile)

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
        except Exception:
            hourly_cost_list.append(0)
            continue
    return hourly_cost_list

def check_bus_pair(path, bus_pair):
    xls = pd.ExcelFile(path)
    if "Transformer Parameters" in xls.sheet_names:
        transformer_df = pd.read_excel(path, sheet_name='Transformer Parameters')
        line_df = pd.read_excel(path, sheet_name='Line Parameters')
        from_bus, to_bus = bus_pair
        transformer_match = (
            ((transformer_df['hv_bus'] == from_bus) & (transformer_df['lv_bus'] == to_bus)) |
            ((transformer_df['hv_bus'] == to_bus) & (transformer_df['lv_bus'] == from_bus))
        ).any()
        line_match = (
            ((line_df['from_bus'] == from_bus) & (line_df['to_bus'] == to_bus)) |
            ((line_df['from_bus'] == to_bus) & (line_df['to_bus'] == from_bus))
        ).any()
        if transformer_match:
            return True
        if line_match:
            return False
    return None

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

def generate_line_outages(outage_hours, line_down, capped_contingency_mode=False):
    if not outage_hours or not line_down:
        return []
    sheet_name = "Line Parameters"
    df_line = pd.read_excel(path, sheet_name=sheet_name)
    no_of_lines_in_network = len(df_line) - 1
    capped_limit = math.floor(0.2 * no_of_lines_in_network)
    line_outages = []
    for hour, line in zip(outage_hours, line_down):
        from_bus, to_bus = line
        line_outages.append((from_bus, to_bus, hour))
    if capped_contingency_mode and len(line_outages) > capped_limit:
        line_outages = line_outages[:capped_limit]
    return line_outages

# Visualization Functions
def get_color(pct):
    if pct is None:
        return '#FF0000'
    elif pct == 0:
        return '#000000'
    elif pct <= 0.75 * max_loading_capacity:
        return '#00FF00'
    elif pct <= 0.9 * max_loading_capacity:
        return '#FFFF00'
    elif pct < max_loading_capacity:
        return '#FFA500'
    else:
        return '#FF0000'

def get_color_trafo(pct):
    if pct is None:
        return '#FF0000'
    elif pct == 0:
        return '#000000'
    elif pct <= 0.75 * max_loading_capacity_transformer:
        return '#00FF00'
    elif pct <= 0.9 * max_loading_capacity_transformer:
        return '#FFFF00'
    elif pct < max_loading_capacity_transformer:
        return '#FFA500'
    else:
        return '#FF0000'

# Page 1: Network Initialization
if selection == "Network Initialization":
    # Primary project title
    st.title("Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine")

    # Secondary page-specific title
    st.header("Network Initialization")

    # File uploader for the Excel file
    uploaded_file = st.file_uploader("Upload your network Excel file (e.g., IEEE_Bus_Parameters.xlsx)", type=["xlsx"], key="file_uploader")

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

                    # Define the updated process_temperature function
                    def process_temperature(intensity, time_period, risk_score, df_line):
                        # Temperature thresholds for intensity levels
                        thresholds = {"Low": 35, "Medium": 38, "High": 41}
                        thresholds_p = {"Low": 50, "Medium": 100, "High": 150}
                        thresholds_w = {"Low": 10, "Medium": 15, "High": 20}

                        if intensity not in thresholds or time_period not in ["Monthly", "Weekly"]:
                            raise ValueError("Invalid intensity or time period")

                        # Use df_line from session state instead of loading from Excel
                        df = df_line.copy()

                        from_buses = df["from_bus"].tolist()
                        to_buses = df["to_bus"].tolist()
                        all_lines = list(df[["from_bus", "to_bus"]].itertuples(index=False, name=None))

                        df["geodata"] = df["geodata"].apply(lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x)
                        line_geometries = [LineString(coords) for coords in df["geodata"]]
                        gdf = gpd.GeoDataFrame(df, geometry=line_geometries, crs="EPSG:4326")

                        # Create Folium map instead of geemap
                        m = folium.Map(location=[30, 70], zoom_start=5, width=800, height=600)

                        # Convert transmission lines to EE FeatureCollection
                        features = [
                            ee.Feature(ee.Geometry.LineString(row["geodata"]), {
                                "line_id": i,
                                "geodata": str(row["geodata"])
                            }) for i, row in df.iterrows()
                        ]
                        line_fc = ee.FeatureCollection(features)

                        # Add transmission lines using Folium's add_ee_layer
                        m.add_ee_layer(line_fc.style(**{'color': 'blue', 'width': 2}), {}, "Transmission Lines")

                        # Define date range (last 10 years)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365 * 10)

                        # Select dataset based on time period
                        dataset_name = "ECMWF/ERA5/MONTHLY" if time_period == "Monthly" else "ECMWF/ERA5_LAND/DAILY_AGGR"
                        dataset = ee.ImageCollection(dataset_name).filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                        dataset_forecast = ee.ImageCollection("NOAA/GFS0P25")
                        d = dataset_forecast.first()

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

                        mid1_ft = thresholds[intensity] * 0.90
                        mid2_ft = thresholds[intensity] * 0.80
                        mid1_fw = thresholds_w[intensity] * 0.90
                        mid2_fw = thresholds_w[intensity] * 0.80
                        mid1_fp = thresholds_p[intensity] * 0.90
                        mid2_fp = thresholds_p[intensity] * 0.80

                        now = datetime.utcnow()
                        nearest_gfs_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
                        future = nearest_gfs_time + timedelta(hours=24)
                        now_str = nearest_gfs_time.strftime('%Y-%m-%dT%H:%M:%S')
                        future_str = future.strftime('%Y-%m-%dT%H:%M:%S')

                        latest_image = dataset_forecast.sort("system:time_start", False).first()
                        latest_timestamp = latest_image.date().format().getInfo()

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

                        m.add_ee_layer(classified_t, classified_viz, f"Temperature Occurrence Classification ({time_period})")
                        m.add_ee_layer(classified_p, classified_viz, f"Precipitation Occurrence Classification ({time_period})")
                        m.add_ee_layer(classified_w, classified_viz, f"Wind Occurrence Classification ({time_period})")
                        m.add_ee_layer(combined_layer, vis_params, "Combined Historic Classification")

                        fut = [latest_timestamp]

                        daily_dfs = {}
                        results_per_day = []
                        risk_scores = []

                        for i in range(1, 2):
                            future = nearest_gfs_time + timedelta(hours=24 * i)
                            future_str = future.strftime('%Y-%m-%dT%H:%M:%S')
                            fut.append(future_str)

                            forecast_24h = dataset_forecast.filterDate(latest_timestamp, future_str)

                            forecast_temp = forecast_24h.select(temp_forecast).max().rename(f"forecast_temp_C_day_{i}")
                            forecast_u = forecast_24h.select(u_forecast).max().rename(f"forecast_u_day_{i}")
                            forecast_v = forecast_24h.select(v_forecast).max().rename(f"forecast_v_day_{i}")
                            forecast_pre = forecast_24h.select(precip_forecast).max().multiply(86400).rename(f"forecast_prec_day_{i}")

                            forecast_wind_magnitude = forecast_u.expression(
                                "sqrt(pow(u, 2) + pow(v, 2))",
                                {"u": forecast_u, "v": forecast_v}
                            ).rename(f"forecast_wind_magnitude_day_{i}")

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

                            m.add_ee_layer(combined_forecast, combined_viz, "Day Ahead - Risk Score")

                            reduced = combined_forecast.reduceRegions(
                                collection=line_fc,
                                reducer=ee.Reducer.max(),
                                scale=1000
                            )

                            results = reduced.getInfo()

                            data = []
                            daily_results = []

                            for feature in results["features"]:
                                props = feature["properties"]
                                line_id = props["line_id"]
                                max_risk = props.get("max", 0)
                                from_bus = df.loc[line_id, "from_bus"]
                                to_bus = df.loc[line_id, "to_bus"]
                                daily_results.append((int(from_bus), int(to_bus), int(max_risk)))
                                data.append({
                                    "line_id": line_id,
                                    "geodata": props["geodata"],
                                    "risk_score": max_risk
                                })
                                risk_scores.append(max_risk)

                            results_per_day.append(daily_results)
                            daily_dfs[f"Day_{i}"] = pd.DataFrame(data)

                        day_1_results = results_per_day[0]
                        filtered_lines_day1 = [(from_bus, to_bus) for from_bus, to_bus, score in day_1_results if score >= risk_score]
                        length_lines = len(filtered_lines_day1)
                        outage_hour_day = [random.randint(11, 15) for _ in range(length_lines)]

                        # Add layer control
                        folium.LayerControl(collapsed=False).add_to(m)

                        return m, filtered_lines_day1, outage_hour_day, risk_scores

                    # Call the function with selected parameters
                    weather_map, filtered_lines_day1, outage_hour_day, risk_scores = process_temperature(
                        intensity,
                        study_period,
                        risk_score,
                        st.session_state.network_data['df_line']
                    )

                    # Construct additional outputs to match original app
                    risk_df = pd.DataFrame({
                        "line_id": range(len(st.session_state.network_data['df_line'])),
                        "from_bus": st.session_state.network_data['df_line']["from_bus"],
                        "to_bus": st.session_state.network_data['df_line']["to_bus"],
                        "risk_score": risk_scores
                    })

                    line_outage_data = {
                        "lines": filtered_lines_day1,
                        "hours": outage_hour_day,
                        "risk_scores": [{"line_id": i, "from_bus": f, "to_bus": t, "risk_score": s} 
                                       for i, (f, t, s) in enumerate(zip(st.session_state.network_data['df_line']["from_bus"], 
                                                                         st.session_state.network_data['df_line']["to_bus"], 
                                                                         risk_scores))]
                    }

                    outage_data = [
                        {"line": f"From Bus {from_bus} to Bus {to_bus}", "outage_hours": hours, "risk_score": score}
                        for (from_bus, to_bus), hours, score in zip(filtered_lines_day1, outage_hour_day, 
                                                                    [s for _, _, s in [(f, t, s) for f, t, s in zip(st.session_state.network_data['df_line']["from_bus"], 
                                                                                                                    st.session_state.network_data['df_line']["to_bus"], 
                                                                                                                    risk_scores)] if (f, t) in filtered_lines_day1])
                    ]

                    # Mock max occurrences since they're not returned by the new function
                    max_occurrence_t = max_occurrence_p = max_occurrence_w = 0

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
                    risk_df_display = st.session_state.risk_df[["line_id", "from_bus", "to_bus", "risk_score"]].sort_values(by="risk_score", ascending=False)
                    risk_df_display.columns = ["Line ID", "From Bus", "To Bus", "Risk Score"]
                    st.dataframe(risk_df_display, use_container_width=True)

                    # Display lines expected to face outage based on threshold
                    if st.session_state.outage_data:
                        st.write(f"### Lines Expected to Face Outage (Risk Score ≥ {risk_score})")
                        outage_df = pd.DataFrame(st.session_state.outage_data)
                        outage_df.columns = ["Transmission Line", "Expected Outage Hours", "Risk Score"]
                        st.dataframe(outage_df, use_container_width=True)

                        # Summary statistics
                        st.write("### Outage Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Number of Lines at Risk", len(st.session_state.outage_data))
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

def Network_initialize(uploaded_file):
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
        pp.create_load(net, bus=row["bus"],
                       p_mw=row["p_mw"],
                       q_mvar=row["q_mvar"],
                       in_service=row["in_service"])

    # --- Create Generators/External Grid ---
    df_slack = pd.read_excel(uploaded_file, sheet_name="Generator Parameters", index_col=0)
    for idx, row in df_slack.iterrows():
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

    # --- Create Lines ---
    df_line = pd.read_excel(uploaded_file, sheet_name="Line Parameters", index_col=0)
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

    xls = pd.ExcelFile(uploaded_file)
    if "Transformer Parameters" in xls.sheet_names:
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

    df_load_profile = pd.read_excel(uploaded_file, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()
    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus = int(m.group(1))
            q_col = f"q_mvar_bus_{bus}"
            if q_col in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": q_col}

    df_gen_profile = pd.read_excel(uploaded_file, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()
    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            numbers = re.findall(r'\d+', col)
            if numbers:
                bus = int(numbers[-1])
                gen_dynamic[bus] = col

    num_hours = len(df_load_profile)

    if "Transformer Parameters" in xls.sheet_names:
        return [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile, df_trafo]
    return [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile]

if selection == "Business As Usual":
    st.title("Business As Usual Analysis")

    if "line_outage_data" not in st.session_state or st.session_state.line_outage_data is None:
        st.warning("Please run the Weather Risk Visualisation Using GEE page first to generate outage data.")
    else:
        contingency_mode = st.selectbox(
            "Select Contingency Mode",
            options=["Capped Contingency Mode", "Maximum Contingency Mode"],
            help="Capped: Limits outages to 20% of lines. Maximum: Allows all outages."
        )
        capped_mode = True if contingency_mode == "Capped Contingency Mode" else False

        if st.button("Run Business As Usual Analysis"):
            with st.spinner("Running Business As Usual analysis..."):
                try:
                    outage_hours = st.session_state.line_outage_data["hours"]
                    line_down = st.session_state.line_outage_data["lines"]
                    uploaded_file = st.session_state.uploaded_file
                    if uploaded_file is None:
                        raise ValueError("No uploaded file found in session state.")
                    xls = pd.ExcelFile(uploaded_file)
                    if "Transformer Parameters" in xls.sheet_names:
                        [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile, df_trafo] = Network_initialize(uploaded_file)
                    else:
                        [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile] = Network_initialize(uploaded_file)
                    business_as_usual_cost = calculating_hourly_cost(uploaded_file)

                    df_lines = df_line.copy()
                    df_lines["geodata"] = df_lines["geodata"].apply(
                        lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
                    )
                    global max_loading_capacity
                    max_loading_capacity = max(df_lines['max_loading_percent'].dropna().tolist())
                    global max_loading_capacity_transformer
                    if "Transformer Parameters" in xls.sheet_names:
                        max_loading_capacity_transformer = max(df_trafo['max_loading_percent'].dropna().tolist())

                    gdf = gpd.GeoDataFrame(
                        df_lines,
                        geometry=[LineString(coords) for coords in df_lines["geodata"]],
                        crs="EPSG:4326"
                    )

                    load_df = pd.read_excel(uploaded_file, sheet_name='Load Parameters')
                    load_df['coordinates'] = load_df['load_coordinates'].apply(lambda x: ast.literal_eval(x))

                    line_idx_map = {
                        (row["from_bus"], row["to_bus"]): idx
                        for idx, row in net.line.iterrows()
                    }
                    line_idx_map.update({
                        (row["to_bus"], row["from_bus"]): idx
                        for idx, row in net.line.iterrows()
                    })

                    if "Transformer Parameters" in xls.sheet_names:
                        trafo_idx_map = {
                            (row["hv_bus"], row["lv_bus"]): idx
                            for idx, row in net.trafo.iterrows()
                        }
                        trafo_idx_map.update({
                            (row["lv_bus"], row["hv_bus"]): idx
                            for idx, row in net.trafo.iterrows()
                        })

                    net.load["bus"] = net.load["bus"].astype(int)
                    cumulative_load_shedding = {
                        bus: {"p_mw": 0.0, "q_mvar": 0.0} for bus in net.load["bus"].unique()
                    }

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
                    loading_records = []
                    loading_percent_bau = []
                    served_load_per_hour = []
                    gen_per_hour_bau = []
                    slack_per_hour_bau = []
                    shedding_buses = []
                    seen_buses = set()

                    line_outages = generate_line_outages(outage_hours, line_down, capped_mode)

                    for hour in range(num_hours):
                        for (fbus, tbus, start_hr) in line_outages:
                            if hour < start_hr:
                                continue
                            is_trafo = check_bus_pair(uploaded_file, (fbus, tbus))
                            if is_trafo:
                                mask_tf = (
                                    ((net.trafo.hv_bus == fbus) & (net.trafo.lv_bus == tbus)) |
                                    ((net.trafo.hv_bus == tbus) & (net.trafo.lv_bus == fbus))
                                )
                                if mask_tf.any():
                                    for tf_idx in net.trafo[mask_tf].index:
                                        net.trafo.at[tf_idx, "in_service"] = False
                            else:
                                idx = line_idx_map.get((fbus, tbus))
                                if idx is not None:
                                    net.line.at[idx, "in_service"] = False

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

                        df_load_params = pd.read_excel(uploaded_file, sheet_name="Load Parameters", index_col=0)
                        criticality_map = dict(zip(df_load_params["bus"], df_load_params["criticality"]))
                        net.load["criticality"] = net.load["bus"].map(criticality_map)

                        try:
                            pp.runpp(net)
                        except:
                            continue

                        intermediate_cont = transform_loading(net.res_line["loading_percent"])
                        if "Transformer Parameters" in xls.sheet_names:
                            intermediate_cont.extend(transform_loading(net.res_trafo["loading_percent"].tolist()))
                        loading_records.append(transform_loading(intermediate_cont))

                        intermediate_var = transform_loading(net.res_line["loading_percent"])
                        if "Transformer Parameters" in xls.sheet_names:
                            intermediate_var.extend(transform_loading(net.res_trafo["loading_percent"].tolist()))
                        loading_percent_bau.append(intermediate_var)

                        overloads = overloaded_lines(net)
                        overloads_trafo = overloaded_transformer(net)
                        all_loads_zero_flag = False

                        if not overloads and not overloads_trafo and all_real_numbers(loading_records[-1]):
                            slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))
                            served_load_per_hour.append(net.load["p_mw"].tolist() if not net.load["p_mw"].isnull().any() else [None] * len(net.load))
                            gen_per_hour_bau.append(net.res_gen["p_mw"].tolist() if not net.res_gen["p_mw"].isnull().any() else [None] * len(net.res_gen))
                            continue
                        else:
                            hour_shed = 0.0
                            while (overloads or overloads_trafo) and not all_loads_zero_flag:
                                for crit in sorted(net.load['criticality'].dropna().unique(), reverse=True):
                                    for ld_idx in net.load[net.load['criticality'] == crit].index:
                                        if not overloads and not overloads_trafo:
                                            break
                                        value = max_loading_capacity_transformer if "Transformer Parameters" in xls.sheet_names else max_loading_capacity
                                        factor = ((1/500) * value - 0.1) / 2
                                        bus = net.load.at[ld_idx, 'bus']
                                        dp = factor * net.load.at[ld_idx, 'p_mw']
                                        hour_shed += dp
                                        dq = factor * net.load.at[ld_idx, 'q_mvar']
                                        net.load.at[ld_idx, 'p_mw'] -= dp
                                        net.load.at[ld_idx, 'q_mvar'] -= dq
                                        shedding_buses.append((hour, int(bus)))
                                        cumulative_load_shedding[bus]['p_mw'] += dp
                                        cumulative_load_shedding[bus]['q_mvar'] += dq
                                        hourly_shed_bau[hour] += dp
                                        try:
                                            pp.runpp(net)
                                        except:
                                            overloads.clear()
                                            if "Transformer Parameters" in xls.sheet_names:
                                                overloads_trafo.clear()
                                            break
                                        if dp < 0.01:
                                            all_loads_zero_flag = True
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

                        served_load_per_hour.append(net.load["p_mw"].tolist() if not net.load["p_mw"].isnull().any() else [None] * len(net.load))
                        gen_per_hour_bau.append(net.res_gen["p_mw"].tolist() if not net.res_gen["p_mw"].isnull().any() else [None] * len(net.res_gen))
                        slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))

                    summary_data = []
                    for bus, shed in cumulative_load_shedding.items():
                        total = total_demand_per_bus.get(bus, {"p_mw": 0.0, "q_mvar": 0.0})
                        if shed["p_mw"] > 0 or shed["q_mvar"] > 0:
                            summary_data.append({
                                "Load Bus": bus,
                                "Load Shedding (MWh)": shed["p_mw"],
                                "Load Shedding (MVARh)": shed["q_mvar"],
                                "Total Demand (MWh)": total["p_mw"],
                                "Total Demand (MVARh)": total["q_mvar"]
                            })

                    st.session_state.bau_summary = pd.DataFrame(summary_data) if summary_data else None
                    st.session_state.loading_records = loading_records
                    st.session_state.df_lines = df_lines
                    st.session_state.df_trafo = df_trafo if "Transformer Parameters" in xls.sheet_names else None

                    st.subheader("Day-End Summary")
                    if st.session_state.bau_summary is not None and not st.session_state.bau_summary.empty:
                        st.dataframe(st.session_state.bau_summary, use_container_width=True)
                    else:
                        st.success("No load shedding occurred today.")
                except Exception as e:
                    st.error(f"Error running Business As Usual analysis: {str(e)}")

        if st.session_state.bau_summary is not None:
            visualize = st.selectbox(
                "Visualize Business As Usual Using GEE?",
                options=["No", "Yes"],
                help="Select Yes to visualize line loading for a specific hour."
            )

            if visualize == "Yes" and st.session_state.loading_records:
                hour_options = list(range(num_hours))
                selected_hour = st.selectbox(
                    "Select Hour to Visualize",
                    options=hour_options,
                    help="Choose the hour to visualize line loading."
                )

                if st.button(f"Generate Visualization for Hour {selected_hour}"):
                    with st.spinner("Generating visualization..."):
                        try:
                            line_outages = [(line[0], line[1], hour) for line, hour in zip(st.session_state.line_outage_data["lines"], st.session_state.line_outage_data["hours"]) if hour <= selected_hour]
                            weather_down_set = set()
                            for (fbus, tbus, start_hr) in line_outages:
                                if selected_hour >= start_hr:
                                    is_trafo = check_bus_pair(uploaded_file, (fbus, tbus))
                                    idx = trafo_idx_map[(fbus, tbus)] if is_trafo and "Transformer Parameters" in xls.sheet_names else line_idx_map[(fbus, tbus)]
                                    weather_down_set.add(idx)

                            gdf_hour = st.session_state.df_lines.copy()
                            gdf_hour["idx"] = gdf_hour.index
                            gdf_hour["loading"] = gdf_hour["idx"].map(
                                lambda i: st.session_state.loading_records[selected_hour][i] if i < len(st.session_state.loading_records[selected_hour]) else 0.0
                            )
                            gdf_hour["down_weather"] = gdf_hour["idx"].apply(lambda i: i in weather_down_set)
                            geojson = gdf_hour.__geo_interface__

                            def style_fn_dynamic(feat):
                                props = feat["properties"]
                                if props.get("down_weather", False):
                                    return {"color": "#000000", "weight": 3}
                                pct = props.get("loading", 0.0)
                                use_trafo = len(st.session_state.loading_records[0]) > len(st.session_state.df_lines) - (len(st.session_state.df_trafo) if st.session_state.df_trafo else 0)
                                color = get_color_trafo(pct) if use_trafo else get_color(pct)
                                return {"color": color, "weight": 3}

                            m = folium.Map(location=[27.0, 66.5], zoom_start=7, width=800, height=600)
                            folium.GeoJson(
                                geojson,
                                name=f'Transmission Net at Hour {selected_hour}',
                                style_function=style_fn_dynamic
                            ).add_to(m)

                            shedding_at_hr = [bus for (shed_hr, bus) in shedding_buses if shed_hr == selected_hour]
                            for bus in load_df['bus'].dropna().values:
                                lat, lon = ast.literal_eval(load_df.loc[load_df['bus']==bus, 'load_coordinates'].values[0])
                                color = "red" if bus in shedding_at_hr else "green"
                                folium.Circle(
                                    location=(lat, lon),
                                    radius=20000,
                                    color=color,
                                    fill_color=color,
                                    fill_opacity=0.5
                                ).add_to(m)

                            html_legend = """
                            <style>
                            .legend-entry { display: flex; align-items: center; margin-bottom: 4px; }
                            .legend-swatch { width: 12px; height: 12px; margin-right: 6px; display: inline-block; }
                            .circle { border-radius: 70%; }
                            </style>
                            <div style="background:white; padding:8px; border:1px solid #ccc;">
                              <strong>Line Load Level (% of Max) and Load Status</strong>
                              <div class="legend-entry"><span class="legend-swatch" style="background:#00FF00"></span>Below 75%</div>
                              <div class="legend-entry"><span class="legend-swatch" style="background:#FFFF00"></span>75-90%</div>
                              <div class="legend-entry"><span class="legend-swatch" style="background:#FFA500"></span>90-100%</div>
                              <div class="legend-entry"><span class="legend-swatch" style="background:#FF0000"></span>Overloaded Line (>100%)</div>
                              <div class="legend-entry"><span class="legend-swatch" style="background:#000000"></span>Weather Impacted Line</div>
                              <div class="legend-entry"><span class="legend-swatch circle" style="background:#008000"></span>Fully Served Load</div>
                              <div class="legend-entry"><span class="legend-swatch circle" style="background:#FF0000"></span>Not Fully Served Load</div>
                            </div>
                            """
                            m.get_root().html.add_child(folium.Element(html_legend))

                            title = folium.Element(f"""<div style='font-size:18px;
                                                    font-weight:bold;
                                                    background-color:rgba(255,255,255,0.8);
                                                    padding:4px;'>
                                            Business As Usual: Hour {selected_hour}</div>""")
                            m.get_root().html.add_child(title)

                            st_folium(m, width=800, height=600, key=f"bau_map_{selected_hour}")
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
        else:
            st.info("Run the analysis or generate visualization data first.")

# Page 4: Weather Aware System
elif selection == "Weather Aware System":
    st.title("Weather Aware System")

    # Check if required data is available
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file is None:
        st.warning("Please upload an Excel file on the Network Initialization page first.")
    elif "network_data" not in st.session_state or st.session_state.network_data is None:
        st.warning("Please upload and initialize network data on the Network Initialization page first.")
    elif "line_outage_data" not in st.session_state or st.session_state.line_outage_data is None:
        st.warning("Please process weather risk data on the Weather Risk Visualisation Using GEE page first.")
    elif "business_as_usual_cost" not in st.session_state:
        st.warning("Please run the Business As Usual analysis first to compare costs.")
    else:
        # Dropdown for contingency type
        contingency_type = st.selectbox(
            "Select Contingency Type",
            options=["N-1 Contingency", "N-M Contingency"],
            help="N-1: Simulate outage of a single line. N-M: Simulate outage of all lines projected to be down based on weather risk."
        )

        # Determine single_contingency parameter
        single_contingency = (contingency_type == "N-1 Contingency")

        # Extract line outages and hours from session state (from Page 2)
        line_down = st.session_state.line_outage_data.get("lines", [])
        outage_hours = st.session_state.line_outage_data.get("hours", [])

        # Generate line outages based on contingency type
        line_outages = generate_line_outages(outage_hours, line_down, single_contingency)

        # Button to run the simulation
        if st.button("Run Weather Aware Analysis"):
            with st.spinner("Running Weather Aware Analysis..."):
                try:
                    # Reinitialize the network using the uploaded file
                    uploaded_file = st.session_state.uploaded_file
                    if uploaded_file is None:
                        st.error("Uploaded file not found in session state.")
                    else:
                        [net, df_bus, df_slack, df_line, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile] = Network_initialize(uploaded_file)

                        # Precompute hourly generation costs
                        weather_aware_cost = calculating_hourly_cost(uploaded_file, net, num_hours, load_dynamic, gen_dynamic, df_load_profile, df_gen_profile)

                        # Read line parameters & build GeoDataFrame
                        df_lines = df_line.copy()
                        df_lines["geodata"] = df_lines["geodata"].apply(
                            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
                        )
                        gdf = gpd.GeoDataFrame(
                            df_lines,
                            geometry=[LineString(coords) for coords in df_lines["geodata"]],
                            crs="EPSG:4326"
                        )

                        # Load data for coordinates
                        load_df = pd.read_excel(uploaded_file, sheet_name='Load Parameters')
                        load_df['coordinates'] = load_df['load_coordinates'].apply(lambda x: ast.literal_eval(x))

                        # Create the line index map (bi-directional)
                        line_idx_map = {
                            (row["from_bus"], row["to_bus"]): idx
                            for idx, row in net.line.iterrows()
                        }
                        line_idx_map.update({
                            (row["to_bus"], row["from_bus"]): idx
                            for idx, row in net.line.iterrows()
                        })

                        # Ensure load.bus is integer
                        net.load["bus"] = net.load["bus"].astype(int)

                        # Containers for storing results
                        loading_records = []
                        shedding_buses = []
                        seen_buses = set()
                        cumulative_load_shedding = {
                            bus: {"p_mw": 0.0, "q_mvar": 0.0} for bus in net.load["bus"].unique()
                        }
                        hourly_shed_weather_aware = [0] * num_hours
                        max_loading_capacity = 100

                        # Hourly simulation: PF → conditional OPF → record
                        for hour in range(num_hours):
                            # Take lines out of service as soon as their start hour arrives
                            for fbus, tbus, start_hr in line_outages:
                                if hour >= start_hr:
                                    idx = line_idx_map.get((fbus, tbus))
                                    if idx is not None:
                                        net.line.at[idx, "in_service"] = False

                            # Update load profiles for this hour
                            for idx in net.load.index:
                                bus = net.load.at[idx, "bus"]
                                if bus in load_dynamic:
                                    pcol, qcol = load_dynamic[bus]["p"], load_dynamic[bus]["q"]
                                    net.load.at[idx, "p_mw"] = df_load_profile.at[hour, pcol]
                                    net.load.at[idx, "q_mvar"] = df_load_profile.at[hour, qcol]

                            # Read updated criticality from Excel
                            df_load_params = pd.read_excel(uploaded_file, sheet_name="Load Parameters", index_col=0)
                            criticality_map = dict(zip(df_load_params["bus"], df_load_params["criticality"]))
                            net.load["bus"] = net.load["bus"].astype(int)
                            net.load["criticality"] = net.load["bus"].map(criticality_map)

                            # Update PV/gen profiles for this hour
                            planned_gen_output = {}
                            for idx in net.gen.index:
                                bus = net.gen.at[idx, "bus"]
                                if bus in gen_dynamic:
                                    p = df_gen_profile.at[hour, gen_dynamic[bus]]
                                    net.gen.at[idx, "p_mw"] = p
                                    planned_gen_output[idx] = p

                            # Run initial power flow
                            try:
                                pp.runpp(net)
                            except Exception:
                                loading_records.append(net.res_line["loading_percent"].copy())
                                continue

                            # Check for overloads
                            ovl = overloaded_lines(net)
                            if not ovl:
                                loading_records.append(net.res_line["loading_percent"].copy())
                                continue

                            # Record planned slack output
                            planned_slack = {}
                            if not net.ext_grid.empty:
                                for idx in net.ext_grid.index:
                                    pw = "p_mw" if "p_mw" in net.res_ext_grid else "p_kw"
                                    planned_slack[idx] = net.res_ext_grid.at[idx, pw]

                            # Take a copy of the last PF result
                            pf_loadings = net.res_line["loading_percent"].copy()

                            # Run OPF with load shedding if necessary
                            opf_attempts = 0
                            max_opf_attempts = len(net.load) * 5
                            while opf_attempts < max_opf_attempts:
                                try:
                                    pp.runopp(net)
                                    weather_aware_cost[hour] = net.res_cost
                                    break
                                except Exception:
                                    opf_attempts += 1
                                    # Determine overloads from the PF snapshot
                                    ovl = [idx for idx, loading in pf_loadings.items()
                                           if loading > net.line.at[idx, "max_loading_percent"]]
                                    if not ovl:
                                        break

                                    for crit in sorted(net.load['criticality'].dropna().unique(), reverse=True):
                                        for ld_idx in net.load[net.load['criticality'] == crit].index:
                                            bus = net.load.at[ld_idx, 'bus']
                                            dp = 0.1 * net.load.at[ld_idx, 'p_mw']
                                            hourly_shed_weather_aware[hour] += dp
                                            shedding_buses.append((hour, int(bus)))
                                            dq = 0.1 * net.load.at[ld_idx, 'q_mvar']
                                            net.load.at[ld_idx, 'p_mw'] -= dp
                                            net.load.at[ld_idx, 'q_mvar'] -= dq
                                            bus_id = net.load.at[ld_idx, 'bus']
                                            cumulative_load_shedding[bus_id]["p_mw"] += dp
                                            cumulative_load_shedding[bus_id]["q_mvar"] += dq

                                    # After shedding, re-run PF to refresh pf_loadings
                                    try:
                                        pp.runpp(net)
                                        pf_loadings = net.res_line["loading_percent"].copy()
                                    except Exception:
                                        break

                            # Record post-OPF loadings
                            loading_records.append(net.res_line["loading_percent"].copy())

                        # Store results in session state
                        st.session_state.weather_aware_results = {
                            "loading_records": loading_records,
                            "line_outages": line_outages,
                            "shedding_buses": shedding_buses,
                            "gdf": gdf,
                            "load_df": load_df,
                            "line_idx_map": line_idx_map
                        }
                        st.session_state.weather_aware_cumulative_load_shedding = cumulative_load_shedding
                        st.session_state.weather_aware_cost = weather_aware_cost

                except Exception as e:
                    st.error(f"Error running Weather Aware analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

        # Display results if available
        if "weather_aware_cumulative_load_shedding" in st.session_state and "weather_aware_cost" in st.session_state:
            st.subheader("Load Shedding Summary")
            st.write("At the end of the day, the load shedding at each Load Bus is as follows:")
            shedding_data = []
            for bus, data in st.session_state.weather_aware_cumulative_load_shedding.items():
                if data["p_mw"] > 0 or data["q_mvar"] > 0:  # Only show buses with shedding
                    shedding_data.append({
                        "Load Bus": bus,
                        "Total Load Shed (MWh)": round(data['p_mw'], 2),
                        "Total Reactive Load Shed (MVARh)": round(data['q_mvar'], 2)
                    })
            if shedding_data:
                shedding_df = pd.DataFrame(shedding_data)
                st.dataframe(shedding_df, use_container_width=True)
            else:
                st.write("No load shedding occurred.")

            st.subheader("Hourly Generation Cost (PKR) Comparison")
            cost_data = []
            for i in range(len(st.session_state.weather_aware_cost)):
                weather_cost = st.session_state.weather_aware_cost[i]
                bau_cost = st.session_state.business_as_usual_cost[i]
                cost_diff = weather_cost - bau_cost
                cost_data.append({
                    "Hour": i,
                    "Weather Aware Cost (PKR)": round(weather_cost, 2),
                    "Business As Usual Cost (PKR)": round(bau_cost, 2),
                    "Cost Difference (PKR)": round(cost_diff, 2)
                })
            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df, use_container_width=True)

        # Option to visualize 24-hour load profile via GEE
        if "weather_aware_results" in st.session_state:
            visualize_maps = st.checkbox("Visualize 24-Hour Load Profile via GEE", help="Generate 24 hourly maps showing line loading and load shedding.")
            if visualize_maps:
                # Dropdown to select the hour to display
                hour_to_display = st.selectbox("Select Hour to Display", options=list(range(24)), key="weather_aware_hour_select")

                with st.spinner(f"Generating map for Hour {hour_to_display}..."):
                    try:
                        # Retrieve uploaded_file from session state
                        uploaded_file = st.session_state.uploaded_file
                        if uploaded_file is None:
                            st.error("Uploaded file not found in session state. Please run the Weather Aware analysis first.")
                        else:
                            # Extract data from session state
                            loading_records = st.session_state.weather_aware_results["loading_records"]
                            line_outages = st.session_state.weather_aware_results["line_outages"]
                            shedding_buses = st.session_state.weather_aware_results["shedding_buses"]
                            gdf = st.session_state.weather_aware_results["gdf"]
                            load_df = st.session_state.weather_aware_results["load_df"]
                            line_idx_map = st.session_state.weather_aware_results["line_idx_map"]

                            max_loading_capacity = 100

                            # Function to map loading percentage to color
                            def get_color(pct):
                                if pct is None:
                                    return '#FF0000'
                                elif pct == 0:
                                    return '#000000'
                                elif pct <= 0.75 * max_loading_capacity:
                                    return '#00FF00'
                                elif pct <= 0.9 * max_loading_capacity:
                                    return '#FFFF00'
                                elif pct < max_loading_capacity:
                                    return '#FFA500'
                                else:
                                    return '#FF0000'

                            # Style function for GEE map
                            def style_fn(feat):
                                props = feat["properties"]
                                if props.get("down_weather"):
                                    return {"color": "#000000", "weight": 5}
                                return {"color": get_color(props.get("loading", 0.0)), "weight": 3}

                            # Display only the selected hour's map
                            hr = hour_to_display
                            loading = loading_records[hr]

                            # Compute weather-down line indices for this hour
                            weather_down_set = set()
                            for (fbus, tbus, start_hr) in line_outages:
                                if hr >= start_hr:
                                    idx = line_idx_map.get((fbus, tbus))
                                    if idx is not None:
                                        weather_down_set.add(idx)

                            # Prepare GeoDataFrame for this hour
                            gdf_hour = gdf.copy()
                            gdf_hour["idx"] = gdf_hour.index
                            gdf_hour["loading"] = gdf_hour["idx"].map(lambda i: loading.get(i, 0.0))
                            gdf_hour["down_weather"] = gdf_hour["idx"].isin(weather_down_set)
                            geojson = gdf_hour.__geo_interface__

                            # Create a Folium map
                            m = folium.Map(location=[27.0, 66.5], zoom_start=7, width=800, height=600)

                            # Add GeoJSON layer
                            folium.GeoJson(
                                geojson,
                                name=f'Weather-Aware: Hour {hr}',
                                style_function=lambda feat: style_fn(feat)
                            ).add_to(m)

                            # Add load bus circles using folium.Circle
                            df = pd.read_excel(uploaded_file, sheet_name='Load Parameters')
                            load_buses = df['bus'].dropna().values
                            shedding_at_hr = [bus for (shed_hr, bus) in shedding_buses if shed_hr == hr]

                            for bus in load_buses:
                                coord_str = load_df.loc[load_df['bus'] == bus, 'load_coordinates'].values[0]
                                lat, lon = eval(coord_str)
                                color = "red" if bus in shedding_at_hr else "green"
                                folium.Circle(
                                    location=(lat, lon),
                                    radius=20000,
                                    color=color,
                                    fill=True,
                                    fill_color=color,
                                    fill_opacity=0.5
                                ).add_to(m)

                            # Add legend inside the map
                            legend_html = """
                            <div style="position: absolute; bottom: 50px; right: 10px; z-index: 1000; background-color: rgba(255, 255, 255, 0.9); padding: 10px; border: 2px solid black; font-size: 14px; color: black;">
                                <b>Line Loading Information (%)</b><br>
                                <i style="background: #00FF00; width: 18px; height: 18px; display: inline-block;"></i> Below 75%<br>
                                <i style="background: #FFFF00; width: 18px; height: 18px; display: inline-block;"></i> 75-90%<br>
                                <i style="background: #FFA500; width: 18px; height: 18px; display: inline-block;"></i> 90-100%<br>
                                <i style="background: #FF0000; width: 18px; height: 18px; display: inline-block;"></i> Line Down (Above 100%)<br>
                                <i style="background: #000000; width: 18px; height: 18px; display: inline-block;"></i> Line Down Due to Weather<br>
                                <i style="background: #FF0000; width: 18px; height: 18px; display: inline-block;"></i> Load Shed Area<br>
                                <i style="background: #008000; width: 18px; height: 18px; display: inline-block;"></i> Load Served Area
                            </div>
                            """
                            m.get_root().html.add_child(folium.Element(legend_html))

                            # Add hour label in the top-left corner
                            hour_label = f'<div style="font-size: 16px; font-weight: bold; color: black; background-color: rgba(255, 255, 255, 0.8); padding: 5px; border-radius: 3px;">Weather-Aware System: Hour {hr}</div>'
                            folium.Marker(
                                location=[29.5, 64.5],
                                icon=folium.DivIcon(html=hour_label)
                            ).add_to(m)

                            # Display the map
                            st.write(f"### Transmission Network at Hour {hr}")
                            st_folium(m, width=800, height=600, key=f"weather_aware_map_{hr}")

                    except Exception as e:
                        st.error(f"Error generating GEE maps: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
