import streamlit as st
import pandas as pd
import pandapower as pp
import geemap
import ee
import ast
import folium
from folium import plugins
from streamlit_folium import st_folium
import os
import json

# Load GEE service account credentials from Streamlit secrets
credentials = ee.ServiceAccountCredentials(
    st.secrets["GEE_SERVICE_ACCOUNT"],
    key_data=st.secrets["GEE_PRIVATE_KEY"]
)
ee.Initialize(credentials)

# Streamlit app layout
st.title("IEEE 9-Bus Transmission Network Study")
st.write("Upload your network parameters Excel file.")

# Initialize session state for maps and simulation flag
if 'maps' not in st.session_state:
    st.session_state.maps = []
if 'simulate_done' not in st.session_state:
    st.session_state.simulate_done = False

# File uploader
uploaded_file = st.file_uploader("Upload IEEE 9-Bus Excel File", type="xlsx")

# Button to display parameters
if uploaded_file:
    df_bus = pd.read_excel(uploaded_file, sheet_name="Bus Parameters", index_col=0)
    df_load = pd.read_excel(uploaded_file, sheet_name="Load Parameters", index_col=0)
    df_slack = pd.read_excel(uploaded_file, sheet_name="Generator Parameters", index_col=0)
    df_line = pd.read_excel(uploaded_file, sheet_name="Line Parameters", index_col=0)

    # Create the network
    net = pp.create_empty_network()

    # Create buses, loads, generators, and lines
    for idx, row in df_bus.iterrows():
        pp.create_bus(net, name=row["name"], vn_kv=row["vn_kv"], zone=row["zone"], in_service=row["in_service"],
                      max_vm_pu=row["max_vm_pu"], min_vm_pu=row["min_vm_pu"])

    for idx, row in df_load.iterrows():
        pp.create_load(net, bus=row["bus"], p_mw=row["p_mw"], q_mvar=row["q_mvar"], in_service=row["in_service"])

    for idx, row in df_slack.iterrows():
        if row["slack_weight"] == 1:
            pp.create_ext_grid(net, bus=row["bus"], vm_pu=row["vm_pu"], va_degree=0)
        else:
            pp.create_gen(net, bus=row["bus"], p_mw=row["p_mw"], vm_pu=row["vm_pu"], min_q_mvar=row["min_q_mvar"],
                          max_q_mvar=row["max_q_mvar"], scaling=row["scaling"], in_service=row["in_service"],
                          slack_weight=row["slack_weight"], controllable=row["controllable"],
                          max_p_mw=row["max_p_mw"], min_p_mw=row["min_p_mw"])

    for idx, row in df_line.iterrows():
        geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
        pp.create_line_from_parameters(net, from_bus=row["from_bus"], to_bus=row["to_bus"],
                                       length_km=row["length_km"], r_ohm_per_km=row["r_ohm_per_km"],
                                       x_ohm_per_km=row["x_ohm_per_km"], c_nf_per_km=row["c_nf_per_km"],
                                       max_i_ka=row["max_i_ka"], in_service=row["in_service"],
                                       max_loading_percent=row["max_loading_percent"], geodata=geodata)

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
            elif isinstance(ee_object, ee.imagecollection.ImageCollection):
                ee_object_new = ee_object.mosaic()
                map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                folium.raster_layers.TileLayer(
                    tiles=map_id_dict['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    name=name,
                    overlay=True,
                    control=True
                ).add_to(self)
            elif isinstance(ee_object, ee.geometry.Geometry):
                folium.GeoJson(
                    data=ee_object.getInfo(),
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
            st.error(f"Could not display {name}: {e}")

    folium.Map.add_ee_layer = add_ee_layer

    # Button for simulation
    st.write("### Run Simulation")
    simulate = st.button("Run Simulation")

    if simulate or st.session_state.simulate_done:
        st.session_state.maps = []
        st.session_state.simulate_done = True

        load_4 = [33, 33, 33, 33, 33, 33, 50, 58, 74, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 58, 58, 58, 50, 41]
        load_6 = [67, 60, 53, 53, 53, 53, 67, 80, 87, 87, 93, 93, 93, 100, 100, 100, 100, 87, 87, 74, 67, 67, 67, 67]
        load_8 = [17, 17, 17, 17, 17, 21, 34, 67, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 84, 67, 34]

        for hour in range(24):
            net.load.loc[0, 'p_mw'] = load_4[hour]
            net.load.loc[1, 'p_mw'] = load_6[hour]
            net.load.loc[2, 'p_mw'] = load_8[hour]

            if hour < 13:
                net.line.loc[(df_line['from_bus'] == 7) & (df_line['to_bus'] == 8), 'in_service'] = True
            else:
                net.line.loc[(df_line['from_bus'] == 7) & (df_line['to_bus'] == 8), 'in_service'] = False

            try:
                pp.runopp(net)
            except Exception as e:
                st.write(f"Error running power flow analysis at hour {hour + 1}: {e}")
                continue

            Map = folium.Map(location=[27.0, 66.5], zoom_start=7, height=500)

            features = []
            for idx, row in df_line.iterrows():
                try:
                    geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
                    corrected_coords = [(lon, lat) for lat, lon in geodata]
                    geoline = ee.Geometry.LineString(corrected_coords)
                    color = "#00FF00"
                    if not net.line.loc[idx, "in_service"]:
                        color = "#FF0000"
                    else:
                        loading = net.res_line.loc[idx, "loading_percent"]
                        if loading > 95:
                            color = "#FFFF00"
                        elif 70 <= loading <= 95:
                            color = "#0000FF"

                    feature = ee.Feature(geoline, {
                        "from_bus": row["from_bus"],
                        "to_bus": row["to_bus"],
                        "loading_percent": loading,
                        "color": color,
                        "style": {"color": color, "width": 3}
                    })
                    features.append(feature)
                except Exception as e:
                    st.write(f"Error processing line {row['from_bus']} to {row['to_bus']}: {e}")

            line_layer = ee.FeatureCollection(features)
            Map.add_ee_layer(line_layer.style(**{"styleProperty": "style"}), {}, f"Transmission Lines - Hour {hour + 1}")

            for idx, row in df_line.iterrows():
                try:
                    geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
                except Exception as e:
                    st.write(f"Error parsing geodata for line {row['from_bus']} to {row['to_bus']}: {e}")
                    geodata = None

                if geodata:
                    midpoint = [
                        (geodata[0][1] + geodata[-1][1]) / 2,
                        (geodata[0][0] + geodata[-1][0]) / 2
                    ]

                    from_bus = row["from_bus"]
                    to_bus = row["to_bus"]
                    popup_info = f"<b>Line between Bus {from_bus} and Bus {to_bus}</b><br><br>"

                    point = ee.Geometry.Point(midpoint)
                    feature = ee.Feature(point, {
                        "popup": popup_info,
                        "style": {"color": "blue", "pointSize": 7}
                    })

                    Map.add_ee_layer(ee.FeatureCollection([feature]), {}, f"Midpoint Marker - Line {from_bus} to {to_bus} - Hour {hour + 1}")

            legend_dict = {
                "Line Down": "#FF0000",
                "Over 95% Loading": "#FFFF00",
                "70%-95% Loading": "#0000FF",
                "Below 70% Loading": "#00FF00"
            }
            Map.add_child(folium.LayerControl())

            st.session_state.maps.append(Map)

        for i, Map in enumerate(st.session_state.maps):
            st.write(f"### Hour {i + 1}")
            st_folium(Map, width=725, key=f"map_{i}")
