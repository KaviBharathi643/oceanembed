"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Streamlit Companion & Integration Dashboard

Provides a Streamlit interface that utilizes the same verified OceanEmbedEngine
and checkpoints/oceanembed_best.pt, or directs judges to the custom modern web prototype.
"""

import os
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.engine import OceanEmbedEngine
from src.data.dataset import TARGET_DEPTHS

st.set_page_config(
    page_title="OceanEmbed — Subsurface Temperature Reconstruction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Light Marine CSS styling
st.markdown("""
<style>
    .main { background-color: #F4F8FA; }
    .stMetric { background-color: #FFFFFF; padding: 12px; border-radius: 12px; border: 1px solid #E2E8F0; }
    .css-1d391kg { background-color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 OceanEmbed: Satellite Embedding-Based Reconstruction")
st.caption("SIH 2026 | Problem Statement 26066 | Ministry of Earth Sciences (MoES / INCOIS)")

# Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Select Section",
    ["Working Prototype (Full View)", "Predict Profile", "Data Sources", "Validation (ARGO)", "About Project"],
)

engine = OceanEmbedEngine.get_instance()

if menu == "Working Prototype (Full View)":
    st.info("💡 For the full judge-facing experience with interactive map and 5x5 neighborhood viewer, run: `python run_dashboard.py` to open the web application on port 8000.")
    st.markdown("""
    **Primary Showcase Preset**:
    - **Location**: Central Bay of Bengal (15.25°N, 87.50°E)
    - **Date**: 2024-11-15
    - **Model**: `checkpoints/oceanembed_best.pt` (120,655 params)
    - **Validation Status**: Verified (ARGO RMSE: 0.8464 °C)
    """)

elif menu == "Predict Profile":
    st.subheader("Subsurface Profile Reconstruction")
    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input("Latitude (°N)", min_value=5.0, max_value=25.0, value=15.25, step=0.25)
    with col2:
        lon = st.number_input("Longitude (°E)", min_value=80.0, max_value=100.0, value=87.50, step=0.25)
    with col3:
        date = st.selectbox("Date", ["2024-11-15", "2024-05-15", "2024-07-15", "2024-03-15", "2024-09-15"])

    if st.button("Run Model Inference"):
        res = engine.predict(lat, lon, date)
        if res["success"]:
            st.success(f"Inference completed in {res['model_metadata']['inference_time_ms']} ms")
            
            # Display Surface Conditions
            st.subheader("Extracted 7 Surface Variables (5x5 Center)")
            cols = st.columns(7)
            sc = res["surface_conditions"]
            cols[0].metric("SST", sc["sst"]["display"])
            cols[1].metric("SSS", sc["sss"]["display"])
            cols[2].metric("SLA", sc["sla"]["display"])
            cols[3].metric("Curr U", sc["current_u"]["display"])
            cols[4].metric("Curr V", sc["current_v"]["display"])
            cols[5].metric("Wind U", sc["wind_u"]["display"])
            cols[6].metric("Wind V", sc["wind_v"]["display"])

            # Display Profile Table and Metrics
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.write("**Predicted Profile Table**")
                import pandas as pd
                df_prof = pd.DataFrame(res["profile"])
                st.dataframe(df_prof, use_container_width=True)
                st.caption("💡 Scientific Reference: GLORYS is used as the reconstruction training/reference dataset. Argo observations provide independent validation.")
            with c_right:
                st.write("**Physical Diagnostics**")
                diag = res["diagnostics"]
                st.write(f"- **Mixed Layer Depth (MLD)**: `{diag['mixed_layer_depth_m']} m`")
                st.write(f"- **Thermocline Core Depth**: `{diag['thermocline_core_depth_m']} m`")
                st.write(f"- **Max Gradient**: `{diag['max_vertical_gradient_c_per_m']} °C/m`")
                st.write(f"- **Deep Water (1000m)**: `{diag['deep_temperature_1000m_c']} °C`")
        else:
            st.error(res["error"])

elif menu == "Data Sources":
    st.subheader("Acquired Oceanographic Data Catalog")
    cat = engine.get_data_catalog()
    import pandas as pd
    st.dataframe(pd.DataFrame(cat), use_container_width=True)

elif menu == "Validation (ARGO)":
    st.subheader("Independent Observational Ground-Truth Validation")
    st.markdown("""
    - **Matched Profiles**: `672`
    - **Depth Comparisons**: `9,983`
    - **Overall ARGO RMSE**: `0.8464 °C`
    - **Improvement over Fair Climatology**: `+36.28%`
    - **Global Bias**: `-0.0065 °C`
    """)

elif menu == "About Project":
    st.subheader("About OceanEmbed")
    st.markdown("""
    **OceanEmbed** bridges the gap between satellite surface observation and sparse deep-ocean measurements.
    Trained on dense GLORYS12V1 reanalysis and validated against independent in-situ ARGO profiling floats.
    """)
