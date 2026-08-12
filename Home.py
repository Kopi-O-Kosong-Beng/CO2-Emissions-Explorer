from pathlib import Path

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CO₂ Emissions Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and introduction
st.title("CO₂ Emissions Explorer")
st.markdown(
    """
    Per-capita CO₂ emissions run almost twelve-fold apart across US states, from
    7.8 tonnes per person in Maryland to 92.9 in Wyoming. This app asks whether
    five structural drivers can predict that, and finds that no single national
    model does the job.

    Use the sidebar to move between:

    - **Case Studies**: how emissions moved in Wyoming, North Dakota and Alaska between 1998 and 2023.
    - **Prediction**: forecast per-capita CO₂ from a state-specific regression model.
    - **HASS Reflection**: who carries the burden of those emissions.
    """
)

st.info(
    "Built on a 1,300-row panel: 50 states across 26 years, 1998 to 2023. "
    "Sources are the EIA State Energy Data System, the BEA and the US Census Bureau.",
    icon="ℹ️",
)

# Header image, resolved relative to this file so it loads from any working directory
banner = Path(__file__).resolve().parent / "assets" / "future.png"
if banner.exists():
    st.image(str(banner), width=560)

# Sidebar navigation instructions
st.sidebar.header("Navigation")
st.sidebar.markdown("Pick a page above to begin.")
