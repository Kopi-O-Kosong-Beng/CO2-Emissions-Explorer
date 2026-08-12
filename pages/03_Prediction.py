"""
Prediction page: forecast CO₂ per capita from a state-specific linear model.
"""

import pandas as pd
import streamlit as st

from data_service import (
    feature_bounds,
    fit_state_models,
    latest_observation,
    predict_co2,
)

# Configure the page BEFORE any other Streamlit calls
st.set_page_config(page_title="Prediction", layout="wide")

STATE_NAMES = {"WY": "Wyoming", "ND": "North Dakota", "AK": "Alaska"}

# Label, unit and step for each predictor, in model order.
FEATURE_UI = {
    "renewable energy": ("Renewable energy consumption", "billion Btu per year", 100.0),
    "Coal Electricity Consumption": ("Coal consumed for electricity", "short tons per year", 10_000.0),
    "Natural Gas Electricity Consumption": (
        "Natural gas consumed for electricity",
        "thousand cubic feet per year",
        10_000.0,
    ),
    "PCE per capita": ("Personal consumption expenditure", "USD per person per year", 100.0),
    "Estimated Urban Population": ("Urban population", "people", 100.0),
}

# ─────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────
st.title("CO₂ Emission Prediction")
st.markdown(
    """
    Pick a state model and move its drivers to forecast CO₂ emissions per capita.
    Each control is bounded by what that state actually recorded between 1998 and
    2023, because a linear model says nothing reliable outside the range it was
    fitted on.
    """
)

# ─────────────────────────────────────────────────────────────────
# Sidebar: model selector and inputs
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("Model and inputs")

state = st.sidebar.selectbox(
    "State model",
    ("WY", "ND", "AK"),
    format_func=lambda code: STATE_NAMES[code],
)

bounds = feature_bounds(state)
values: dict[str, float] = {}

for feature, (label, unit, step) in FEATURE_UI.items():
    limits = bounds[feature]
    values[feature] = st.sidebar.slider(
        f"{label} ({unit})",
        min_value=float(limits["min"]),
        max_value=float(limits["max"]),
        value=float(limits["latest"]),
        step=float(step),
        format="%.0f",  # these are counts and dollars, never fractions
        help=(
            f"Observed in {STATE_NAMES[state]} between "
            f"{limits['min']:,.0f} and {limits['max']:,.0f}. "
            "The slider opens at the most recent year on record."
        ),
    )

# ─────────────────────────────────────────────────────────────────
# Prediction, recomputed on every interaction
# ─────────────────────────────────────────────────────────────────
prediction = predict_co2(
    state_code=state,
    renewable_energy=values["renewable energy"],
    coal_elec=values["Coal Electricity Consumption"],
    gas_elec=values["Natural Gas Electricity Consumption"],
    pce_per_capita=values["PCE per capita"],
    urban_pop=values["Estimated Urban Population"],
)

latest_year, latest_actual = latest_observation(state)

left, right = st.columns([1, 1])

with left:
    st.metric(
        label=f"Predicted CO₂ per capita, {STATE_NAMES[state]}",
        value=f"{prediction:.2f} t",
        delta=f"{prediction - latest_actual:+.2f} t vs {latest_year} actual",
        delta_color="off",
    )
    st.caption(
        f"{STATE_NAMES[state]} recorded {latest_actual:.1f} t per person in {latest_year}."
    )

with right:
    coefficients = fit_state_models()[state]
    st.markdown("**How the forecast is built**")
    # Columns are pre-formatted as text so the table renders identically on
    # every Streamlit version, without depending on Styler support.
    contributions = pd.DataFrame(
        {
            "Driver": [FEATURE_UI[f][0] for f in FEATURE_UI],
            "Your input": [f"{values[f]:,.0f}" for f in FEATURE_UI],
            "Contribution (t)": [f"{coefficients[f] * values[f]:+.2f}" for f in FEATURE_UI],
        }
    )
    st.table(contributions)
    st.caption(
        f"Baseline intercept {coefficients['intercept']:.2f} t. "
        "The contributions above sum with it to the predicted value."
    )

# ─────────────────────────────────────────────────────────────────
# Transparency: exactly what went into the model
# ─────────────────────────────────────────────────────────────────
with st.expander("Show the raw inputs and the range each one was fitted on"):
    st.table(
        pd.DataFrame(
            {
                "Driver": [FEATURE_UI[f][0] for f in FEATURE_UI],
                "Unit": [FEATURE_UI[f][1] for f in FEATURE_UI],
                "Value": [f"{values[f]:,.0f}" for f in FEATURE_UI],
                "Observed minimum": [f"{bounds[f]['min']:,.0f}" for f in FEATURE_UI],
                "Observed maximum": [f"{bounds[f]['max']:,.0f}" for f in FEATURE_UI],
            }
        )
    )
