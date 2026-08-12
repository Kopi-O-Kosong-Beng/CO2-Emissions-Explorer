import streamlit as st

from case_service import get_state_co2_series, load_merged_data

# ─── Page configuration ────────────────────────────────────────────────────────
# Set the browser tab title and choose a wide layout so charts span the width.
st.set_page_config(page_title="Case Studies", layout="wide")

# ─── Page header ───────────────────────────────────────────────────────────────
st.title("Case Studies: Wyoming, North Dakota and Alaska")
st.markdown(
    """
    A single pooled model across all 50 states came out as noise, because
    state-level structure swamps any shared pattern. So we narrowed to the three
    highest per-capita emitters and fitted each one its own regression on the
    same five drivers.
    """
)

# ─── Loop over the three states and render each section ────────────────────────
for code, name in [("WY", "Wyoming"), ("ND", "North Dakota"), ("AK", "Alaska")]:
    # Sub-section header for the state
    st.subheader(f"{name} ({code})")

    # Fetch time series and rename the emission column for clarity
    df_series = get_state_co2_series(code).rename(columns={"co2 per capita": "CO₂"})
    # Plot the historical CO₂ per capita as a line chart
    st.line_chart(df_series.set_index("Year"))

    # Narrative text explaining the drivers behind each state's ranking
    if code == "WY":
        st.markdown(
            """
            **Wyoming** ranks **1st** in CO₂ per capita nationwide. The intensity comes
            from enormous coal reserves feeding coal-fired power plants, among the
            largest generators of coal electricity in the country, spread across a very
            small population base. In 2023 over **80%** of its in-state electricity came
            from coal combustion.
            """
        )
    elif code == "ND":
        st.markdown(
            """
            **North Dakota** places **2nd**, driven mainly by its oil and natural gas
            sector. Hydraulic fracturing and associated gas flaring in the Bakken
            release substantial CO₂, while a small population spreads the state total
            across fewer residents.
            """
        )
    else:  # AK
        st.markdown(
            """
            **Alaska** is **3rd**, reflecting heating demand in an Arctic climate and the
            energy intensity of moving oil and gas offshore. Despite growing renewable
            installations, per-capita use of diesel and natural gas stays elevated in
            rural and urban communities alike.
            """
        )

# ─── Optional raw data preview ─────────────────────────────────────────────────
# Give users the option to inspect the underlying numbers
if st.checkbox("Show the raw data for these three states"):
    df = load_merged_data()  # load full dataset
    df_subset = df[df["State"].isin(["WY", "ND", "AK"])]
    df_display = (
        df_subset[["State", "Year", "co2 per capita"]]
        .sort_values(["State", "Year"])
        .reset_index(drop=True)
    )
    st.dataframe(df_display)  # interactive table

# ─── Data sources ─────────────────────────────────────────────────────────────
st.markdown(
    """
    **Sources**
    1. US Energy Information Administration. State Energy Data System, 2023.
    2. US Environmental Protection Agency. Greenhouse Gas Inventory Data, 2022.
    3. National Oceanic and Atmospheric Administration. Arctic Climatology, 2023.
    """
)
