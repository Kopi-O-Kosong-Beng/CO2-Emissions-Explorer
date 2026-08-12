import streamlit as st

# Page settings
st.set_page_config(page_title="HASS Reflection", layout="wide")

# Title
st.title("HASS Reflection")

st.markdown(
    """
    Dive into the social and ethical dimensions of CO₂ emissions through the lenses of environmental justice and responsibility.
    """
)

# Section: Environmental Justice & Reflection
st.header("Environmental Justice & Reflection")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Environmental Justice")
    st.markdown(
        """
        > **Environmental justice** is about ensuring that all people, regardless of race, color, national origin, or income, have equal protection from environmental harms and equal access to environmental benefits.
        >
        > Dr. Robert Bullard, widely called the father of environmental justice
        """
    )
with col2:
    st.subheader("Ethical Reflection")
    st.markdown(
        """
        > “The ones who walk away from Omelas... They leave Omelas, they walk ahead into the darkness, and they do not come back.”
        >
        > Ursula K. Le Guin
        >
        > How do we confront the uncomfortable truth that our prosperity may come at others’ expense?
        """
    )

st.markdown("---")

# Section: Who Bears the Burden?
st.header("Who Bears the Burden?")

burden_cols = st.columns(3)
with burden_cols[0]:
    st.subheader("Industrial Communities")
    st.write(
        "Often located near pollution sources, facing higher health risks and environmental degradation."
    )
with burden_cols[1]:
    st.subheader("Coastal Regions")
    st.write(
        "Rising sea levels and extreme weather disproportionately affect vulnerable populations."
    )
with burden_cols[2]:
    st.subheader("Future Generations")
    st.write(
        "The long-term consequences of today’s emissions will be inherited by those who had no voice in creating them."
    )

# Footer note
st.markdown("---")
st.caption("Reflect on the societal dimensions of CO₂ emissions and environmental justice.")
