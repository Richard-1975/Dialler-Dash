"""Entrypoint."""
import streamlit as st

st.set_page_config(
    page_title="Insurance Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/insurance.py", title="Insurance Operations", icon="🛡️"),
])
pg.run()
