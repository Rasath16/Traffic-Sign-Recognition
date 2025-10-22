"""Upload component for Streamlit."""
import streamlit as st


def file_uploader():
    return st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
