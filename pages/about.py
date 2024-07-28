import streamlit as st

def show():
    st.title("About This Project")
    st.write("""
    This project uses Generative AI to forecast electricity demand and supply gaps across different regions.
    The model is trained on historical data and makes predictions for future years.
    """)
    st.subheader("Project Goals")
    st.write("""
    - To identify potential electricity supply gaps.
    - To assist in planning and decision-making for energy management.
    """)
