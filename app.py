import streamlit as st
from pages import home, predict, about

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Predict", "About"])

if page == "Home":
    home.show()
elif page == "Predict":
    predict.show()
elif page == "About":
    about.show()
