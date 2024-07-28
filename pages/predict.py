import streamlit as st
import pandas as pd
import numpy as np
from model.preprocess import load_scaler, preprocess_data
import tensorflow as tf

def load_model():
    # Load the TensorFlow model
    model = tf.keras.models.load_model('model/Elec_model.h5')
    return model

def show():
    st.title("Forecast Electricity Demand and Supply Gaps")

    # User inputs for prediction
    region = st.selectbox("Select Region", ["Sub-Saharan Africa", "Other Regions"])
    year = st.slider("Select Year", 2024, 2030)

    # Features used in the model
    features = ['el_access_urban', 'el_demand', 'el_access_rural', 'population', 'net_imports', 'el_demand_pc', 
                'fin_support', 'el_from_gas', 'pop_no_el_access_total', 'urban_share', 'income_group_num', 
                'year', 'el_access_total', 'gdp_pc']

    # Load scaler and model
    scaler = load_scaler()
    model = load_model()

    # Preprocess input data
    input_data = preprocess_data(region, year, scaler, features)

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Electricity Demand and Supply Gap for {region} in {year}: {prediction[0][0]}")

    # Optional: Show some example data
    st.subheader("Sample Data")
    data = pd.read_csv("data/data.csv")
    st.dataframe(data.head())
