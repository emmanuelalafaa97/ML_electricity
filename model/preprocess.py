import pandas as pd
import joblib
import numpy as np

def load_scaler():
    # Load the saved scaler
    scaler = joblib.load('model/scaler.pkl')
    return scaler

def preprocess_data(region, year, scaler, features):
    """
    Preprocesses the input data by creating the required features,
    encoding the categorical variable 'region', and scaling the features.

    Args:
        region (str): The region (e.g., 'Sub-Saharan Africa', 'Other Regions').
        year (int): The year for prediction.

    Returns:
        np.ndarray: The preprocessed data ready for prediction.
    """
    # Create a DataFrame for the input data
    data = pd.DataFrame({'region': [region], 'year': [year]})

    # Dummy values for other features; replace with actual logic to obtain these values
    dummy_values = {
        'el_access_urban': 70, 'el_demand': 5000, 'el_access_rural': 50,
        'population': 1000000, 'net_imports': 100, 'el_demand_pc': 500,
        'fin_support': 1000, 'el_from_gas': 100, 'pop_no_el_access_total': 100000,
        'urban_share': 50, 'income_group_num': 2, 'el_access_total': 60, 'gdp_pc': 2000
    }

    for key, value in dummy_values.items():
        data[key] = value

    # Ensure the columns are in the correct order as expected by the model
    features = ['el_access_urban', 'el_demand', 'el_access_rural', 'population', 'net_imports', 'el_demand_pc', 
                'fin_support', 'el_from_gas', 'pop_no_el_access_total', 'urban_share', 'income_group_num', 
                'year', 'el_access_total', 'gdp_pc']
    data = data[features]

    # Scale the input data
    input_data = scaler.transform(data)
    return input_data
