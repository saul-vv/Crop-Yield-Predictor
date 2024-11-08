# Run in the terminal with python -m streamlit run 2_streamlit_app.py

import streamlit as st
import pickle
import numpy as np

# Load the model and the scaler
model = pickle.load(open("trained_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Set up the title
st.title("Crop Yield Predictor")

# Input fields for user input
crop = st.selectbox("Crop:", ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"])
crop_barley = 1 if crop == "Barley" else 0
crop_cotton = 1 if crop == "Cotton" else 0
crop_maize = 1 if crop == "Maize" else 0
crop_rice = 1 if crop == "Rice" else 0
crop_soybean = 1 if crop == "Soybean" else 0
crop_wheat = 1 if crop == "Wheat" else 0

int_rainfall = st.number_input("Expected rainfall (mm):", min_value=0, value=0)
int_temp = st.number_input("Average temperature (ºC):", min_value=-20, max_value=60, value=0)
bool_fertilizer = st.checkbox("Fertilizer used?")
bool_irrigation = st.checkbox("Irrigation used?")
int_days = st.number_input("Days to harvest:", min_value=0, max_value=600, value=0)

soil = st.selectbox("Soil type:", ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"])
soil_chalky = 1 if soil == "Chalky" else 0
soil_clay = 1 if soil == "Clay" else 0
soil_loam = 1 if soil == "Loam" else 0
soil_peaty = 1 if soil == "Peaty" else 0
soil_sandy = 1 if soil == "Sandy" else 0
soil_silt = 1 if soil == "Silt" else 0

if st.button("Predict"): # Button to submit the data and get a prediction
    input_data = np.array([[int_rainfall,
                            int_temp,
                            bool_fertilizer,
                            bool_irrigation,
                            int_days,
                            soil_chalky, soil_clay, soil_loam, soil_peaty, soil_sandy, soil_silt,
                            crop_barley, crop_cotton, crop_maize, crop_rice, crop_soybean, crop_wheat
                            ]])
    
    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    st.write("Yield Prediction (T/ha):", prediction[0])
