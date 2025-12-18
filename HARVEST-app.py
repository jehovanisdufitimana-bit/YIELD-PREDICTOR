# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 21:42:52 2025

@author: G.HOME OF ELECTRONIC
"""

import streamlit as st
import pickle
import numpy as np

# Load the saved model using the registration number [cite: 65, 66]
def load_model():
    filename = '25RP19784.pkl'
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 25RP19784.pkl is in the directory.")
        return None

model = load_model()

# Web App Interface using Streamlit
st.title("YIELD PREDICTOR APP")
st.subheader("Based on Temperature Variations")

st.write("""
This application uses a supervised learning model to forecast crop yields 
based on temperature.
""")

# User Input for Temperature [cite: 20]
temp_input = st.number_input("Enter Average Temperature (in °C):", value=25.0)

if st.button("PREDICT YIELD"):
    if model:
        # Reshape input for prediction [cite: 41, 48]
        input_data = np.array([[temp_input]])
        prediction = model.predict(input_data)
        
        # Display the Result [cite: 19]
        st.success(f"The predicted crop yield for {temp_input}°C is: {prediction[0]:.2f} units")
    else:
        st.warning("Model is not loaded.")

# Add context about the model's reliability [cite: 21, 22]
st.info("Note: This model was trained on a dataset using an 80/20 split and evaluated using MSE and R² scores.")