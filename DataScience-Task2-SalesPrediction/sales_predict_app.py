import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('sales_model.pkl')

st.title("📈 Sales Prediction App")
st.write("Enter advertising budget (in thousands of dollars) to predict expected Sales (in thousands of units).")

# Input fields
tv = st.number_input("TV Advertising Budget (in $1000s)", min_value=0.0, max_value=500.0, value=150.0)
radio = st.number_input("Radio Advertising Budget (in $1000s)", min_value=0.0, max_value=100.0, value=25.0)
newspaper = st.number_input("Newspaper Advertising Budget (in $1000s)", min_value=0.0, max_value=150.0, value=30.0)

if st.button("Predict Sales"):
    features = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]

    st.success(f"Predicted Sales: **{prediction:.2f} thousand units** (~{prediction*1000:.0f} units)")