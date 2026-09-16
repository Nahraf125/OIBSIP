import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('iris_model.pkl')

# Mapping numbers back to species names
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements below to predict its species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_species = species_map[prediction]
    st.success(f"Predicted Species: **{predicted_species}**")

    st.write("### Class Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"- {species_map[i]}: {prob*100:.2f}%")