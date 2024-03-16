import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn

# Load the model and preprocessor
model = pickle.load(open('final_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessing.pkl', 'rb'))

st.title('Wine Quality Prediction')

# Define input sliders for features
alcohol = st.slider('Alcohol', 8, 15, 10)
fixed_acidity = st.slider('Fixed Acidity', 4, 16, 6)
freeSO2 = st.slider('Free sulfur dioxide', 2, 290, 50)
volatile_acidity = st.slider('Volatile acidity', 0.05, 1.2, 0.3, step=0.05)
pH = st.slider('pH value', 2.7, 3.9, 3.0, step=0.1)
residual_sugar = st.slider('Residual sugar', 1, 33, 8)
totalSO2 = st.slider('Total sulfur dioxide', 9, 440, 50)
sulphates = st.slider('Sulphates', 0.2, 1.1, 0.3, step=0.1)
citric_acid = st.slider('Citric acid', 0.0, 1.7, 0.2, step=0.1)

# Prepare the input features as a list of numeric values
input_features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                  0, freeSO2, totalSO2, 0,
                  pH, sulphates, alcohol]  # Assuming 'chlorides' and 'density' are not input features

# Make prediction when the user clicks the 'Predict' button
if st.button('Predict white wine quality'):
    # Transform the input features using the preprocessor
    transformed_features = preprocessor.transform([input_features])

    # Predict the wine quality using the trained model
    predicted_quality = model.predict(transformed_features).flatten()[0]

    st.write('The predicted wine quality is:', predicted_quality,style={"font-size": "20px"})
