# pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

# Load dataset
data = fetch_openml(name='boston', version=1, as_frame=True)
df = data.frame

X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Train model
model = LinearRegression()
model.fit(X, y)

st.title("🏠 House Price Prediction")

st.write("Enter house details:")

# Input fields
CRIM = st.number_input("Crime Rate")
RM = st.number_input("Number of Rooms")
LSTAT = st.number_input("Lower Status Population (%)")

# Create input dataframe
input_data = pd.DataFrame({
    'CRIM': [CRIM],
    'RM': [RM],
    'LSTAT': [LSTAT]
})

# Add missing columns with 0
for col in X.columns:
    if col not in input_data:
        input_data[col] = 0

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: {prediction[0]:.2f}")