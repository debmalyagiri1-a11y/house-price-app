import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset from CSV
# Make sure you have a file named "boston.csv" in your repo
df = pd.read_csv("boston.csv")

# Select important features
features = ['CRIM', 'RM', 'LSTAT', 'AGE', 'TAX']
X = df[features]
y = df['MEDV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
score = r2_score(y_test, y_pred)

# UI
st.title("🏠 House Price Prediction System")
st.write("Enter house details below:")

# Inputs
CRIM = st.number_input("Crime Rate", 0.0)
RM = st.number_input("Number of Rooms", 0.0)
LSTAT = st.number_input("Lower Status Population (%)", 0.0)
AGE = st.number_input("Age of House", 0.0)
TAX = st.number_input("Property Tax Rate", 0.0)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([[CRIM, RM, LSTAT, AGE, TAX]], columns=features)
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0]:.2f}k")

# Show accuracy
st.write(f"📊 Model Accuracy (R² Score): {score:.2f}")
