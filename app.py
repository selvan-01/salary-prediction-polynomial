# ============================================================
# Salary Prediction using Polynomial Regression - Streamlit App
# ============================================================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -------------------------------
# 2. App Title
# -------------------------------
st.title("💼 Salary Prediction using Polynomial Regression")
st.write("Predict salary based on experience level using Machine Learning")

# -------------------------------
# 3. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

dataset = load_data()

# -------------------------------
# 4. Show Dataset
# -------------------------------
st.subheader("📂 Dataset Preview")
st.write(dataset.head())

# -------------------------------
# 5. Prepare Data
# -------------------------------
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# -------------------------------
# 6. Train Models
# -------------------------------
# Linear Regression
modelLR = LinearRegression()
modelLR.fit(X, Y)

# Polynomial Regression
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

modelPLR = LinearRegression()
modelPLR.fit(X_poly, Y)

# -------------------------------
# 7. User Input
# -------------------------------
st.subheader("🎯 Enter Experience Level")
level = st.slider("Select Level", min_value=1, max_value=10, value=5)

# -------------------------------
# 8. Prediction
# -------------------------------
prediction = modelPLR.predict(poly.transform([[level]]))

st.subheader("💰 Predicted Salary")
st.success(f"Salary for Level {level} is ₹ {prediction[0]:,.2f}")

# -------------------------------
# 9. Visualization
# -------------------------------
st.subheader("📈 Model Visualization")

fig, ax = plt.subplots()

# Scatter plot (actual data)
ax.scatter(X, Y)

# Smooth curve for polynomial regression
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
ax.plot(X_grid, modelPLR.predict(poly.transform(X_grid)))

ax.set_title("Polynomial Regression Curve")
ax.set_xlabel("Level")
ax.set_ylabel("Salary")

st.pyplot(fig)

# -------------------------------
# 10. Footer
# -------------------------------
st.write("🚀 Built with Streamlit by Sen")