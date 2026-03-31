# ============================================================
# Project: Salary Prediction using Polynomial Regression
# ============================================================

# -------------------------------
# 1. Import Required Libraries
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -------------------------------
# 2. Load Dataset
# -------------------------------
# If using Google Colab, upload file
from google.colab import files
uploaded = files.upload()

# Read dataset
dataset = pd.read_csv('dataset.csv')

# -------------------------------
# 3. Explore Dataset
# -------------------------------
print("Dataset Shape:", dataset.shape)
print("\nFirst 5 Rows:\n", dataset.head())

# -------------------------------
# 4. Split Dataset into Features (X) and Target (Y)
# -------------------------------
# X = Independent variable (Level)
X = dataset.iloc[:, :-1].values

# Y = Dependent variable (Salary)
Y = dataset.iloc[:, -1].values

# -------------------------------
# 5. Train Linear Regression Model
# -------------------------------
modelLR = LinearRegression()
modelLR.fit(X, Y)

# -------------------------------
# 6. Visualize Linear Regression
# -------------------------------
plt.scatter(X, Y, color="red")  # Actual data points
plt.plot(X, modelLR.predict(X), color="blue")  # Predicted line

plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# -------------------------------
# 7. Convert Data to Polynomial Features
# -------------------------------
"""
Polynomial Features Explanation:
- degree = 2 → x, x^2
- degree = 3 → x, x^2, x^3
- degree = 4 → x, x^2, x^3, x^4
"""

modelPR = PolynomialFeatures(degree=4)
xPoly = modelPR.fit_transform(X)

# -------------------------------
# 8. Train Polynomial Regression Model
# -------------------------------
modelPLR = LinearRegression()
modelPLR.fit(xPoly, Y)

# -------------------------------
# 9. Visualize Polynomial Regression
# -------------------------------
plt.scatter(X, Y, color="red")  # Actual data

# Smooth curve visualization
plt.plot(X, modelPLR.predict(modelPR.transform(X)), color="blue")

plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# -------------------------------
# 10. Predict Salary using Polynomial Model
# -------------------------------
x = 5  # Input level

salaryPred = modelPLR.predict(modelPR.transform([[x]]))

print(f"Salary of a person with Level {x} is {salaryPred[0]:.2f}")