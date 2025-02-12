# Polynomial Regression - Salary Prediction

This project implements **Polynomial Regression** using Python and Scikit-Learn to model salary expectations based on position levels. It includes data preprocessing, model training, visualization, and performance evaluation.

## 📂 Dataset
The dataset used is **Position_Salaries.csv**, containing:
- **Position Level (`X`)**: The independent variable.
- **Salary (`y`)**: The dependent variable (target to predict).

## 📌 Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📜 Code Implementation
### 1️⃣ Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

### 2️⃣ Loading Dataset
```python
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Extract independent variable
y = dataset.iloc[:, -1].values    # Extract dependent variable
```

### 3️⃣ Training the Polynomial Regression Model
```python
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
line_reg = LinearRegression()
line_reg.fit(X_poly, y)
```

### 4️⃣ Predictions
```python
y_predict = line_reg.predict(X_poly)
```

### 5️⃣ Visualization
#### 📊 Basic Polynomial Regression Curve
```python
plt.scatter(X, y, color='red')
plt.plot(X, y_predict, color='blue')
plt.title('Salary Expectation')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
#### 📊 Higher Resolution Curve
```python
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, line_reg.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Salary Expectation (Higher Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

### 6️⃣ Model Evaluation Metrics
```python
mse = mean_squared_error(y, y_predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_predict)
r2 = r2_score(y, y_predict)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R² Score: {r2:.4f}')
```

## 📌 Why R² Score is Better?
The **R² score (coefficient of determination)** is a key metric because it measures how well the model explains the variance in the dependent variable (`y`).

1. **Normalized Metric:** Unlike MSE, RMSE, or MAE, which depend on the scale of `y`, the **R² score is unitless** and ranges from **0 to 1** (or negative if the model performs worse than a naive mean-based model).
   
2. **Variance Explanation:**  
   - If **R² = 1**, the model perfectly predicts the data.
   - If **R² = 0**, the model is no better than simply predicting the mean.
   - If **R² < 0**, the model performs worse than a horizontal line (constant mean prediction).

3. **Comparison Across Models:** Since it’s a **relative** measure, it allows you to compare different models, like a linear regression vs. polynomial regression.

## 📌 Why We Don't Use Feature Scaling?
Feature scaling (like **Standardization or Normalization**) is **not necessary** for polynomial regression with **LinearRegression** because:

1. **Linear Regression Models Are Scale-Invariant:**  
   - The polynomial features are just **powers** of the original features (e.g., `X², X³, X⁴`).
   - The linear regression model can handle different feature magnitudes without affecting the optimization.

2. **Interpretability Remains Intact:**  
   - Scaling modifies feature values, making it harder to interpret results.
   - Since polynomial regression is sensitive to actual values, keeping the original scale makes sense.

3. **No Gradient Descent Used:**  
   - Unlike models like Logistic Regression or Neural Networks that rely on gradient descent (which benefits from scaled data), **Scikit-Learn’s `LinearRegression` solves equations analytically** using the **Normal Equation**, where scaling has minimal impact.

### **When is Feature Scaling Needed?**
- If you're using **Gradient Descent-based algorithms** (like **SGDRegressor** or **Neural Networks**).
- If you have **regularization (L1, L2)** like **Ridge or Lasso Regression**, where large feature magnitudes can dominate the penalty.

## 📌 Finding the Optimal Degree Using Saddle Point
Since polynomial regression introduces multiple feature transformations (e.g., `X², X³, X⁴`), we can analyze the optimal degree using **saddle points**. A **saddle point** occurs when the second derivative changes sign, indicating a transition between increasing and decreasing curvature. 

### Why Use Saddle Points?
1. **Degree Selection Based on Variance & Bias Tradeoff**  
   - Lower degrees (e.g., `degree=2`) may underfit.
   - Higher degrees (e.g., `degree=10`) may overfit.
   - The **saddle point** helps identify a degree where adding more complexity doesn’t significantly improve the model.

2. **Helps Avoid Overfitting**  
   - By analyzing the loss function’s curvature, we find the point where increasing the polynomial degree no longer significantly improves model performance.

3. **Mathematical Justification**  
   - By computing the second derivative of the loss function (`d²MSE/d(degree)²`), we locate the saddle point where the marginal benefit of increasing polynomial complexity diminishes.

## 📌 Why is it Called Polynomial Linear Regression?
Polynomial Regression is a **special case of Linear Regression** where we **transform the input features** to a polynomial form. The key reason it is still considered **linear regression** lies in the **linear relationship between the coefficients and the transformed features**, not in the shape of the data.

### 📌 **Key Points:**
1. **Linear in Coefficients:**  
   - A polynomial regression model of degree **d** has the form:  
     \[
     y = \theta_0 + \theta_1 X + \theta_2 X^2 + \theta_3 X^3 + ... + \theta_d X^d
     \]
   - Despite `X², X³, ...` being non-linear transformations, the model remains **linear in parameters** (`θ` values).

2. **Same Optimization as Linear Regression:**  
   - The model is solved using **Ordinary Least Squares (OLS)**, the same way as standard **Linear Regression**.
   - No need for iterative approaches like **Gradient Descent** unless working with **large datasets**.

3. **Non-Linear Data Fit, but a Linear Model Structure:**  
   - The polynomial regression equation allows for modeling **non-linear patterns**, but the model is still mathematically considered **linear** in terms of parameters.

