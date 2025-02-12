# Polynomial Regression - Salary Prediction

This project implements **Polynomial Regression** using Python and Scikit-Learn to model salary expectations based on position levels. It includes data preprocessing, model training, visualization, and performance evaluation.

## 📂 Dataset
The dataset used is **Position_Salaries.csv**, containing:
- **Position Level (`X`)**: The independent variable.
- **Salary (`y`)**: The dependent variable (target to predict).

## 📌 Overview of Code
The implementation follows these key steps:

1. **Importing Libraries**  
   - Essential libraries such as `pandas`, `numpy`, and `matplotlib` for data handling and visualization.
   - `sklearn.preprocessing` and `sklearn.linear_model` for feature transformation and model training.

2. **Loading the Dataset**  
   - Reads the dataset into a DataFrame.
   - Extracts the relevant features (`X`) and target variable (`y`).

3. **Training the Polynomial Regression Model**  
   - Converts the input `X` into polynomial features using `PolynomialFeatures(degree=4)`.
   - Fits a `LinearRegression` model to these transformed features.
   - Predicts the salaries based on the trained model.

4. **Visualizing the Results**  
   - Creates a scatter plot of the actual data points.
   - Plots the polynomial regression curve to visualize the model fit.
   - Uses a finer resolution (`X_grid`) for a smoother curve.

5. **Making Predictions**  
   - Uses the trained model to predict salaries for given position levels.
   - Evaluates the model using error metrics.

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

## 📊 Metric Score  
The model's performance is evaluated using a metric score, showcasing its accuracy and effectiveness. 📈✅ 

![Metric Score](https://github.com/tanveerj5/Polynomial-Regression---Salary-Prediction/blob/main/score.png)  

## ❌ Why the Linear Model is Not Good for This Dataset  
The linear model fails to capture the complex relationship in the data, leading to poor predictions. 🚫📉  

![Linear Model Issue](https://github.com/tanveerj5/Polynomial-Regression---Salary-Prediction/blob/main/linear%20model.png)  

## 🔍 Test Visualization  
A test visualization demonstrates how the model predicts values, highlighting its generalization ability. 📊🔎  

![Test Visualization](https://github.com/tanveerj5/Polynomial-Regression---Salary-Prediction/blob/main/test%20resolution%20result.png)  

## 🎯 Smooth Visualization  
A high-resolution curve provides a clearer and more refined representation of the model’s predictions. 📈✨  

![Smooth Visualization](https://github.com/tanveerj5/Polynomial-Regression---Salary-Prediction/blob/main/high%20resolution%20curve.png)  
