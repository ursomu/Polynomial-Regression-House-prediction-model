# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

# Loading the dataset
data = pd.read_csv('house.csv')

# Defining features and target
x = data[['House_Size', 'Number_of_Bedrooms', 'House_Age']]
y = data['Price']

# Scatter plots for feature analysis
plt.scatter(data['House_Size'], data['Price'], color="blue", label="House_Size vs Price")
plt.scatter(data['Number_of_Bedrooms'], data['Price'], color="black", label="Number_of_Bedrooms vs Price")
plt.scatter(data['House_Age'], data['Price'], color="grey", label="House_Age vs Price")
plt.title('Scatter Plots')
plt.legend()
plt.show()

# Polynomial transformation of degree 2
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Predictions and MSE for Linear Regression
y_pred_train = linear_model.predict(x_train)
y_pred_test = linear_model.predict(x_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"MSE of Train: {mse_train}")
print(f"MSE of Test: {mse_test}")

# Cross-validation
cross_val_mse = -np.mean(cross_val_score(linear_model, x_poly, y, scoring='neg_mean_squared_error', cv=5))
print(f"Cross-validation MSE: {cross_val_mse}")

# Adding noise to the target
y_noisy = y + np.random.normal(0, 0.1, y.shape)
mse_with_noise = mean_squared_error(y, y_noisy)
print(f"MSE with Noise: {mse_with_noise}")

# Ridge Regression with default alpha
ridge_model = Ridge()
ridge_model.fit(x_train, y_train)
ridge_test_mse = mean_squared_error(y_test, ridge_model.predict(x_test))
print(f"Ridge Test MSE: {ridge_test_mse}")

# Ridge Regression with scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
ridge_model_scaled = Ridge()
ridge_model_scaled.fit(x_train_scaled, y_train)
ridge_test_mse_scaled = mean_squared_error(y_test, ridge_model_scaled.predict(x_test_scaled))
print(f"Ridge Test MSE with Scaling: {ridge_test_mse_scaled}")

# Ridge Regression for multiple alpha values
alphas = [0.1, 1, 10, 100, 1000]
for alpha in alphas:
    ridge_model_alpha = Ridge(alpha=alpha)
    ridge_model_alpha.fit(x_train, y_train)
    ridge_test_mse_alpha = mean_squared_error(y_test, ridge_model_alpha.predict(x_test))
    print(f"Alpha: {alpha}, Ridge Test MSE: {ridge_test_mse_alpha}")

# Ridge Regression (Degree 1)
poly_degree1 = PolynomialFeatures(degree=1)
x_poly_degree1 = poly_degree1.fit_transform(x)
x_train_deg1, x_test_deg1, y_train_deg1, y_test_deg1 = train_test_split(x_poly_degree1, y, test_size=0.2, random_state=42)
ridge_deg1_model = Ridge(alpha=0.0001)
ridge_deg1_model.fit(x_train_deg1, y_train_deg1)
ridge_test_mse_deg1 = mean_squared_error(y_test_deg1, ridge_deg1_model.predict(x_test_deg1))
print(f"Ridge Test MSE (Degree 1): {ridge_test_mse_deg1}")

# Lasso Regression
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(x_train, y_train)
lasso_test_mse = mean_squared_error(y_test, lasso_model.predict(x_test))
print(f"Lasso Test MSE: {lasso_test_mse}")

# Ridge coefficients
print("Ridge Coefficients:", ridge_model.coef_)

# ElasticNet Regression
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_model.fit(x_train, y_train)
elastic_test_mse = mean_squared_error(y_test, elastic_model.predict(x_test))
print(f"ElasticNet Test MSE: {elastic_test_mse}")

# Dataset summary
print(data.describe())

plt.scatter(y_test, linear_model.predict(x_test), label="Without Noise", color="blue", alpha=0.7)
plt.scatter(y_test, linear_model.predict(x_test), label="With Noise", color="red", alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color="black", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("Model Performance With and Without Noise")
plt.show()

