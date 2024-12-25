#Polynomial Regression Project

Overview

This project implements polynomial regression to model the relationship between house features and their prices. The dataset consists of attributes like House_Size, Number_of_Bedrooms, and House_Age as features, and Price as the target variable. Various regression techniques, including Linear Regression, Ridge Regression, Lasso Regression, and ElasticNet Regression, are applied to analyze and improve model performance.

Dataset

The dataset contains the following columns:

    House_Size: Size of the house in square feet.

    Number_of_Bedrooms: Number of bedrooms in the house.

    House_Age: Age of the house in years.

    Price: Selling price of the house.

  Summary of the Dataset

   Number of samples: 150

#Feature Summary:

House_Size: Mean = 2316.3, Std = 995.14

Number_of_Bedrooms: Mean = 2.85, Std = 1.39

House_Age: Mean = 27.19, Std = 13.56

Price: Mean = 391418.33, Std = 155541.97

##Methodology

##1. Data Preprocessing

The dataset is loaded using Pandas.

Feature and target variables are defined:

Features: House_Size, Number_of_Bedrooms, House_Age

Target: Price

Scatter plots are created to visualize the relationship between features and the target variable.

##2. Polynomial Transformation

Features are transformed using PolynomialFeatures with degree 2.

This captures non-linear relationships between features and the target variable.

##3. Modeling

Linear Regression

A basic polynomial regression model is fitted.

Performance Metrics:

Train MSE: 

Test MSE: 

Cross-validation MSE: 

Ridge Regression

Used to address overfitting by adding an L2 penalty.

Results:

Test MSE (default): 656552.23

Scaled Test MSE: 26494053.51

Optimal alpha (): Test MSE (Degree 1): 

Lasso Regression

Adds an L1 penalty to encourage feature selection.

##Performance:

Test MSE: 4370.20

ElasticNet Regression

Combines L1 and L2 penalties to balance feature selection and regularization.

##Performance:

Test MSE: 140222004.70

##4. Coefficient Analysis

Ridge regression coefficients:

House_Size: 150.00

Number_of_Bedrooms: 24999.99

House_Age: -1000.00

These coefficients indicate:

Larger houses and more bedrooms increase price.

Older houses decrease price.

Key Observations

Linear Regression:

Extremely low MSE indicates potential overfitting.

Sensitive to noise in the data.

Ridge Regression:

Balances bias and variance.

Performs better with appropriately tuned alpha.

Lasso Regression:

Effective feature selection with reasonable performance.

ElasticNet Regression:

Less effective for this dataset, possibly due to suboptimal hyperparameters.

Feature Relationships:

House_Size and Number_of_Bedrooms positively correlate with price.

House_Age negatively correlates with price.

Repository Structure

##project-root/<br>
├── house.csv             # Dataset<br>
├── main.py # Implementation code<br>
├── README.md             # Project documentation<br>

Usage

##Prerequisites

Python 3.8+

##Libraries: numpy, pandas, matplotlib, scikit-learn

##Thankyou 
