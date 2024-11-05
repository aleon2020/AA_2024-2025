# coding: utf-8


import sys
# * from python_environment_check import check_packages
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



# * import sys
# Import the sys module, which is a Python standard library module.
# This module provides access to variables and functions that interact strongly with the
# Python interpreter, such as manipulating module search path and input/output
# standard, among others.
# * sys.path
# It is a list containing the paths in which the Python interpreter looks for modules when
# you use import. When you try to import a module, Python searches the paths specified in this
# list.
# * sys.path.insert(0, '..')
# Insert the path '..' (representing the parent directory) at the beginning of the sys.path list.
# Adding it in position 0 ensures that when Python looks for modules to import,
# first check in the parent directory before continuing with the default paths.

sys.path.insert(0, '..')


# Check recommended package versions:



# Import the check_packages function from the python_environment_check module. 
# This module, from its name, appears to be designed to verify that the Python environment 
# have the correct versions of certain packages installed.
# * d = {...}
# Defines a dictionary d that contains the names of several packages as keys 
# (e.g. numpy, scipy, matplotlib, etc.) and as values ​​the minimum versions 
# required from those packages.
# * check_packages(d)
# The check_packages function takes as input the dictionary d and probably performs a 
# check on current Python environment to ensure installed versions 
# of these packages are at least those specified in the dictionary. If any of the packages 
# is not installed or has the wrong version, the function may throw an error or 
# suggest installing/updating the packages.

d = {
    'numpy': '1.21.2',
    'mlxtend': '0.19.0',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2',
}
check_packages(d)


# # Example 3 - Regression Dataset

# ### Overview

# - [Data Analysis](#data-analysis)
#     - [Importing libraries for data analysis and scaling](#importing-libraries-for-data-analysis-and-scaling)
#     - [Loading the dataset from a CSV file](#loading-the-dataset-from-a-csv-file)
#     - [Anonymization and correlation analysis of the dataset](#anonymization-and-correlation-analysis-of-the-dataset)
#     - [Separation of dataset features and labels](#separation-of-dataset-features-and-labels)
#     - [Visualization of the correlation matrix in a heat map](#visualization-of-the-correlation-matrix-in-a-heat-map)
#     - [Viewing distributions in histograms](#viewing-distributions-in-histograms)
#     - [Division of the dataset into training (70%) and test (30%)](#division-of-the-dataset-into-training-70-and-test-30)
#     - [Calculation of mean squared error](#calculation-of-mean-squared-error)
#     - [Calculation of the mean absolute error](#calculation-of-the-mean-absolute-error)
#     - [Calculation of the coefficient of determination](#calculation-of-the-coefficient-of-determination)
# - [Regression Methods](#regression-methods)
#     - [Training and evaluation of the model by linear regression](#training-and-evaluation-of-the-model-by-linear-regression)
#     - [Training and evaluation of the model by quadratic polynomial regression](#training-and-evaluation-of-the-model-by-quadratic-polynomial-regression)
#     - [Training and evaluation of the model by cubic polynomial regression](#training-and-evaluation-of-the-model-by-cubic-polynomial-regression)
#     - [Training and evaluation of the model by Decision Tree regression](#training-and-evaluation-of-the-model-by-decision-tree-regression)
#     - [Training and evaluation of the model by Random Forest regression](#training-and-evaluation-of-the-model-by-random-forest-regression)
# - [Summary](#summary)



# * from IPython.display
# Import from the display submodule of the IPython package. This module is designed to display 
# and render different types of data within interactive environments, such as Jupyter Notebooks.
# * import Image
# Import the Image class from the display module. The Image class is used to display 
# images in the interactive environment (for example, in a Jupyter Notebook cell).
# * %matplotlib inline
# This is a magic command specific to IPython/Jupyter Notebook.
# Enables display of matplotlib plots directly within cells of the 
# notebook. Graphics are rendered "inline" (within the same notebook) without the need 
# to open pop-up windows.



# # Data Analysis

# ## Importing libraries for data analysis and scaling





# ## Loading the dataset from a CSV file



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6',
           'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Target']

df = pd.read_csv('dataset_para_regresion.csv', 
                 sep=',',
                 usecols=columns)

df.columns

df.shape

df.head(1)

df.info()

df.describe()


# ## Anonymization and correlation analysis of the dataset



dataset_anonymized = df.drop(["Target"], axis=1)
dataset_anonymized.to_csv('dataset_para_regresion_anonymized.csv', index=False)
dataset_anonymized.corr()


# ## Separation of dataset features and labels



X = dataset_anonymized
y = df.get("Target")
print('Class labels:', np.unique(y))


# ## Visualization of the correlation matrix in a heat map




cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)

plt.tight_layout()

plt.show()


# ## Viewing distributions in histograms




scatterplotmatrix(df.values, figsize=(12, 10), 
                  names=df.columns, alpha=0.5)

plt.tight_layout()

plt.show()

sb.pairplot(df)
plt.show()


# ## Division of the dataset into training (70%) and test (30%)




target = 'Target'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)





slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)




x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
ax1.set_ylabel('Residuals')

for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)

plt.tight_layout()

plt.show()


# ## Calculation of mean squared error




mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')


# ## Calculation of the mean absolute error




mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')


# ## Calculation of the coefficient of determination




r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')


# # Regression Methods

# ## Training and evaluation of the model by linear regression



X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


slr = LinearRegression()

slr.fit(X_train, y_train)




y_pred = slr.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)




coefficients = slr.coef_
intercept = slr.intercept_

print("Coefficients:", coefficients)

print("Intercept:", intercept)


# ## Training and evaluation of the model by quadratic polynomial regression



X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic.fit_transform(X_train)

regr_quadratic = regr.fit(X_train_quadratic, y_train)

print("Quadratic Model Coefficients:", regr_quadratic.coef_)
print("Quadratic Model Intercept:", regr_quadratic.intercept_)

new_data_quadratic = np.array([[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]])

transformed_new_data_quadratic = quadratic.transform(new_data_quadratic)
print("Quadratic Transformed Data:", transformed_new_data_quadratic[0])

predicted_target_quadratic = regr_quadratic.predict(transformed_new_data_quadratic)
print("Predicted Target:", predicted_target_quadratic)

# CÁLCULO MANUAL
# coefficients = regr_quadratic.coef_
# intercept = regr_quadratic.intercept_
# manual_prediction = np.dot(coefficients, transformed_new_data_quadratic[0]) + intercept
# print("Manually Calculated Target:", manual_prediction)




X_test_quadratic = quadratic.fit_transform(X_test)

y_pred_quadratic = regr.predict(X_test_quadratic)

mae = mean_absolute_error(y_test, y_pred_quadratic)
mse = mean_squared_error(y_test, y_pred_quadratic)
R2 = r2_score(y_test, y_pred_quadratic)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# ## Training and evaluation of the model by cubic polynomial regression






X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

regr = LinearRegression()

cubic = PolynomialFeatures(degree=3)
X_train_cubic = cubic.fit_transform(X_train)

regr_cubic = regr.fit(X_train_cubic, y_train)

print("Cubic Model Coefficients:", regr_cubic.coef_)
print("Cubic Model Intercept:", regr_cubic.intercept_)

new_data_cubic = np.array([[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]])

transformed_new_data_cubic = cubic.transform(new_data_cubic)
print("Cubic Transformed Data:", transformed_new_data_cubic[0])

predicted_target_cubic = regr_cubic.predict(transformed_new_data_cubic)
print("Predicted Target:", predicted_target_cubic)

# CÁLCULO MANUAL
# coefficients = regr_cubic.coef_
# intercept = regr_cubic.intercept_
# manual_prediction = np.dot(coefficients, transformed_new_data_cubic[0]) + intercept
# print("Manually Calculated Target:", manual_prediction)




X_test_cubic = cubic.fit_transform(X_test)

y_pred_cubic = regr_cubic.predict(X_test_cubic)

# print(y_pred_cubic)




# Calculate evaluation metrics


mae = mean_absolute_error(y_test, y_pred_cubic)
mse = mean_squared_error(y_test, y_pred_cubic)
R2 = r2_score(y_test, y_pred_cubic)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# ## Training and evaluation of the model by Decision Tree regression





X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)




y_pred_random_tree = tree.predict(X_test)

# print(y_pred_random_tree)





mae = mean_absolute_error(y_test, y_pred_random_tree)
mse = mean_squared_error(y_test, y_pred_random_tree)
R2 = r2_score(y_test, y_pred_random_tree)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# ## Training and evaluation of the model by Random Forest regression





X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

x_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='squared_error',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)

y_pred_random_forest = forest.predict(X_test)





mae = mean_absolute_error(y_test, y_pred_random_forest)
mse = mean_squared_error(y_test, y_pred_random_forest)
R2 = r2_score(y_test, y_pred_random_forest)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# # Summary

# ---
# 
# Readers may ignore the next cell.



# Run a command in the terminal from a Python environment (such as a Jupyter Notebook or a 
# script that allows system commands to convert a Jupyter notebook to a file Python script. 
# * !
# This symbol is used in environments such as Jupyter Notebooks to execute system commands 
# operational directly from the notebook. In this case, the command is an execution of a 
# Python Script.
# * python ../.convert_notebook_to_script.py
# This command runs a Python script called convert_notebook_to_script.py. This file 
# is located in the previous directory (../ indicates that it is one level up in the system 
# files). The purpose of this script is to convert a Jupyter notebook (.ipynb) into a 
# Python script file (.py).
# * --input ex03.ipynb
# This is an option or argument that tells the script what the input file is, in this 
# case, the notebook ex03.ipynb.
# * --output ex03.py
# This option tells the script to save the output (the converted file) with the name
# ex03.py, which is a Python script.


