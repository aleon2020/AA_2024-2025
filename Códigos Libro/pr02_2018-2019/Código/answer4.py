# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# # Práctica 1 Pregunta 4: Regresión usando 2 características de los datos

# Selecciona 2 características de los datos. Usando estas características, implementa los mismos métodos de regresión del punto 2). Describe en el informe los parámetros usados y los resultados obtenidos con los distintos métodos y deposita el código Python en el Aula Virtual en el fichero 'answer4.ipynb'.

# ## Importación de bibliotecas para análisis de datos y escalado





# ## Carga del dataset desde un archivo CSV



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6',
           'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Target']

df = pd.read_csv('dataset_practica_2.csv', 
                 sep=',',
                 usecols=columns)


# ## Entrenamiento y evaluación del modelo por regresión lineal (Linear Regression)



X = df[['Col8', 'Col9']].values
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


# ## Entrenamiento y evaluación del modelo por regresión polinómica cuadrática (Quadratic Polynomial Regression)



X = df[['Col8', 'Col9']].values
y = df['Target'].values



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic.fit_transform(X_train)

regr_quadratic = regr.fit(X_train_quadratic, y_train)

print("Quadratic Model Coefficients:", regr_quadratic.coef_)
print("Quadratic Model Intercept:", regr_quadratic.intercept_)

new_data_quadratic = np.array([[2, 4]])

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


# ## Entrenamiento y evaluación del modelo por regresión polinómica cúbica (Cubic Polynomial Regression)






X = df[['Col8', 'Col9']].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

regr = LinearRegression()

cubic = PolynomialFeatures(degree=3)
X_train_cubic = cubic.fit_transform(X_train)

regr_cubic = regr.fit(X_train_cubic, y_train)

print("Cubic Model Coefficients:", regr_cubic.coef_)
print("Cubic Model Intercept:", regr_cubic.intercept_)

new_data_cubic = np.array([[2, 4]])

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


# ## Entrenamiento y evaluación del modelo por árboles de decisión (Decision Tree Regression)





X = df[['Col8', 'Col9']].values
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


# ## Entrenamiento y evaluación del modelo por bosques aleatorios (Random Forest Regression)





X = df[['Col8', 'Col9']].values
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


# ## Conversión de Jupyter Notebook en un archivo Python




