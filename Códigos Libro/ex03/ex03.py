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
# Importa el módulo sys, que es un módulo de la biblioteca estándar de Python.
# Este módulo proporciona acceso a variables y funciones que interactúan fuertemente con el
# intérprete de Python, como la manipulación de la ruta de búsqueda de módulos y la entrada/salida
# estándar, entre otros.
# * sys.path
# Es una lista que contiene las rutas en las que el intérprete de Python busca los módulos cuando
# usas import. Al intentar importar un módulo, Python busca en las rutas especificadas en esta
# lista.
# * sys.path.insert(0, '..')
# Inserta la ruta '..' (que representa el directorio padre) al inicio de la lista sys.path.
# Al agregarla en la posición 0, se asegura que cuando Python busque módulos para importar,
# primero verifique en el directorio padre antes de continuar con las rutas predeterminadas.

sys.path.insert(0, '..')


# Check recommended package versions:



# Importa la función check_packages desde el módulo python_environment_check. 
# Este módulo, por su nombre, parece estar diseñado para verificar que el entorno de Python 
# tenga instaladas las versiones correctas de ciertos paquetes.
# * d = {...}
# Define un diccionario d que contiene como claves los nombres de varios paquetes 
# (por ejemplo, numpy, scipy, matplotlib, etc.) y como valores las versiones mínimas 
# requeridas de esos paquetes.
# * check_packages(d)
# La función check_packages toma como entrada el diccionario d y probablemente realiza una 
# verificación en el entorno actual de Python para asegurarse de que las versiones instaladas 
# de estos paquetes sean al menos las especificadas en el diccionario. Si alguno de los paquetes 
# no está instalado o tiene una versión incorrecta, es posible que la función lance un error o 
# sugiera instalar/actualizar los paquetes.

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

# - [Importación de bibliotecas para análisis de datos y escalado](#importación-de-bibliotecas-para-análisis-de-datos-y-escalado)
# - [Carga del dataset desde un archivo CSV](#carga-del-dataset-desde-un-archivo-csv)
# - [Anonimización y análisis de la correlación del dataset](#anonimización-y-análisis-de-la-correlación-del-dataset)
# - [Separación de características y etiquetas del dataset](#separación-de-características-y-etiquetas-del-dataset)
# - [Visualización de la matriz de correlación en un mapa de calor](#visualización-de-la-matriz-de-correlación-en-un-mapa-de-calor)
# - [Visualización de las distribuciones en histogramas](#visualización-de-las-distribuciones-en-histogramas)
# - [División del dataset en entrenamiento (70%) y prueba (30%)](#división-del-dataset-en-entrenamiento-70-y-prueba-30)
# - [Cálculo del error cuadrático medio](#cálculo-del-error-cuadrático-medio)
# - [Cálculo del error absoluto medio](#cálculo-del-error-absoluto-medio)
# - [Cálculo del coeficiente de determinación](#cálculo-del-coeficiente-de-determinación)
# - [Entrenamiento y evaluación del modelo por regresión lineal (Linear Regression)](#entrenamiento-y-evaluación-del-modelo-por-regresión-lineal-linear-regression)
# - [Entrenamiento y evaluación del modelo por regresión polinómica cuadrática (Quadratic Polynomial Regression)](#entrenamiento-y-evaluación-del-modelo-por-regresión-polinómica-cuadrática-quadratic-polynomial-regression)
# - [Entrenamiento y evaluación del modelo por regresión polinómica cúbica (Cubic Polynomial Regression)](#entrenamiento-y-evaluación-del-modelo-por-regresión-polinómica-cúbica-cubic-polynomial-regression)
# - [Entrenamiento y evaluación del modelo por árboles de decisión (Decision Tree Regression)](#entrenamiento-y-evaluación-del-modelo-por-árboles-de-decisión-decision-tree-regression)
# - [Entrenamiento y evaluación del modelo por bosques aleatorios (Random Forest Regression)](#entrenamiento-y-evaluación-del-modelo-por-bosques-aleatorios-random-forest-regression)
# - [Conversión de Jupyter Notebook en un archivo Python](#conversión-de-jupyter-notebook-en-un-archivo-python)



# * from IPython.display
# Importa desde el submódulo display del paquete IPython. Este módulo está diseñado para mostrar 
# y renderizar diferentes tipos de datos dentro de entornos interactivos, como Jupyter Notebooks.
# * import Image
# Importa la clase Image desde el módulo display. La clase Image se utiliza para mostrar 
# imágenes en el entorno interactivo (por ejemplo, en una celda de Jupyter Notebook).
# * %matplotlib inline
# Esto es una "magic command" (comando mágico) específico de IPython/Jupyter Notebook.
# Habilita la visualización de gráficos de matplotlib directamente dentro de las celdas del 
# notebook. Los gráficos se renderizan "en línea" (dentro del mismo cuaderno) sin necesidad 
# de abrir ventanas emergentes.



# ## Importación de bibliotecas para análisis de datos y escalado





# ## Carga del dataset desde un archivo CSV



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


# ## Anonimización y análisis de la correlación del dataset



dataset_anonymized = df.drop(["Target"], axis=1)
dataset_anonymized.to_csv('dataset_para_regresion_anonymized.csv', index=False)
dataset_anonymized.corr()


# ## Separación de características y etiquetas del dataset



X = dataset_anonymized
y = df.get("Target")
print('Class labels:', np.unique(y))


# ## Visualización de la matriz de correlación en un mapa de calor




cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)

plt.tight_layout()

plt.show()


# ## Visualización de las distribuciones en histogramas




scatterplotmatrix(df.values, figsize=(12, 10), 
                  names=df.columns, alpha=0.5)

plt.tight_layout()

plt.show()

sb.pairplot(df)
plt.show()


# ## División del dataset en entrenamiento (70%) y prueba (30%)




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


# ## Cálculo del error cuadrático medio




mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')


# ## Cálculo del error absoluto medio




mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')


# ## Cálculo del coeficiente de determinación




r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')


# ## Entrenamiento y evaluación del modelo por regresión lineal (Linear Regression)



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


# ## Entrenamiento y evaluación del modelo por regresión polinómica cuadrática (Quadratic Polynomial Regression)



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

# new_data_quadratic = np.array([[7, 1000, 1500]])

# transformed_new_data_quadratic = quadratic.transform(new_data_quadratic)

# print("Quadratic Trasformed Data:", transformed_new_data_quadratic[0])

# predicted_price = regr_quadratic.predict(transformed_new_data_quadratic)

# print("Predicted SalePrice:", predicted_price)




# coefficients = regr_quadratic.coef_
# intercept = regr_quadratic.intercept_

# transformed_new_data = np.array(
#     [1,             # Intercept Term
#     7,              # X1 (Overall Qual)
#     1000,           # X2 (Total Bsmt SF)
#     1500,           # X3 (Gr Liv Area)
#     49,             # X1 ^ 2
#     7000,           # X1 * X2
#     10500,          # X1 * X3
#     1000000,        # X2 ^ 2
#     1500000,        # X2 * X3
#     2250000])       # X3 ^ 2

# manual_prediction = np.dot(coefficients, transformed_new_data) + intercept

# print("Manually Calculated SalePrice:", manual_prediction)




X_test_quadratic = quadratic.fit_transform(X_test)

y_pred_quadratic = regr.predict(X_test_quadratic)

mae = mean_absolute_error(y_test, y_pred_quadratic)
mse = mean_squared_error(y_test, y_pred_quadratic)
R2 = r2_score(y_test, y_pred_quadratic)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# ## Entrenamiento y evaluación del modelo por regresión polinómica cúbica (Cubic Polynomial Regression)






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

# new_data_cubic = np.array([[7, 1000, 1500]])

# transformed_new_data_cubic = cubic.transform(new_data_cubic)

# print("Cubic Transformed Data:", transformed_new_data_cubic[0])

# predicted_price_cubic = regr_cubic.predict(transformed_new_data_cubic)

# print("Predicted SalePrice:", predicted_price_cubic)




# coefficients_cubic = regr_cubic.coef_
# intercept_cubic = regr_cubic.intercept_

# transformed_new_data_cubic = np.array(
#     [1,             # Intercept Term
#     7,              # X1 (Overall Qual)
#     1000,           # X2 (Total Bsmt SF)
#     1500,           # X3 (Gr Liv Area)
#     49,             # X1 ^ 2
#     7000,           # X1 * X2
#     10500,          # X1 * X3
#     1000000,        # X2 ^ 2
#     1500000,        # X2 * X3
#     2250000,        # X3 ^ 2
#     343,            # X2 ^ 3
#     49000,          # X1 ^ 2 * X2
#     73500,          # X1 ^ 2 * X3
#     7000000,        # X2 ^ 2 * X1
#     10500000,       # X1 * X2 * X3
#     15750000,       # X3 ^ 2 * X1
#     1000000000,     # X2 ^ 3
#     1500000000,     # X2 ^ 2 * X3
#     2250000000,     # X3 ^ 2 * X2
#     3375000000])    # X3 ^ 3

# manual_prediction_cubic = np.dot(coefficients_cubic, transformed_new_data_cubic) + intercept_cubic

# print("Manually Calculated SalePrice:", manual_prediction_cubic)




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


# ## Entrenamiento y evaluación del modelo por bosques aleatorios (Random Forest Regression)





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


# ## Conversión de Jupyter Notebook en un archivo Python




