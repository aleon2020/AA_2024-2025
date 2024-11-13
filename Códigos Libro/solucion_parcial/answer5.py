# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# # PREGUNTA 5

# ## Configuración de las Rutas de Importación

# Se añade el directorio padre (..) al path (sys.path), lo que permite al entorno de Python acceder a módulos o paquetes ubicados en directorios superiores al actual. Esto es útil para poder importar scripts o paquetes personalizados sin tener que mover ficheros o el directorio de trabajo.



sys.path.insert(0, '..')


# ## Verificación de las Versiones de los Paquetes

# Se utiliza la función check_packages() para verificar que los paquetes y sus respectivas versiones indicadas en el diccionario 'd' estén instalados correctamente dentro del entorno. Este paso es importante para verificar la compatibilidad de cada paquete para poder evitar errores por diferencia de versión.



d = {
    'numpy': '1.21.2',
    'scipy': '1.7.0',
    'mlxtend' : '0.19.0',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# ## Importación de Paquetes

# Se importan los paquetes esenciales para analizar y visualizar datos: numpy para cálculos numéricos, pandas para manipular datos y matplotlib.pyplot para visualizar gráficos, entre otros.





# ---



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Target']
df = pd.read_csv("dataset_regression.csv", 
                 sep=',',
                 usecols=columns)




X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8']].values
y = df['Target'].values


# PARÁMETROS POR DEFECTO
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123)

# PARÁMETROS ÓPTIMOS
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=1)




regr = LinearRegression()

linear = PolynomialFeatures(degree=1)
X_train_linear = linear.fit_transform(X_train)

regr_linear = regr.fit(X_train_linear, y_train)

print("Linear Model Coefficients:", regr_linear.coef_)
print("Linear Model Intercept:", regr_linear.intercept_)

new_data_linear = np.array([[180.0, 10.4, 120.0, 28, 162.0, 765.0, 830.0, 275.0]])

transformed_new_data_linear = linear.transform(new_data_linear)
print("Linear Transformed Data:", transformed_new_data_linear[0])

predicted_target_linear = regr_linear.predict(transformed_new_data_linear)
print("Predicted Target:", predicted_target_linear)




X_train_linear = linear.fit_transform(X_train)
X_test_linear = linear.fit_transform(X_test)

y_train_linear = regr.predict(X_train_linear)
y_test_linear = regr.predict(X_test_linear)





mae_train = mean_absolute_error(y_train, y_train_linear)
mae_test = mean_absolute_error(y_test, y_test_linear)

mse_train = mean_squared_error(y_train, y_train_linear)
mse_test = mean_squared_error(y_test, y_test_linear)

r2_train = r2_score(y_train, y_train_linear)
r2_test = r2_score(y_test, y_test_linear)

print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')

print(f'R² train: {r2_train:.2f}')
print(f'R² test: {r2_test:.2f}')


# ---

# ## Convertir Jupyter Notebook a Fichero Python

# ### Script en el Directorio Actual





# ### Script en el Directorio Padre




