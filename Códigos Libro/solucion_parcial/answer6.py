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

# # PREGUNTA 6

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
    X, y, test_size=0.30, random_state=123)

# PARÁMETROS ÓPTIMOS
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.30, random_state=1)




regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic.fit_transform(X_train)

regr_quadratic = regr.fit(X_train_quadratic, y_train)

print("Quadratic Model Coefficients:", regr_quadratic.coef_)
print("Quadratic Model Intercept:", regr_quadratic.intercept_)

new_data_quadratic = np.array([[207.0, 5.0, 161.0, 28, 179.0, 736.0, 867.0, 132.0]])

transformed_new_data_quadratic = quadratic.transform(new_data_quadratic)
print("Quadratic Transformed Data:", transformed_new_data_quadratic[0])

predicted_target_quadratic = regr_quadratic.predict(transformed_new_data_quadratic)
print("Predicted Target:", predicted_target_quadratic)




X_train_quadratic = quadratic.fit_transform(X_train)
X_test_quadratic = quadratic.fit_transform(X_test)

y_train_quadratic = regr.predict(X_train_quadratic)
y_test_quadratic = regr.predict(X_test_quadratic)





mae_train = mean_absolute_error(y_train, y_train_quadratic)
mae_test = mean_absolute_error(y_test, y_test_quadratic)

mse_train = mean_squared_error(y_train, y_train_quadratic)
mse_test = mean_squared_error(y_test, y_test_quadratic)

r2_train = r2_score(y_train, y_train_quadratic)
r2_test = r2_score(y_test, y_test_quadratic)

print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')

print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

print(f'R² train: {r2_train:.2f}')
print(f'R² test: {r2_test:.2f}')


# ---

# ## Convertir Jupyter Notebook a Fichero Python

# ### Script en el Directorio Actual





# ### Script en el Directorio Padre




