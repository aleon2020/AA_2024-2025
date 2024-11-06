# coding: utf-8


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

# # Práctica 1 Pregunta 1: Análisis exploratorio de los datos

# Importa los datos del fichero dataset.csv y realiza el análisis exploratorio de los datos. Describe en el informe los resultados de este análisis y deposita el código Python en Aula Virtual en el fichero 'answer1.ipynb'.

# ## Importación de bibliotecas para análisis de datos y escalado





# ## Exploración inicial del dataset con Pandas



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6',
           'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Target']

df = pd.read_csv('dataset_practica_2.csv', 
                 sep=',',
                 usecols=columns)

df.columns

df.shape

df.head(1)

df.info()

df.describe()


# ## Anonimización y análisis de la correlación del dataset



dataset_anonymized = df.drop(["Target"], axis=1)
dataset_anonymized.to_csv('dataset_practica_2_anonymized.csv', index=False)
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


# IMPORTANTE: Para las preguntas 3 y 4, habrá que seleccionar 4 y 2 de las mejores columnas entre todas las posibles, respectivamente. Para saber qué columnas poseen una mayor/mejor correlación, hay que hacer lo siguiente.
# 
# PASO 1: Coger todos los valores de la fila 'Target' y pasarlos a valor absoluto.
# 
# - Col1:   |0.57|  = 0.57
# - Col2:   |-0.30| = 0.30
# - Col3:   |-0.23| = 0.23
# - Col4:   |0.06|  = 0.06
# - Col5:   |0.28|  = 0.28
# - Col6:   |0.07|  = 0.07
# - Col7:   |-0.10| = 0.10
# - Col8:   |0.63|  = 0.63
# - Col9:   |0.63|  = 0.63
# - Col10:  |0.41|  = 0.41
# - Col11:  |-0.07| = 0.07
# 
# Sabiendo esto, podemos escoger para la pregunta 3, en la que se pide calcular la regresión usando solo 4 características del dataset, las columnas 1, 8, 9 y 10.
# 
# Y por último, para la pregunta 4, en la que se pide calcular la regresión usando solo 2 características del dataset, las columnas 8 y 9.

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


# ## Conversión de Jupyter Notebook en un archivo Python




