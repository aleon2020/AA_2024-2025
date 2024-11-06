# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn.model_selection import train_test_split

# # Práctica 1 Pregunta 1: Análisis exploratorio de los datos

# Importa los datos del fichero dataset.csv y realiza el análisis exploratorio de los datos. Describe en el informe los resultados de este análisis y deposita el código Python en Aula Virtual en el fichero 'answer1.ipynb'.

# ## Importación de bibliotecas para análisis de datos y escalado





# ## Exploración inicial del dataset con Pandas



dataset = pd.read_csv("dataset.csv")
pd.set_option('display.max_columns', len(dataset.columns))

dataset.columns

dataset.shape

dataset.head(1)

dataset.info()

dataset.describe()


# ## Anonimización y análisis de la correlación del dataset



dataset_anonymized = dataset.drop(["Target"], axis=1)
dataset_anonymized.to_csv('dataset_anonymized.csv', index=False)
dataset_anonymized.corr()


# ## Separación de características y etiquetas del dataset



X = dataset_anonymized
y = dataset.get("Target")
print('Class labels:', np.unique(y))


# ## Visualización de la matriz de correlación en un mapa de calor



fig, ax = plt.subplots(figsize=(9,9))
sb.heatmap(dataset.corr(), linewidth = 0.5, annot=True)


# ## Visualización de las distribuciones en histogramas



columns = dataset_anonymized.columns
fig = plt.figure(figsize=(12,12))
for i in range(0,11):
  ax = plt.subplot(4,4,i+1)
  ax.hist(dataset_anonymized[columns[i]],bins = 20, color='blue', edgecolor='black')
  ax.set_title(dataset_anonymized.head(0)[columns[i]].name)
plt.tight_layout()
plt.show()


# ## División del dataset en entrenamiento (75%) y prueba (25%)



X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.25, random_state=1, stratify=y)


# ## Estandarización del balance de clases



sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)




print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# ## Conversión de Jupyter Notebook en un archivo Python




