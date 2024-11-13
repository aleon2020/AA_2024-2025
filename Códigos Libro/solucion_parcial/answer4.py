# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix

# # PREGUNTA 4

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

# Se importan los paquetes esenciales para analizar y visualizar datos: numpy para cálculos numéricos, pandas para manipular datos y matplotlib.pyplot para visualizar gráficos.





# ---



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Target']
df = pd.read_csv("dataset_regression.csv", 
                 sep=',',
                 usecols=columns)
pd.set_option('display.max_columns', len(df.columns))
df.columns
df.shape
df.head(1)
df.info()
df.describe()




dataset_regression_anonymized = df.drop(["Target"], axis=1)
dataset_regression_anonymized.to_csv('dataset_regression_anonymized.csv', index=False)
dataset_regression_anonymized.corr()




X = dataset_regression_anonymized
y = df.get("Target")
print('Class labels:', np.unique(y))




fig, ax = plt.subplots(figsize=(9,9))
sb.heatmap(df.corr(), linewidth = 0.5, annot=True)




cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()




columns = dataset_regression_anonymized.columns
fig = plt.figure(figsize=(12,12))
for i in range(0,8):
  ax = plt.subplot(3,3,i+1)
  ax.hist(dataset_regression_anonymized[columns[i]],bins = 20, color='blue', edgecolor='black')
  ax.set_title(dataset_regression_anonymized.head(0)[columns[i]].name)
plt.tight_layout()
plt.show()




scatterplotmatrix(df.values, figsize=(12, 10), 
                  names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()




sb.pairplot(df)
plt.show()


# ---

# ## Convertir Jupyter Notebook a Fichero Python

# ### Script en el Directorio Actual





# ### Script en el Directorio Padre




