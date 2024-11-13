# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# # PREGUNTA 3

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



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Target']
df = pd.read_csv("dataset_classification.csv", 
                 sep=',',
                 usecols=columns)




dataset_classification_anonymized = df.drop(["Target"], axis=1)
dataset_classification_anonymized.to_csv('dataset_classification_anonymized.csv', index=False)
dataset_classification_anonymized.corr()




X = dataset_classification_anonymized
y = df.get("Target")




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y)




print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))





# PARÁMETROS POR DEFECTO
tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)

# PARÁMETROS ÓPTIMOS
# tree_model = DecisionTreeClassifier(criterion='gini', 
#                                     max_depth=10, 
#                                     random_state=1)

tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))




feature_names = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6']
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)
plt.show()
y_pred = tree_model.predict(X_test)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % tree_model.score(X_test, y_test))




input_data = pd.DataFrame([[1, 3, 1, 2, 3, 2]], columns=feature_names)
print(f'Class probability: {tree_model.predict_proba(input_data)}')
print('Most probable class: %d' % tree_model.predict(input_data)[0])


# ---

# ## Convertir Jupyter Notebook a Fichero Python

# ### Script en el Directorio Actual





# ### Script en el Directorio Padre




