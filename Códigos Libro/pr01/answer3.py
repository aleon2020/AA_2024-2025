# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# # Práctica 1 Pregunta 3: Clasificación usando 4 características de los datos

# Selecciona 4 características de los datos. Usando estas características, implementa los métodos Logistic Regression, SVM y Random Trees para clasificar los datos. Describe en el informe los parámetros usados y los resultados obtenidos con los distintos métodos y deposita el código Python en Aula Virtual en el fichero 'answer3.ipynb'.

# ## Importación de bibliotecas para análisis de datos y escalado





# ## Carga del dataset desde un archivo CSV



dataset = pd.read_csv("dataset.csv")


# ## Anonimización y análisis de la correlación del dataset



dataset_anonymized = dataset.drop(["Target"], axis=1)
dataset_4_characteristics = dataset_anonymized.drop(["Col2", "Col3", "Col4", "Col6", "Col8", "Col9", "Col10"], axis=1)
dataset_4_characteristics.to_csv('dataset_4_characteristics.csv', index=False)
dataset_4_characteristics.corr()


# ## Separación de características y etiquetas del dataset



X = dataset_4_characteristics
y = dataset.get("Target")
print('Class labels:', np.unique(y))


# ## División del dataset en entrenamiento (75%) y prueba (25%)



X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.25, random_state=1, stratify=y)


# ## Estandarización del balance de clases



sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ## Entrenamiento y evaluación del modelo por regresión logística (Logistic Regression)



lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % lr.score(X_test_std, y_test))


# ## Entrenamiento y evaluación del modelo por máquinas de soporte vectorial (SVM)



svm = SVC(kernel='rbf', random_state=1, gamma=0.7, C=30.0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % svm.score(X_test_std, y_test))


# ## Entrenamiento y evaluación del modelo por árboles de decisión (Random Trees)



tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))




feature_names = ['Col1', 'Col5', 'Col7', 'Col11']
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)
plt.show()
y_pred = tree_model.predict(X_test)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % tree_model.score(X_test, y_test))


# ## Conversión de Jupyter Notebook en un archivo Python




