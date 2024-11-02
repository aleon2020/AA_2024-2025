# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# # FORMULARIO EXAMEN PARCIAL 1

# # ÍNDICE

# - [TEMA 1: Configuración y Visualización del Entorno](#tema-1-configuración-y-visualización-del-entorno)
#     - [1.1 Configuración de las Rutas de Importación](#11-configuración-de-las-rutas-de-importación)
#     - [1.2 Verificación de las Versiones de los Paquetes](#12-verificación-de-las-versiones-de-los-paquetes)
#     - [1.3 Visualización de Imágenes](#13-visualización-de-imágenes)
#     - [1.4 Importación de Paquetes](#14-importación-de-paquetes)
# - [TEMA 2: Análisis Exploratorio de Datos (Clasificadores)](#tema-2-análisis-exploratorio-de-datos-clasificadores)
#     - [2.1 Carga y Exploración Inicial del Dataset](#21-carga-y-exploración-inicial-del-dataset)
#     - [2.2 Anonimización y Cálculo de la Correlación entre Características](#22-anonimización-y-cálculo-de-la-correlación-entre-características)
#     - [2.3 División de Variables Independientes y Dependiente](#23-división-de-variables-independientes-y-dependiente)
#     - [2.4 Mapa de Calor de Correlaciones](#24-mapa-de-calor-de-correlaciones)
#     - [2.5 Histogramas de Distribución de las Características](#25-histogramas-de-distribución-de-las-características)
#     - [2.6 División de Datos y Estandarización](#26-división-de-datos-y-estandarización)
# - [TEMA 3: Clasificadores](#tema-3-clasificadores)
#     - [3.1 Regresión Logística (Logistic Regression)](#31-regresión-logística-logistic-regression)
#     - [3.2 Máquinas de Soporte Vectorial (SVM)](#32-máquinas-de-soporte-vectorial-svm)
#     - [3.3 Árboles de Decisión (Decision Trees)](#33-árboles-de-decisión-decision-trees)
#     - [3.4 Bosque Aleatorio (Random Forest)](#34-bosque-aleatorio-random-forest)
#     - [3.5 Vecinos Más Cercanos (KNN)](#35-vecinos-más-cercanos-knn)
# - [TEMA 4: Análisis Exploratorio de Datos (Regresión)](#tema-4-análisis-exploratorio-de-datos-regresión)
#     - [4.1 Carga y Exploración Inicial del Dataset](#41-carga-y-exploración-inicial-del-dataset)
#     - [4.2 Anonimización y Cálculo de la Correlación entre Características](#42-anonimización-y-cálculo-de-la-correlación-entre-características)
#     - [4.3 División de Variables Independientes y Dependiente](#43-división-de-variables-independientes-y-dependiente)
#     - [4.4 Mapa de Calor de Correlaciones](#44-mapa-de-calor-de-correlaciones)
#     - [4.5 Histogramas de Distribución de las Características](#45-histogramas-de-distribución-de-las-características)
#     - [4.6 División de Datos y Estandarización](#46-división-de-datos-y-estandarización)
#     - [4.7 Error Cuadrático Medio (MSE)](#47-error-cuadrático-medio-mse)
#     - [4.8 Error Absoluto Medio (MAE)](#48-error-absoluto-medio-mae)
#     - [4.9 Coeficiente de Determinación (R²)](#49-coeficiente-de-determinación-r)
# - [TEMA 5: Métodos de Regresión](#tema-5-métodos-de-regresión)
#     - [5.1 Regresión Lineal (Linear Regression)](#51-regresión-lineal-linear-regression)
#     - [5.2 Regresión Cuadrática (Quadratic Regression)](#52-regresión-cuadrática-quadratic-regression)
#     - [5.3 Regresión Cúbica (Cubic Regression)](#53-regresión-cúbica-cubic-regression)
#     - [5.4 Regresión con Árboles de Decisión (Decision Tree Regression)](#54-regresión-con-árboles-de-decisión-decision-tree-regression)
#     - [5.5 Regresión con Bosque Aleatorio (Random Forest Regression)](#55-regresión-con-bosque-aleatorio-random-forest-regression)
# - [ANEXO: Convertir Jupyter Notebook a Fichero Python](#anexo-convertir-jupyter-notebook-a-fichero-python)
#     - [A.1 Script en el Directorio Actual](#a1-script-en-el-directorio-actual)
#     - [A.2 Script en el Directorio Padre](#a2-script-en-el-directorio-padre)

# # TEMA 1: Configuración y Visualización del Entorno

# ## 1.1 Configuración de las Rutas de Importación

# Se añade el directorio padre (..) al path (sys.path), lo que permite al entorno de Python acceder a módulos o paquetes ubicados en directorios superiores al actual. Esto es útil para poder importar scripts o paquetes personalizados sin tener que mover ficheros o el directorio de trabajo.



sys.path.insert(0, '..')


# ## 1.2 Verificación de las Versiones de los Paquetes

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


# ## 1.3 Visualización de Imágenes

# Se utiliza display de IPython.display para renderizar una imagen en HTML.









# Este código genera un contenedor div centrado, que muestra una imagen con unas dimensiones específicas. Esto es útil para insertar gráficos u otras figuras en el Notebook, lo que facilita la visualización de resultados o ejemplos.



display(HTML("""
<div style="display: flex; justify-content: center;">
    <img src="./figures/01_01.png" width="500" height="300" format="png">
</div>
"""))


# ## 1.4 Importación de Paquetes

# Se importan los paquetes esenciales para analizar y visualizar datos: numpy para cálculos numéricos, pandas para manipular datos y matplotlib.pyplot para visualizar gráficos.





# # TEMA 2: Análisis Exploratorio de Datos (Clasificadores)

# ## 2.1 Carga y Exploración Inicial del Dataset

# En primer lugar, se carga el dataset 'dataset.csv' y se configura la visualización de todas sus columnas. Además, se exploran las características más importantes del dataset como las columnas, su forma, una muestra de los primeros registros, así como un resumen de la información general y otras estadísticas descriptivas.



dataset = pd.read_csv("dataset.csv")
pd.set_option('display.max_columns', len(dataset.columns))

dataset.columns
dataset.shape
dataset.head(1)
dataset.info()
dataset.describe()


# ANÁLISIS DE LOS RESULTADOS
# 
# El dataset contiene un total de 12 columnas y 1599 filas. La exploración inicial revela la media, la desviación estándar, el mínimo, el máximo y los cuartiles de cada columna, lo que permite observar detalladamente la dispersión y la distribución de los datos, además de poder detectar problemas como valores nulos o distribuciones atípicas que puedan afectar al análsis posterior.

# ## 2.2 Anonimización y Cálculo de la Correlación entre Características

# Una vez realizado el primer análisis general de los datos, se elimina la columna 'Target' para crear un dataset anonimizado (dataset_anonymized), el cual se guarda en un fichero con el mismo nombre. Después, se calcula la matriz de correlación, que permite evaluar relaciones lineales entre las columnas.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
dataset_anonymized_classifiers = dataset.drop(["Target"], axis=1)
dataset_anonymized_classifiers.to_csv('dataset_anonymized_classifiers.csv', index=False)
dataset_anonymized_classifiers.corr()

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# dataset_anonymized_classifiers = dataset.drop(["Target"], axis=1)
# dataset_N_characteristics = dataset_anonymized_classifiers.drop(["Col1", "Col2", ..., "ColN"], axis=1)
# dataset_N_characteristics.to_csv('dataset_N_characteristics.csv', index=False)
# dataset_N_characteristics.corr()


# ## 2.3 División de Variables Independientes y Dependiente

# Después de anonimizar los datos, se separan las variables independientes (X) de la variable objetivo (y), en este caso 'Target', y se imprimen las etiquetas de clase únicas, las cuales permiten preparar el dataset para su uso en modelos de clasificación.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = dataset_anonymized_classifiers
y = dataset.get("Target")
print('Class labels:', np.unique(y))

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = dataset_N_characteristics
# y = dataset.get("Target")
# print('Class labels:', np.unique(y))


# ANÁLISIS DE LOS RESULTADOS
# 
# El dataset contiene 6 clases (3, 4, 5, 6, 7 y 8), lo que implica un problema de clasificación con múltiples etiquetas. El balance entre clases se verifica a través de la distribución de etiquetas, permitiendo conocer si existe un desbalance significativo que pudiera requerir estrategias adicionales de ajuste o balanceo en el modelo.

# ## 2.4 Mapa de Calor de Correlaciones

# Con las variables X e y ya definidas, se genera un mapa de calor que representa gráficamente a la matriz de correlación de todas las características del dataset, lo que facilita la detección visual de las relaciones entre variables.



fig, ax = plt.subplots(figsize=(9,9))
sb.heatmap(dataset.corr(), linewidth = 0.5, annot=True)


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el mapa de calor muestra varias relaciones significativas entre algunas características.
# 
# Para otros apartados, habrá que seleccionar N de las mejores columnas entre todas las posibles, respectivamente. Para saber qué columnas poseen una mayor/mejor correlación, se seleccionan todos los valores de la fila 'Target' y pasarlos a valor absoluto, y una vez hecho esto, seleccionar los N valores más altos.
# 
# - Col1:   |-0.39|   =   0.39
# - Col2:   |-0.17|   =   0.17
# - Col3:   |-0.058|  =   0.058
# - Col4:   |-0.19|   =   0.19
# - Col5:   |0.25|    =   0.25
# - Col6:   |-0.13|   =   0.13
# - Col7:   |0.48|    =   0.48
# - Col8:   |-0.051|  =   0.051
# - Col9:   |0.12|    =   0.12
# - Col10:  |0.014|   =   0.014
# - Col11:  |0.23|    =   0.23

# ## 2.5 Histogramas de Distribución de las Características

# El siguiente bloque de código crea un histograma para cada una de las 11 columnas que componen el dataset anonimizado, donde se visualiza la distribución de cada característica en función de su frecuencia.



columns = dataset_anonymized_classifiers.columns
fig = plt.figure(figsize=(12,12))
for i in range(0,11):
  ax = plt.subplot(4,4,i+1)
  ax.hist(dataset_anonymized_classifiers[columns[i]],bins = 20, color='blue', edgecolor='black')
  ax.set_title(dataset_anonymized_classifiers.head(0)[columns[i]].name)
plt.tight_layout()
plt.show()


# ## 2.6 División de Datos y Estandarización

# Y por último, se dividen los datos en conjuntos de entrenamiento y prueba, donde las características se estandarizan mediante la función StandardScaler(), de tal forma que se garantice que todas las características tengan media 0 y desviación estándar 1, algo muy importante en algoritmos sensibles a la escala de datos.



X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.25, random_state=1, stratify=y)




sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)




print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el dataset ha quedado dividido en un 25% de datos de prueba y un 75% de datos de entrenamiento. Es importante mencionar que la estandarización asegura que todas las características se encuentren en una escala comparable, algo esencial para algoritmos como Redes Neuronales o Máquinas de Soporte Vectorial (SVM, Support Vector Machines), lo que les permite mejorar la estabilidad y la precisión del modelo.

# # TEMA 3: Clasificadores

# ## 3.1 Regresión Logística (Logistic Regression)

# A continuación, se explican brevemente los parámetros utilizados:
# 
# * C = 100.0: El parámetro de regularización C controla la penalización aplicada a los errores. Un valor alto de C indica que se permite una menor penalización, por lo que el modelo intenta ajustar más los datos.
# 
# * solver = ’lbfgs’: El solver ’lbfgs’ es un optimizador recomendado para solucionar problemas pequeños y medianos.
# 
# * multi_class = ’ovr’: Procedente de las siglas OnevsRest (OVR), significa que para la clasificación multiclase, el modelo entrenara un clasificador independiente para cada clase.



lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % lr.score(X_test_std, y_test))

# CÁLCULO DE LA PRECISIÓN CON ACCURACY SCORE
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ÁNÁLISIS DE LOS RESULTADOS
# 
# El modelo de Regresión Logística (Logistic Regression) muestra una precisión del 58.8% (0.588) en el conjunto de prueba, con 165 muestras mal clasificadas. Esto significa que el modelo logra clasificar correctamente la mayoría de los casos, aunque aún persisten algunos errores.
# 
# Este modelo es útil para establecer una línea base, ya que es rápido y su interpretabilidad es alta, aunque puede darse el caso en el que no se capaz de captar relaciones complejas entre los datos.

# ## 3.2 Máquinas de Soporte Vectorial (SVM)

# A continuación, se explican brevemente los parámetros utilizados:
# 
# * kernel = ’rbf’: Se utiliza el kernel radial base (RBF), que es adecuado para problemas no lineales.
# 
# * gamma = 0.7: Controla el grado de influencia de los puntos individuales. Un valor bajo significa que el área de influencia de cada punto es alta, mientras que un valor alto restringe el área.
# 
# * C = 30.0: Controla el grado de penalización aplicado a los errores de clasificación. Un valor más alto de C tiende a reducir los errores de clasificación en el conjunto de entrenamiento.



svm = SVC(kernel='rbf', random_state=1, gamma=0.7, C=30.0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % svm.score(X_test_std, y_test))

# CÁLCULO DE LA PRECISIÓN CON ACCURACY SCORE
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ANÁLISIS DE LOS RESULTADOS
# 
# El modelo de Máquinas de Soporte Vectorial (SVM) alcanza una precisión del 60.8% (0.608) en el conjunto de prueba, con 157 muestras mal clasificadas, lo que hace ver que el modelo tiene un buen desempeño en la separación de clases del dataset.
# 
# Este modelo con núcleo RBF es eficaz en problemas de clasificación no lineales, aunque en ocasiones se puede requerir un ajuste de hiperparámetros para optimizar su rendimiento.

# ## 3.3 Árboles de Decisión (Decision Trees)

# A continuacion, se explican brevemente los parámetros utilizados:
# 
# * criterion = ’gini’: Utiliza el ı́ndice de Gini para medir la pureza de los nodos.
# 
# * max_depth = 4: La profundidad máxima del árbol se fija en 4 para evitar el sobreajuste.



tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))





# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
feature_names = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# feature_names = ['Col1', 'Col2', ..., 'ColN']

tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)
plt.show()
y_pred = tree_model.predict(X_test)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % tree_model.score(X_test, y_test))

# CÁLCULO DE LA PRECISIÓN CON ACCURACY SCORE
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ANÁLISIS DE LOS RESULTADOS
# 
# El modelo de Árboles de Decisión (Decision Trees) alcanza una precisión del 57.0% (0.570) en el conjunto de prueba, con 172 muestras mal clasificadas.
# 
# Este modelo proporciona una interpretación visual y clara del proceso de decisión y las variables importantes. Sin embargo, aunque no puede ser tan preciso como modelos más complejos, es valioso por su interpretabilidad y facilidad para detectar patrones simples en los datos.

# ## 3.4 Bosque Aleatorio (Random Forest)

# A continuación, se explican brevemente los parámetros utilizados:
# 
# * n_estimators = 25: Número de árboles en el bosque. Un mayor número generalmente mejora la precisión hasta cierto punto, pero incrementa el tiempo de cálculo.
# 
# * random_state = 1: Controla la aleatoriedad en la construcción de árboles, permitiendo reproducibilidad de los resultados al fijarse a un valor específico.
# 
# * n_jobs = 2: Número de núcleos de procesamiento que se utilizarán. Si se establece en -1, se usarán todos los núcleos disponibles, acelerando el entrenamiento en sistemas multiprocesador.




forest = RandomForestClassifier(n_estimators=25, 
                                random_state=1,
                                n_jobs=2)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % forest.score(X_test, y_test))

# CÁLCULO DE LA PRECISIÓN CON ACCURACY SCORE
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ANÁLISIS DE LOS RESULTADOS
# 
# El modelo de Bosque Aleatorio (Random Forest) alcanza una precisión del 66.7% (0.667) en el conjunto de prueba, con 133 muestras mal clasificadas.
# 
# Este modelo es robusto y suele ofrecer una alta precisión, dado que combina múltiples árboles y promedia sus predicciones, lo que reduce el riesgo de sobreajuste y mejora la generalización.

# ## 3.5 Vecinos Más Cercanos (KNN)

# A continuación, se explican brevemente los parámetros utilizados:
# 
# * n_neighbors = 2: Número de vecinos a considerar para clasificar una muestra. Un valor bajo puede hacer que el modelo sea sensible al ruido, mientras que un valor muy alto puede suavizar en exceso.
# 
# * p = 2: Parámetro de la distancia de Minkowski. Cuando p=2 se usa la distancia euclidiana; si p=1, se usa la distancia de Manhattan.
# 
# * metric = 'minkowski': La métrica de distancia utilizada para calcular la cercanía entre puntos. "minkowski" es una opción común, que permite ajustar la distancia con el parámetro p.




knn = KNeighborsClassifier(n_neighbors=2, 
                           p=2, 
                           metric='minkowski')

knn.fit(X_train_std, y_train)

y_pred = knn.predict(X_test_std)
print('Misclassification samples: %d' % (y_test != y_pred).sum())
print(y_test != y_pred)
print('Accuracy: %.3f' % knn.score(X_test_std, y_test))

# CÁLCULO DE LA PRECISIÓN CON ACCURACY SCORE
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# ANÁLISIS DE LOS RESULTADOS
# 
# El modelo de Vecinos Más Cercanos (KNN) alcanza una precisión del 57.5% (0.575) en el conjunto de prueba, con 170 muestras mal clasificadas.
# 
# Este modelo, aunque puede ofrecer un buen rendimiento, su precisión suele depender de la elección del número de vecinos y de una adecuada estandarización de los datos. Además, es útil en conjuntos de datos pequeños y bien distribuidos, aunque puede ser sensible a la escala de datos y al ruido.

# # TEMA 4: Análisis Exploratorio de Datos (Regresión)

# ## 4.1 Carga y Exploración Inicial del Dataset

# En primer lugar, se carga el dataset 'dataset.csv' y se configura la visualización de todas sus columnas. Además, se exploran las características más importantes del dataset como las columnas, su forma, una muestra de los primeros registros, así como un resumen de la información general y otras estadísticas descriptivas.



columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Target']

df = pd.read_csv('dataset.csv', sep=',',usecols=columns)

df.columns
df.shape
df.head(1)
df.info()
df.describe()


# ANÁLISIS DE LOS RESULTADOS
# 
# El dataset contiene un total de 12 columnas y 1599 filas. La exploración inicial revela la media, la desviación estándar, el mínimo, el máximo y los cuartiles de cada columna, lo que permite observar detalladamente la dispersión y la distribución de los datos, además de poder detectar problemas como valores nulos o distribuciones atípicas que puedan afectar al análsis posterior.

# ## 4.2 Anonimización y Cálculo de la Correlación entre Características

# Una vez realizado el primer análisis general de los datos, se elimina la columna 'Target' para crear un dataset anonimizado (dataset_anonymized), el cual se guarda en un fichero con el mismo nombre. Después, se calcula la matriz de correlación, que permite evaluar relaciones lineales entre las columnas.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS dataset_anonymized_regression
dataset_anonymized_regression = df.drop(["Target"], axis=1)
dataset_anonymized_regression.to_csv('dataset_anonymized_regression.csv', index=False)
dataset_anonymized_regression.corr()

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# dataset_anonymized_regression = df.drop(["Target"], axis=1)
# dataset_N_characteristics = dataset_anonymized_regression.drop(["Col1", "Col2", ..., "ColN"], axis=1)
# dataset_N_characteristics.to_csv('dataset_N_characteristics.csv', index=False)
# dataset_N_characteristics.corr()


# ## 4.3 División de Variables Independientes y Dependiente

# Después de anonimizar los datos, se separan las variables independientes (X) de la variable objetivo (y), en este caso 'Target', y se imprimen las etiquetas de clase únicas, las cuales permiten preparar el dataset para su uso en modelos de clasificación.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = dataset_anonymized_regression
y = df.get("Target")
print('Class labels:', np.unique(y))

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = dataset_N_characteristics
# y = df.get("Target")
# print('Class labels:', np.unique(y))


# ANÁLISIS DE LOS RESULTADOS
# 
# El dataset contiene 6 clases (3, 4, 5, 6, 7 y 8), lo que implica un problema de clasificación con múltiples etiquetas. El balance entre clases se verifica a través de la distribución de etiquetas, permitiendo conocer si existe un desbalance significativo que pudiera requerir estrategias adicionales de ajuste o balanceo en el modelo.

# ## 4.4 Mapa de Calor de Correlaciones

# Con las variables X e y ya definidas, se genera un mapa de calor que representa gráficamente a la matriz de correlación de todas las características del dataset, lo que facilita la detección visual de las relaciones entre variables.




cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)

plt.tight_layout()
plt.show()


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el mapa de calor muestra varias relaciones significativas entre algunas características.
# 
# Para otros apartados, habrá que seleccionar N de las mejores columnas entre todas las posibles, respectivamente. Para saber qué columnas poseen una mayor/mejor correlación, se seleccionan todos los valores de la fila 'Target' y pasarlos a valor absoluto, y una vez hecho esto, seleccionar los N valores más altos.
# 
# - Col1:   |-0.39|   =   0.39
# - Col2:   |-0.17|   =   0.17
# - Col3:   |-0.06|   =   0.06
# - Col4:   |-0.19|   =   0.19
# - Col5:   |0.25|    =   0.25
# - Col6:   |-0.13|   =   0.13
# - Col7:   |0.48|    =   0.48
# - Col8:   |-0.05|   =   0.05
# - Col9:   |0.12|    =   0.12
# - Col10:  |0.01|    =   0.01
# - Col11:  |0.23|    =   0.23

# ## 4.5 Histogramas de Distribución de las Características

# El siguiente bloque de código crea un histograma para cada una de las columnas que componen el dataset, donde se visualiza la distribución de cada característica en función de su frecuencia.




scatterplotmatrix(df.values, figsize=(12, 10), 
                  names=df.columns, alpha=0.5)

plt.tight_layout()
plt.show()

sb.pairplot(df)
plt.show()


# ## 4.6 División de Datos y Estandarización

# Y por último, se dividen los datos en conjuntos de entrenamiento y prueba, donde las características se estandarizan mediante la función StandardScaler(), de tal forma que se garantice que todas las características tengan media 0 y desviación estándar 1, algo muy importante en algoritmos sensibles a la escala de datos.




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


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el dataset ha quedado dividido en un 30% de datos de prueba y un 70% de datos de entrenamiento. Es importante mencionar que la estandarización asegura que todas las características se encuentren en una escala comparable, algo esencial para algoritmos como Redes Neuronales o Máquinas de Soporte Vectorial (SVM, Support Vector Machines), lo que les permite mejorar la estabilidad y la precisión del modelo.

# ## 4.7 Error Cuadrático Medio (MSE)

# Este snippet de código calcula el Error Cuadrático Medio (Mean Squared Error) para los conjuntos de entrenamiento y de prueba, permitiendo ayudar a medir el rendimiento del modelo en términos de precisión.




mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, un MSE bajo en ambos conjuntos, tanto en el de entrenamiento como en el de prueba indican un buen ajuste del modelo. Sin embargo, si el MSE en el conjunto de prueba es significativamente mayor que en el de entrenamiento, se tendría un indicador de sobreajuste.
# 
# En este caso, los valores del MSE obtenidos son de 0.41 para el conjunto de entrenamiento y de 0.43 para el conjunto de prueba.

# ## 4.8 Error Absoluto Medio (MAE)

# Este snippet de código calcula el Error Absoluto Medio (Mean Absolute Error), encargado de medir la media de las diferencias absolutas entre predicciones y valores reales.




mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')


# ANÁLISIS DE LOS RESULTADOS
# 
# El MAE ofrece una interpretación directa de la media del error en las predicciones. Al igual que con el MSE, un MAE menor en el conjunto de prueba que en el de entrenamiento, sugiere que el modelo realiza una buena generalización. Sin embargo, si se tienen valores altos, esto indicaría la necesidad de mejorar el ajuste del modelo.
# 
# En este caso, los valores del MAE obtenidos son de 0.50, tanto para el conjunto de entrenamiento como para el conjunto de prueba.

# ## 4.9 Coeficiente de Determinación (R²)

# Este snippet de código calcula el Coeficiente de Determinación (R²) tanto para el conjunto de entrenamiento como el conjunto de prueba, el cual proporciona una métrica de ajuste del modelo.




r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')


# ANÁLISIS DE LOS RESULTADOS
# 
# El valor de R² indica la proporción de la variación en y explicada por X, por lo que un valor cercano a 1 para los conjuntos de entrenamiento y prueba señalan un buen ajuste. Sin embargo, una gran diferencia entre el R² de entrenamiento y de prueba podría sugerir un caso de sobreajuste.
# 
# En este caso, los valores de R² obtenidos son de 0.39 para el conjunto de entrenamiento y de 0.28 para el conjunto de prueba.

# # TEMA 5: Métodos de Regresión

# ## 5.1 Regresión Lineal (Linear Regression)

# En este apartado se implementa un modelo de Regresión Lineal (Linear Regression).
# 
# En primer lugar, se divide el conjunto de datos en entrenamiento y prueba, donde el modelo se ajusta a los datos de entrenamiento en función de la columna 'Target'.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = ['Col1', 'Col2', ..., 'ColN'].values
# y = df['Target'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


slr = LinearRegression()

slr.fit(X_train, y_train)


# Y por último, se calcula el Error Absoluto Medio (MAE), el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²) para poder evaluar el rendimiento del modelo.



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


# ANÁLISIS DE LOS RESULTADOS
# 
# * MAE: Un MAE de 0.5026385105672768 indica que, en promedio, las predicciones se desvían de los valores reales en aproximadamente 0.5026385105672768 unidades, lo que además sugiere la precisión general del modelo.
# 
# * MSE: Un MSE de 0.43455004866201363 indica que el modelo muestra una desviación cuadrática promedio de 0.43455004866201363 unidades respecto a los valores reales, lo que refleja un nivel de error mayor en predicciones extremas.
# 
# * R²: Un R² de 0.2793728634848818 indica que el modelo explica aproximadamente el 27.93728634848818% de la variabilidad en la columna 'Target'.

# ## 5.2 Regresión Cuadrática (Quadratic Regression)

# En este apartado se implementa un modelo de Regresión Polinómica Cuadrática (Quadratic Regression).
# 
# En primer lugar, se transforma el conjunto de características para incluir términos cuadráticos y se entrena un modelo de regresión lineal en el espacio transformado.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = ['Col1', 'Col2', ..., 'ColN'].values
# y = df['Target'].values



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


# Y por último, se calcula el Error Absoluto Medio (MAE), el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²) para poder evaluar la precisión del modelo cuadrático.



X_test_quadratic = quadratic.fit_transform(X_test)

y_pred_quadratic = regr.predict(X_test_quadratic)

mae = mean_absolute_error(y_test, y_pred_quadratic)
mse = mean_squared_error(y_test, y_pred_quadratic)
R2 = r2_score(y_test, y_pred_quadratic)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# ANÁLISIS DE LOS RESULTADOS
# 
# * MAE: un MAE de 0.513535914951333 indica que el modelo cuadrático mejora la precisión en comparación con el modelo lineal si el valor es menor, ya que captura relaciones no lineales en los datos.
# 
# * MSE: Un MSE de 0.4462239183173426 representa la penalización por errores más grandes. Si el valor obtenido en este caso es menor que el obtenido en el modelo lineal, se puede suponer que el modelo cuadrático estaría capturando las variaciones mejor que el modelo lineal.
# 
# * R²: un R² de 0.2600137418194426 indica que el modelo cuadrático abarca un 26.00137418194426% de la variabilidad de la columna 'Target'. Si el valor de R² obtenido en este caso es mayor que el obtenido en el modelo lineal, se considera más adecuado el modelo cuadrático.

# ## 5.3 Regresión Cúbica (Cubic Regression)

# En este apartado se implementa un modelo de Regresión Polinómica Cúbica (Cubic Regression).






# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = ['Col1', 'Col2', ..., 'ColN'].values
# y = df['Target'].values

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





mae = mean_absolute_error(y_test, y_pred_cubic)
mse = mean_squared_error(y_test, y_pred_cubic)
R2 = r2_score(y_test, y_pred_cubic)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", R2)


# ANÁLISIS DE LOS RESULTADOS
# 
# * MAE: Un MAE de 0.7091287257843457 indica que el modelo cúbico reduce el error promedio en comparación con modelos de menor grado, en caso de que el MAE obtenido en este caso sea menor que el obtenidos en los otros modelos de menor grado.
# 
# * MSE: Un MSE de 2.0499047678552964 indica un ajuste más preciso para relaciones más complejas en los datos. Sin embargo, si este valor es inferior al obtenido en modelos anteriores, se refleja una mejora.
# 
# * R²: Un R² de -2.3994174147181084 indica que el modelo podría capturar relaciones más intrincadas en los datos, pero si el valor obtenido en este modelo es menor que el obtenido en los otros modelos, podría tenerse un signo claro de sobreajuste.

# ## 5.4 Regresión con Árboles de Decisión (Decision Tree Regression)

# En este apartado se implementa un modelo de Regresión con Árboles de Decisión cuya profundidad está limitada a 3 para poder predecir mejor 'Target' (Decision Tree Regression).





# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = ['Col1', 'Col2', ..., 'ColN'].values
# y = df['Target'].values

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


# ANÁLISIS DE LOS RESULTADOS
# 
# * MAE: Un MAE de 0.5435151604013354 indica la precisión del modelo en las predicciones. Si se obtiene un valor relativamente bajo, se supone que el árbol captura correctamente patrones en los datos.
# 
# * MSE: Un MSE de 0.4882864235878708 refleja el Error Cuadrático Medio. Si el valor obtenido es mayor en comparación con otros modelos, se puede suponer que el árbol de decisión pueda estar limitado por la profundidad impuesta (en este caso, 3).
# 
# * R²: Un R² de 0.19026025123514279 indica que el modelo abarca un 19.026025123514279% de la variabilidad en 'Target'. Si el valor de R² obtenido es bajo, se supone que se necesita un valor mayor de la profundidad para mejorar la captura de relaciones complejas.

# ## 5.5 Regresión con Bosque Aleatorio (Random Forest Regression)

# En este apartado se implementa un modelo de Regresión con Bosque Aleatorio en el que se utilizan 1000 árboles (Random Forest Regression).





# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11']].values
y = df['Target'].values

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = ['Col1', 'Col2', ..., 'ColN'].values
# y = df['Target'].values

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


# ANÁLISIS DE LOS RESULTADOS
# 
# * MAE:Un MAE de 0.42310625 indica que el modelo de Bosque Aleatorio logra ser preciso en sus predicciones, reflejando su robustez a la hora de capturar patrones complejos.
# 
# * MSE: Un MSE de 0.3368855354166667 indica una mejora frente a otros modelos, en caso de que el valor obtenido sea menor que el de los tros modelos, por lo que los errores son menos penalizados.
# 
# * R²: Un R² de 0.4413328005182279 indica que el modelo abarca un 44.13328005182279% de la variabilidad en 'Target'. Este valor suele ser alto en Bosques Aleatorios, lo que le permite representar una mejora frente a otros métodos, ya que los Bosques Aleatorios manejan la complejidad mejor que los modelos polinómicos.

# # ANEXO: Convertir Jupyter Notebook a Fichero Python

# ## A.1 Script en el Directorio Actual



# Script 'convert_notebook_to_script.py' en el directorio actual


# ## A.2 Script en el Directorio Padre



# Script 'convert_notebook_to_script.py' en el directorio padre

