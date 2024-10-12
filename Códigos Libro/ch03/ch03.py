# coding: utf-8


import sys
# * from python_environment_check import check_packages
from python_environment_check import check_packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# # Chapter 3 - A Tour of Machine Learning Classifiers Using Scikit-Learn

# ### Overview

# - [Choosing a classification algorithm](#Choosing-a-classification-algorithm)
# - [First steps with scikit-learn](#First-steps-with-scikit-learn)
#     - [Training a perceptron via scikit-learn](#Training-a-perceptron-via-scikit-learn)
# - [Modeling class probabilities via logistic regression](#Modeling-class-probabilities-via-logistic-regression)
#     - [Logistic regression intuition and conditional probabilities](#Logistic-regression-intuition-and-conditional-probabilities)
#     - [Learning the weights of the logistic loss function](#Learning-the-weights-of-the-logistic-loss-function)
#     - [Training a logistic regression model with scikit-learn](#Training-a-logistic-regression-model-with-scikit-learn)
#     - [Tackling overfitting via regularization](#Tackling-overfitting-via-regularization)
# - [Maximum margin classification with support vector machines](#Maximum-margin-classification-with-support-vector-machines)
#     - [Maximum margin intuition](#Maximum-margin-intuition)
#     - [Dealing with the nonlinearly separable case using slack variables](#Dealing-with-the-nonlinearly-separable-case-using-slack-variables)
#     - [Alternative implementations in scikit-learn](#Alternative-implementations-in-scikit-learn)
# - [Solving nonlinear problems using a kernel SVM](#Solving-nonlinear-problems-using-a-kernel-SVM)
#     - [Using the kernel trick to find separating hyperplanes in higher dimensional space](#Using-the-kernel-trick-to-find-separating-hyperplanes-in-higher-dimensional-space)
# - [Decision tree learning](#Decision-tree-learning)
#     - [Maximizing information gain – getting the most bang for the buck](#Maximizing-information-gain-–-getting-the-most-bang-for-the-buck)
#     - [Building a decision tree](#Building-a-decision-tree)
#     - [Combining weak to strong learners via random forests](#Combining-weak-to-strong-learners-via-random-forests)
# - [K-nearest neighbors – a lazy learning algorithm](#K-nearest-neighbors-–-a-lazy-learning-algorithm)
# - [Summary](#Summary)



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



# # Choosing a classification algorithm

# # First steps with scikit-learn

# Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower examples. The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.



# IMPORTACIÓN DE BIBLIOTECAS
# * sklearn.datasets
# Se importa el módulo datasets de scikit-learn, que contiene varios conjuntos de datos clásicos, 
# como el Iris, que es utilizado en ejemplos de clasificación.
# * numpy (np)
# Se importa la biblioteca numpy bajo el alias np para manejar operaciones numéricas.

# CARGA DEL CONJUNTO DE DATOS IRIS
# * load_iris() 
# Carga el conjunto de datos Iris, un dataset clásico que contiene características de tres tipos 
# de flores de iris: Setosa, Versicolor, y Virginica. El conjunto de datos incluye 150 
# ejemplos, con 4 características por ejemplo (longitud y anchura de sépalo y pétalo), y las 
# etiquetas de las clases correspondientes a las especies.
iris = datasets.load_iris()

# SELECCIÓN DE CARACTERÍSTICAS
# * iris.data
# Contiene una matriz de 150 filas y 4 columnas, donde cada fila es un ejemplo con sus 4 
# características.
# * [:, [2, 3]]
# Se seleccionan únicamente las columnas 2 y 3, que corresponden a la longitud y anchura
# del pétalo, respectivamente. Así, X contiene solo estas dos características.
X = iris.data[:, [2, 3]]

# ASIGNACIÓN DE ETIQUETAS DE CLASE
# * iris.target
# Contiene las etiquetas de clase para cada ejemplo en el conjunto de datos. Las etiquetas 
# son 0, 1, y 2, que corresponden a las tres especies de flores de iris (Setosa, Versicolor, 
# y Virginica).
y = iris.target

# IMPRESIÓN DE LAS ETIQUETAS DE CLASE ÚNICAS
# * np.unique(y)
# Esta función de NumPy devuelve los valores únicos de las etiquetas en y. En este caso, 
# imprimirá las etiquetas de clase únicas: [0, 1, 2], que representan las tres especies de 
# flores en el dataset Iris.
print('Class labels:', np.unique(y))


# Splitting data into 70% training and 30% test data:



# IMPORTACIÓN DEL MÓDULO
# Se importa la función train_test_split del módulo model_selection de scikit-learn. 
# Esta función se utiliza para dividir un conjunto de datos en dos subconjuntos: 
# uno para entrenamiento y otro para prueba.

# DIVISIÓN DE LOS DATOS
# * X
# Son las características (variables independientes) del conjunto de datos.
# * y
# Son las etiquetas (la variable dependiente o de clase).
# La función train_test_split divide estos datos en cuatro subconjuntos:
# * X_train
# Conjunto de características para entrenamiento.
# * X_test
# Conjunto de características para prueba.
# * y_train
# Etiquetas correspondientes al conjunto de entrenamiento.
# * y_test
# Etiquetas correspondientes al conjunto de prueba.

# PARÁMETROS
# * test_size=0.3
# Indica que el 30% de los datos (0.3) se reservará para el conjunto de prueba, y el 70% 
# restante se usará para entrenamiento.
# * random_state=1
# Establece una semilla aleatoria para que la división sea reproducible, es decir, al usar 
# la misma semilla (1), se obtendrá la misma división cada vez que se ejecute.
# * stratify=y
# Garantiza que la proporción de clases en y (las etiquetas) sea la misma en ambos conjuntos 
# (entrenamiento y prueba). Esto es útil cuando las clases están desbalanceadas.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)




# IMPRESIÓN DEL CONTEO DE ETIQUETAS EN Y
# * np.bincount(y)
# Esta función cuenta la cantidad de veces que aparece cada valor entero en el array y. 
# En este caso, y contiene las etiquetas de clase (0, 1, 2) del conjunto de datos completo.
# El resultado es una lista en la que cada posición corresponde a una etiqueta, y el valor en 
# esa posición indica cuántas veces aparece esa etiqueta en y.
# * Ejemplo
# Si y tiene 50 ejemplos de la clase 0, 50 de la clase 1, y 50 de la clase 2, el resultado 
# será algo como [50, 50, 50].
print('Labels counts in y:', np.bincount(y))

# IMPRESIÓN DEL CONTEO DE ETIQUETAS EN Y_TRAIN
# Similar al caso anterior, esta línea cuenta las ocurrencias de cada etiqueta en el conjunto 
# de entrenamiento (y_train), que se obtuvo después de dividir los datos con train_test_split.
# Como se usó el parámetro stratify al dividir los datos, el conteo de etiquetas en y_train 
# mantendrá las proporciones de las clases originales.
print('Labels counts in y_train:', np.bincount(y_train))

# IMPRESIÓN DEL CONTEO DE ETIQUETAS EN Y_TEST
# Aquí se cuenta cuántas veces aparece cada etiqueta en el conjunto de prueba (y_test).
# Nuevamente, las proporciones de las clases en y_test serán las mismas que en los datos 
# originales debido al uso de stratify.
print('Labels counts in y_test:', np.bincount(y_test))


# Standardizing the features:



# IMPORTACIÓN DEL MÓDULO
# Se importa la clase StandardScaler del módulo sklearn.preprocessing. Esta clase se utiliza 
# para estandarizar características, es decir, hacer que los datos tengan una media de 0 y una 
# desviación estándar de 1.

# CREACIÓN DE UN OBJETO STANDARD SCALER
# Se crea una instancia de StandardScaler llamada sc. Este objeto se usará para ajustar 
# (calcular los parámetros) y transformar los datos.
sc = StandardScaler()

# AJUSTE DEL ESCALADOR EN EL CONJUNTO DE ENTRENAMIENTO
# El método fit(X_train) ajusta el escalador usando los datos de entrenamiento (X_train). 
# Durante este proceso, el StandardScaler calcula la media y la desviación estándar de cada 
# característica en X_train. Estos valores se almacenan y se usarán posteriormente para 
# transformar tanto los datos de entrenamiento como los de prueba.
sc.fit(X_train)

# TRANSFORMACIÓN DEL CONJUNTO DE ENTRENAMIENTO
# El método transform(X_train) utiliza los valores de la media y desviación estándar 
# calculados en el paso anterior para transformar las características de X_train. Cada valor 
# de las características de X_train se estandariza restando la media y dividiendo por la 
# desviación estándar. El resultado es un nuevo conjunto de datos X_train_std, donde las 
# características tienen una media de 0 y una desviación estándar de 1.
X_train_std = sc.transform(X_train)

# TRANSFORMACIÓN DEL CONJUNTO DE PRUEBA
# Se aplica la misma transformación a los datos de prueba (X_test) utilizando los parámetros 
# de estandarización (media y desviación estándar) calculados a partir del conjunto de 
# entrenamiento. Esto garantiza que las transformaciones en los datos de prueba sean 
# consistentes con las del conjunto de entrenamiento. El resultado es el conjunto estandarizado 
# X_test_std.
X_test_std = sc.transform(X_test)


# ## Training a perceptron via scikit-learn



# IMPORTACIÓN DE PERCEPTRÓN
# Se importa la clase Perceptron del módulo sklearn.linear_model. El Perceptrón es un algoritmo 
# de clasificación lineal que actualiza los pesos mediante la regla de aprendizaje del perceptrón.

# CREACIÓN DE UN MODELO PERCEPTRÓN
# Se crea una instancia del modelo Perceptrón con los siguientes parámetros:
# * eta0=0.1
# Tasa de aprendizaje, que controla cuánto se ajustan los pesos en cada iteración.
# * random_state=1
# Establece una semilla aleatoria para asegurar la reproducibilidad de los resultados.
ppn = Perceptron(eta0=0.1, random_state=1)

# ENTRENAMIENTO DEL MODELO
# El modelo ppn se entrena con el conjunto de datos estandarizados de entrenamiento 
# (X_train_std) y sus etiquetas correspondientes (y_train).
# El método fit ajusta el modelo calculando los pesos que mejor separan las clases en los 
# datos de entrenamiento.
ppn.fit(X_train_std, y_train)




# PREDICCIÓN SOBRE EL CONJUNTO DE PRUEBA
# Se utiliza el modelo entrenado para hacer predicciones sobre el conjunto de prueba 
# estandarizado (X_test_std), generando las predicciones de clase y_pred.
y_pred = ppn.predict(X_test_std)

# CÁLCULO DE EJEMPLOS MAL CLASIFICADOS
# Se compara las etiquetas reales (y_test) con las predicciones del modelo (y_pred) para contar 
# cuántos ejemplos fueron mal clasificados. Esto se logra con la expresión 
# (y_test != y_pred).sum(), que cuenta cuántos valores son diferentes entre y_test y y_pred.
print('Misclassified examples: %d' % (y_test != y_pred).sum())




# CÁLCULO DE LA PRECISIÓN (ACCURACY)
# Se importa la función accuracy_score del módulo sklearn.metrics para calcular la precisión 
# (el porcentaje de ejemplos correctamente clasificados) comparando y_test con y_pred.
# La precisión se imprime con tres decimales usando la expresión '%.3f'.
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))




# PRECISIÓN UTILIZANDO EL MÉTODO SCORE DEL PERCEPTRÓN
# Alternativamente, se calcula la precisión usando el método score del Perceptrón (ppn), que 
# también devuelve la precisión del modelo sobre el conjunto de prueba (X_test_std y y_test).
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))




# IMPORTACIÓN DE LAS BIBLIOTECAS
# * ListedColormap
# Permite crear un mapa de colores personalizado para la visualización.
# * matplotlib.pyplot
# Proporciona funciones para generar gráficos.
# * numpy
# Es la biblioteca para realizar cálculos numéricos.
# * LooseVersion
# Se importa para verificar la compatibilidad de versiones de matplotlib 
# (aunque no se utiliza en este fragmento).

# DEFINICIÓN DE LA FUNCIÓN PLOT_DECISION_REGIONS
# * X
# El conjunto de datos con dos características (columnas) a visualizar.
# * y
# Las etiquetas de clase asociadas a los datos.
# * classifier
# El modelo de clasificación que se utilizará para predecir las clases.
# * test_idx
# Índices opcionales para resaltar los ejemplos del conjunto de prueba.
# * resolution
# La resolución de la malla que se dibujará para las regiones de decisión.
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # CONFIGURACIÓN DE LOS COLORES Y MARCADORES
    # Se definen varios tipos de marcadores (o, s, ^, etc.) y colores para cada clase.
    # El mapa de colores cmap se genera para que cada clase tenga un color único, en 
    # función del número de clases en los datos (np.unique(y)).
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # GENERACIÓN DE LA MALLA DE DECISIÓN
    # Se calcula el rango de valores de las dos características (X[:, 0] y X[:, 1]) para 
    # crear una malla (grid) de puntos que cubra todo el espacio bidimensional. Los 
    # límites se amplían en 1 unidad para que no se recorten los datos.
    # Se utiliza np.meshgrid para crear matrices de coordenadas que cubren el espacio.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # PREDICCIÓN SOBRE LA MALLA
    # Se aplican las predicciones del clasificador sobre cada punto de la malla generada.
    # Los resultados se transforman y reorganizan para coincidir con la forma de la malla, 
    # permitiendo crear un gráfico continuo de las regiones de decisión.
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    # DIBUJO DE LAS REGIONES DE DECISIÓN
    # * contourf
    # Se utiliza para dibujar las regiones de decisión, coloreando cada región según las 
    # predicciones del clasificador.
    # Se ajustan los límites de los ejes para que coincidan con el rango de valores de las 
    # características.
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # VISUALIZACIÓN DE LOS PUNTOS DE DATOS
    # Se dibujan los puntos de datos originales sobre las regiones de decisión. Cada clase 
    # tiene un color y marcador diferentes, y los puntos tienen un borde negro para resaltarlos.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # RESALTADO DE LOS EJEMPLOS DEL CONJUNTO DE PRUEBA
    # Si se proporcionan índices de los ejemplos de prueba (test_idx), estos puntos se resaltan 
    # en el gráfico usando un círculo grande con borde negro y sin color de relleno (c='none').
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Test set')        


# Training a perceptron model using the standardized training data:



# COMBINACIÓN DE CONJUNTOS DE DATOS
# * np.vstack
# Se utiliza para apilar verticalmente los conjuntos de datos estandarizados de entrenamiento 
# (X_train_std) y prueba (X_test_std). Esto genera una única matriz X_combined_std que contiene 
# todos los datos estandarizados.
# * np.hstack
# Se utiliza para apilar horizontalmente las etiquetas de las clases del conjunto de 
# entrenamiento (y_train) y del conjunto de prueba (y_test). Esto crea un solo array y_combined 
# que contiene todas las etiquetas de clase.
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions (definida previamente) para visualizar las 
# regiones de decisión del modelo Perceptrón (ppn).
# Los parámetros pasados son:
# * X=X_combined_std
# Los datos estandarizados combinados.
# * y=y_combined
# Las etiquetas combinadas.
# * classifier=ppn
# El clasificador utilizado para predecir las clases.
# * test_idx=range(105, 150)
# Indica qué ejemplos del conjunto de prueba deben ser resaltados en la visualización. 
# Aquí se están resaltando los índices 105 a 149.
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

# ETIQUETADO DE LOS EJES
# Se añaden etiquetas a los ejes X e Y del gráfico, que representan las características de 
# longitud y ancho de los pétalos estandarizados.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')

# LEYENDA
# Se añade una leyenda al gráfico en la esquina superior izquierda para identificar las 
# diferentes clases.
plt.legend(loc='upper left')

# AJUSTES DEL DISEÑO
# Esta función ajusta automáticamente el espaciado del gráfico para que los elementos no se 
# superpongan y se vean más ordenados.
plt.tight_layout()

# MOSTRAR EL GRÁFICO
# La línea que guarda el gráfico como un archivo PNG está comentada. Si se descomenta, el 
# gráfico se guardaría en la ruta especificada con una resolución de 300 dpi.
# * plt.show()
# Muestra el gráfico generado en una ventana emergente.
# plt.savefig('figures/03_01.png', dpi=300)
plt.show()


# # Modeling class probabilities via logistic regression

# ### Logistic regression intuition and conditional probabilities



# IMPORTACIÓN DE BIBLIOTECAS
# * matplotlib.pyplot
# Se utiliza para crear gráficos en Python.
# * numpy
# Se emplea para realizar cálculos numéricos y manipular arreglos.

# DEFINICIÓN DE LA FUNCIÓN SIGMOIDE
# Se define la función sigmoid, que toma un valor (o un arreglo de valores) z y aplica la 
# fórmula de la función sigmoide. La salida está en el rango de (0, 1), lo que la hace útil 
# para interpretar como probabilidades.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# GENERACIÓN DE VALORES PARA Z
# Se crea un arreglo z que contiene valores desde -7 hasta 7 (sin incluir 7), con un paso de 
# 0.1. Este rango cubre la mayoría del comportamiento interesante de la función sigmoide.
z = np.arange(-7, 7, 0.1)

# CÁLCULO DE LA FUNCIÓN SIGMOIDE
# Se aplica la función sigmoide a todos los valores de z y se almacena el resultado en sigma_z.
sigma_z = sigmoid(z)

# CREACIÓN DEL GRÁFICO
# * plt.plot(z, sigma_z)
# Se traza el gráfico de la función sigmoide con z en el eje X y sigma_z en el eje Y.
# * plt.axvline(0.0, color='k')
# Se dibuja una línea vertical en z = 0 (en color negro), lo que ayuda a visualizar el punto 
# en el que la función sigmoide es 0.5.
# * plt.ylim(-0.1, 1.1)
# Se establecen los límites del eje Y para que el gráfico se vea más limpio.
# * plt.xlabel('z') y plt.ylabel('$\sigma (z)$')
# Se añaden etiquetas a los ejes X e Y, respectivamente. La etiqueta del eje Y usa notación 
# matemática para mostrar la función sigmoide.
plt.plot(z, sigma_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')

# CONFIGURACIÓN DE LAS MARCAS Y LÍNEAS DE LA CUADRÍCULA
# * plt.yticks([0.0, 0.5, 1.0])
# Se configuran las marcas del eje Y para que solo muestre 0.0, 0.5 y 1.0.
# ax.yaxis.grid(True)
# Se activan las líneas de la cuadrícula en el eje Y para mejorar la legibilidad del gráfico.
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

# AJUSTE DEL DISEÑO Y VISUALIZACIÓN
# * plt.tight_layout()
# Ajusta el diseño del gráfico para que no haya superposiciones.
# La línea que guarda el gráfico como un archivo PNG está comentada. Si se descomenta, 
# el gráfico se guardaría en la ruta especificada con una resolución de 300 dpi.
# * plt.show()
# Muestra el gráfico generado en una ventana emergente.
plt.tight_layout()
# plt.savefig('figures/03_02.png', dpi=300)
plt.show()




# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_03.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_03.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_25.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_25.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ### Learning the weights of the logistic loss function



# DEFINICIÓN DE FUNCIONES DE PÉRDIDA
# * loss_1(z)
# Calcula la función de pérdida cuando la etiqueta de clase es 1. Utiliza la función sigmoide y la fórmula de pérdida de entropía cruzada para clasificaciones positivas.
# * loss_0(z)
# Calcula la función de pérdida cuando la etiqueta de clase es 0. Similarmente, usa la fórmula de pérdida de entropía cruzada para clasificaciones negativas.
def loss_1(z):
    return - np.log(sigmoid(z))
def loss_0(z):
    return - np.log(1 - sigmoid(z))

# GENERACIÓN DE VALORES PARA Z
# Se crea un arreglo z que contiene valores desde -10 hasta 10 (sin incluir 10), con un paso de 
# 0.1. Esto permite observar cómo se comportan las funciones de pérdida en un rango amplio de 
# valores.
# Se calcula sigma_z, que representa los valores de la función sigmoide evaluados en el rango 
# de z.
z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)

# CÁLCULO DE LA PÉRDIDA PARA CUANDO Y=1
# Se utiliza una comprensión de lista para calcular la pérdida usando loss_1 para cada valor en 
# z. El resultado se almacena en c1.
# Luego, se grafica c1 en función de sigma_z, etiquetando la línea como L(w, b) if y=1.
c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='L(w, b) if y=1')

# CÁLCULO DE LA PÉRDIDA PARA CUANDO Y=0
# Similar al paso anterior, se calcula la pérdida usando loss_0 para cada valor en z y se 
# almacena en c0.
# Se grafica c0 en función de sigma_z, usando un estilo de línea discontinua (linestyle='--') 
# y etiquetándola como L(w, b) if y=0.
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')

# CONFIGURACIÓN DE LOS LÍMITES Y ETIQUETAS DEL GRÁFICO
# * plt.ylim(0.0, 5.1)
# Establece el límite del eje Y entre 0 y 5.1 para centrarse en las pérdidas calculadas.
# * plt.xlim([0, 1])
# Establece el límite del eje X entre 0 y 1, que es el rango de la función sigmoide.
# Se añaden etiquetas a los ejes X e Y, mostrando que el eje X representa la salida de la 
# función sigmoide y el eje Y representa la pérdida.
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')

# ADICIÓN DE LEYENDA Y AJUSTE DEL DISEÑO
# * plt.legend(loc='best')
# Añade una leyenda al gráfico en la mejor posición disponible.
# * plt.tight_layout()
# Ajusta automáticamente el diseño del gráfico para que no haya superposiciones.
# La línea que guarda el gráfico como un archivo PNG está comentada. Si se descomenta, 
# el gráfico se guardaría en la ruta especificada con una resolución de 300 dpi.
# * plt.show()
# Muestra el gráfico generado en una ventana emergente.
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('figures/03_04.png', dpi=300)
plt.show()




# DEFINICIÓN DE LA CLASE LOGISTICREGRESSIONGD
# Esta clase implementa un clasificador de regresión logística basado en el algoritmo de 
# descenso de gradiente. La docstring proporciona detalles sobre los parámetros y atributos 
# de la clase.
class LogisticRegressionGD:

    # PARÁMETROS DE INICIALIZACIÓN
    # * eta
    # Tasa de aprendizaje que determina el tamaño del paso en cada iteración del descenso 
    # de gradiente.
    # * n_iter
    # Número de pasadas (épocas) sobre el conjunto de entrenamiento.
    # * random_state
    # Semilla para el generador de números aleatorios, que se utiliza para 
    # la inicialización de pesos.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # MÉTODO DE AJUSTE
    # Este método ajusta el modelo a los datos de entrenamiento.
    # - X: Matriz de características de entrada (n_examples, n_features).
    # - y: Vector de etiquetas objetivo (n_examples).
    # Dentro del método fit:
    # - Inicializa los pesos y el sesgo con valores aleatorios pequeños.
    # - Crea una lista para almacenar la pérdida durante el entrenamiento.
    # Para cada iteración:
    # - Calcula la entrada neta (net_input).
    # - Aplica la función sigmoide para obtener la salida del modelo (activation).
    # - Calcula los errores (diferencia entre la salida esperada y la salida del modelo).
    # - Actualiza los pesos y el sesgo usando la regla de actualización del descenso de gradiente.
    # - Calcula la pérdida utilizando la función de pérdida logarítmica (log loss) y la almacena
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self

    # MÉTODO DE ENTRADA NETA
    # Calcula la entrada neta del modelo, que es el producto punto de las características de 
    # entrada con los pesos más el sesgo.
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # FUNCIÓN DE ACTIVACIÓN
    # Aplica la función sigmoide a la entrada neta, lo que transforma los valores en un rango 
    # entre 0 y 1. Utiliza np.clip para evitar desbordamientos numéricos.
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    # MÉTODO DE PREDICCIÓN
    # Devuelve la etiqueta de clase (0 o 1) para las muestras de entrada. Si la activación es 
    # mayor o igual a 0.5, devuelve 1; de lo contrario, devuelve 0.
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




# FILTRADO DE DATOS
# * X_train_01_subset
# Crea un subconjunto de las características de entrenamiento (X_train_std) que incluye solo 
# las muestras donde la etiqueta (y_train) es 0 o 1. Esto se hace para simplificar el problema 
# de clasificación a un caso binario, eliminando las clases adicionales.
# * y_train_01_subset
# Similarmente, crea un vector de etiquetas que contiene solo las etiquetas correspondientes 
# a las clases 0 y 1.
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# INSTANCIACIÓN Y ENTRENAMIENTO DEL MODELO
# * lrgd
# Se instancia un objeto de la clase LogisticRegressionGD con una tasa de aprendizaje (eta) 
# de 0.3 y un número de iteraciones (n_iter) de 1000. La random_state se establece en 1 para 
# garantizar la reproducibilidad de los resultados.
# * fit
# El método fit se llama para ajustar el modelo a los datos de entrenamiento del subconjunto 
# binario (0 y 1).
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

# VISUALIZACIÓN DE REGIONES DE DECISIÓN
# * plot_decision_regions
# Esta función visualiza las regiones de decisión del clasificador entrenado. Muestra cómo el 
# modelo divide el espacio de características entre las clases 0 y 1.
plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

# ETIQUETAS, VISUALIZACIÓN Y LEYENDA
# Se añaden etiquetas a los ejes X e Y para indicar que representan la longitud y el ancho de 
# los pétalos, respectivamente. Se incluye una leyenda en la esquina superior izquierda.
# * tight_layout
# Ajusta el diseño para que no se solapen elementos de la gráfica.
# * show
# Muestra la gráfica generada.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_05.png', dpi=300)
plt.show()


# ### Training a logistic regression model with scikit-learn



# IMPORTACIÓN DEL MODELO
# Se importa la clase LogisticRegression de scikit-learn, que permite crear un clasificador 
# basado en regresión logística.

# INSTANCIACIÓN DEL CLASIFICADOR
# * C=100.0
# Este parámetro inverso de regularización controla la cantidad de regularización aplicada 
# al modelo. Un valor alto como 100.0 significa poca regularización, lo que puede permitir 
# que el modelo se ajuste más a los datos de entrenamiento.
# * solver='lbfgs'
# Especifica el algoritmo a utilizar para optimizar la función de pérdida. lbfgs es un método 
# que utiliza aproximaciones de tipo cuasi-Newton.
# * multi_class='ovr'
# Establece el enfoque para manejar problemas de clasificación multiclase. El método 'ovr' 
# (One-vs-Rest) entrena un clasificador por cada clase contra las demás.
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')

# ENTRENAMIENTO DEL MODELO
# Se entrena el modelo lr utilizando el conjunto de datos X_train_std (características 
# estandarizadas) y y_train (etiquetas de clase). El modelo ajusta los parámetros internos 
# para minimizar la función de pérdida en el conjunto de entrenamiento.
lr.fit(X_train_std, y_train)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions, que visualiza las regiones de decisión 
# generadas por el modelo entrenado lr. Esto muestra cómo el modelo clasifica el espacio de 
# características basado en las clases en y_combined.
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))

# ETIQUETAS, VISUALIZACIÓN Y LEYENDA
# Se añaden etiquetas a los ejes X e Y para indicar que representan la longitud y el ancho 
# de los pétalos estandarizados.
# Se incluye una leyenda en la esquina superior izquierda.
# * tight_layout
# Ajusta el diseño para que no se solapen elementos de la gráfica.
# * show
# Muestra la gráfica generada.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_06.png', dpi=300)
plt.show()




# DESCRIPCIÓN
# Esta línea llama al método predict_proba del clasificador lr, que devuelve las probabilidades 
# de clase para las muestras en X_test_std.
# DETALLES
# X_test_std[:3, :] selecciona las primeras 3 muestras del conjunto de datos de prueba 
# estandarizado.
# El resultado es un arreglo de probabilidades en el que cada fila corresponde a una muestra 
# y cada columna a la probabilidad de pertenecer a cada clase.

lr.predict_proba(X_test_std[:3, :])




# DESCRIPCIÓN
# Esta línea calcula la suma de las probabilidades predichas para cada una de las 3 muestras.
# DETALLES
# sum(axis=1) suma las probabilidades a lo largo de las columnas (es decir, para cada muestra).
# El resultado debe ser un arreglo con valor 1 para cada muestra, ya que las probabilidades 
# de pertenencia a todas las clases deben sumar 1.

lr.predict_proba(X_test_std[:3, :]).sum(axis=1)




# DESCRIPCIÓN
# Esta línea encuentra el índice de la clase con la mayor probabilidad para cada una de las 3 
# muestras.
# DETALLES
# argmax(axis=1) devuelve el índice de la clase con la probabilidad más alta para cada fila 
# (muestra).
# Esto permite identificar cuál es la clase más probable según el modelo para cada una de las 
# muestras.

lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)




# DESCRIPCIÓN
# Esta línea predice las clases para las primeras 3 muestras en el conjunto de datos de prueba 
# estandarizado.
# DETALLES
# El método predict devuelve las clases predichas directamente, basándose en las probabilidades 
# calculadas.
# Utiliza un umbral de 0.5 para clasificar si una muestra pertenece a la clase positiva o 
# negativa, o utiliza el índice de la clase con la mayor probabilidad en el caso de múltiples 
# clases.

lr.predict(X_test_std[:3, :])




# DESCRIPCIÓN
# Esta línea predice la clase de una única muestra (la primera) del conjunto de datos de prueba 
# estandarizado.
# DETALLES
# X_test_std[0, :].reshape(1, -1) reestructura la primera muestra para que tenga la forma 
# adecuada que espera el método predict.
# Al usar reshape(1, -1), se convierte la muestra en un arreglo de una fila con las 
# características en columnas, lo que es necesario para que el modelo funcione correctamente 
# con una sola muestra.

lr.predict(X_test_std[0, :].reshape(1, -1))


# ### Tackling overfitting via regularization



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_07.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_07.png', que es una ruta relativa al directorio actual.
# * width=700
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# INICIALIZACIÓN DE LISTAS
# Se crean dos listas vacías: weights para almacenar los coeficientes del modelo y params para 
# almacenar los valores de C.
weights, params = [], []

# BUCLE SOBRE VALORES DE C
# Se itera sobre un rango de valores de c desde -5 hasta 5.
# Para cada valor de c:
# - Se define un modelo de regresión logística con el parámetro de regularización C = 10^C . 
# Este parámetro controla la regularización: valores más pequeños de C aumentan la 
# regularización, mientras que valores más grandes permiten más complejidad en el modelo.
# - Se ajusta el modelo (fit) usando los datos de entrenamiento estandarizados 
# (X_train_std y y_train).
# - Se almacenan los coeficientes del modelo (lr.coef_[1] que corresponde a la segunda clase 
# en un modelo de clasificación binaria) en la lista weights.
# - También se agrega el valor de C (calculado como 10^C) en la lista params.
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

# CONVERSIÓN A NUMPY ARRAY
# Se convierte la lista de pesos a un arreglo de Numpy para facilitar el manejo de los datos.
weights = np.array(weights)

# VISUALIZACIÓN DE RESULTADOS
# Se grafican los coeficientes de las características (longitud y ancho del pétalo) en 
# función de los valores de C.
# Se utiliza plt.plot para trazar los coeficientes correspondientes a la longitud del 
# pétalo y el ancho del pétalo.
# Se añaden etiquetas a los ejes y una leyenda para identificar cada línea.
# Se establece la escala del eje x como logarítmica (plt.xscale('log')) para visualizar mejor 
# los cambios en los coeficientes a lo largo de varios órdenes de magnitud.
# Finalmente, se muestra el gráfico.
plt.plot(params, weights[:, 0],
         label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
# plt.savefig('figures/03_08.png', dpi=300)
plt.show()


# # Maximum margin classification with support vector machines



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_09.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_09.png', que es una ruta relativa al directorio actual.
# * width=700
# Ajusta el ancho de la imagen a 700 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## Maximum margin intuition

# ## Dealing with the nonlinearly separable case using slack variables



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_10.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_10.png', que es una ruta relativa al directorio actual.
# * width=600
# Ajusta el ancho de la imagen a 600 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# IMPLEMENTACIÓN DE LA CLASE SVC
# Se importa la clase SVC (Support Vector Classifier) de la biblioteca sklearn, que permite crear 
# modelos de clasificación utilizando Máquinas de Soporte Vectorial.

# CREACIÓN DEL MODELO SVM
# Se crea una instancia del clasificador SVM
# * kernel='linear'
# Se especifica que se usará un kernel lineal, lo que significa que se busca una frontera de 
# decisión lineal entre las clases.
# * C=1.0
# Este parámetro de regularización controla el equilibrio entre un margen más amplio y la 
# clasificación de los puntos de datos. Un valor de C más alto intenta clasificar todos los 
# puntos de datos correctamente, mientras que un valor más bajo permite un margen más amplio, 
# incluso si eso significa que algunos puntos están mal clasificados.
# * random_state=1
# Este parámetro se establece para garantizar la reproducibilidad de los resultados,
# especialmente en la inicialización aleatoria de los modelos.
svm = SVC(kernel='linear', C=1.0, random_state=1)

# ENTRENAMIENTO DEL MODELO
# Se ajusta el modelo SVM a los datos de entrenamiento (X_train_std y y_train), donde:
# * X_train_std
# Características de los datos de entrenamiento estandarizados.
# * y_train
# Etiquetas de las clases correspondientes a los datos de entrenamiento.
svm.fit(X_train_std, y_train)

# VISUALIZACIÓN DE LA FRONTERA DE DECISIÓN
# Se llama a la función plot_decision_regions, que grafica la frontera de decisión del 
# clasificador SVM.
# * X_combined_std
# Datos combinados (tanto de entrenamiento como de prueba) estandarizados para la visualización.
# * y_combined
# Etiquetas de clase correspondientes a los datos combinados.
# * classifier=svm
# Se pasa el clasificador SVM entrenado para que la función pueda utilizarlo para predecir la 
# clase y graficar la frontera de decisión.
# * test_idx=range(105, 150)
# Índices de los puntos de prueba que se destacarán en la visualización.
plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))

# ETIQUETAS Y ESTÉTICA DE LA GRÁFICA
# Se configuran las etiquetas de los ejes (xlabel y ylabel) para describir las características 
# utilizadas en el gráfico.
# Se añade una leyenda para identificar las clases en el gráfico.
# plt.tight_layout() ajusta el diseño del gráfico para que no se superpongan los elementos.
# Finalmente, plt.show() muestra la gráfica en pantalla. La línea comentada con plt.savefig 
# sugiere que se podría guardar la figura generada como un archivo PNG.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_11.png', dpi=300)
plt.show()


# ## Alternative implementations in scikit-learn



# IMPORTACIÓN DEL CLASIFICADOR SGDCLASSIFIER
# Se importa la clase SGDClassifier de la biblioteca sklearn, que permite implementar diferentes 
# algoritmos de clasificación utilizando el método de descenso de gradiente estocástico.

# CREACIÓN DE UN CLASIFICADOR PERCEPTRON
# Se crea una instancia del clasificador con la función de pérdida del Perceptrón:
# * loss='perceptron'
# Especifica que se usará la regla de aprendizaje del perceptrón, un clasificador lineal que se 
# entrena utilizando la técnica de aprendizaje en línea.
ppn = SGDClassifier(loss='perceptron')

# CREACIÓN DE UN CLASIFICADOR DE REGRESIÓN LOGÍSTICA
# Se crea una instancia del clasificador con la función de pérdida de Regresión Logística:
# * loss='log'
# Indica que se utilizará la función de pérdida logarítmica, adecuada para la clasificación 
# binaria, lo que significa que el modelo se ajustará para maximizar la probabilidad de las 
# clases utilizando la regresión logística.
lr = SGDClassifier(loss='log')

# CREACIÓN DE UN CLASIFICADOR SVM
# Se crea una instancia del clasificador con la función de pérdida Hinge:
# * loss='hinge'
# Utiliza la función de pérdida hinge, que es comúnmente empleada en Máquinas de Soporte 
# Vectorial (SVM) para clasificaciones lineales. Este tipo de clasificador busca maximizar 
# el margen entre las clases.
svm = SGDClassifier(loss='hinge')


# # Solving non-linear problems using a kernel SVM



# IMPORTACIÓN DE BIBLIOTECAS
# Se importan las bibliotecas necesarias: matplotlib.pyplot para crear gráficos y numpy para 
# manipular datos numéricos.

# CONFIGURACIÓN DE LA SEMILLA ALEATORIA
# Se establece una semilla para el generador de números aleatorios de numpy para garantizar la 
# reproducibilidad de los resultados, de modo que se obtengan los mismos números aleatorios en 
# cada ejecución.
np.random.seed(1)

# GENERACIÓN DE DATOS ALEATORIOS
# Se generan 200 puntos de datos aleatorios en un espacio bidimensional (2 características) 
# utilizando una distribución normal estándar (media 0 y desviación estándar 1).
X_xor = np.random.randn(200, 2)

# CREACIÓN DE ETIQUETAS XOR
# Se utiliza la función np.logical_xor para asignar etiquetas (clases) a los puntos de datos. 
# La clase 1 se asigna a los puntos donde una de las coordenadas es positiva y la otra es 
# negativa, mientras que la clase 0 se asigna donde ambas coordenadas son iguales 
# (ambas positivas o ambas negativas).
# El resultado es un array y_xor que contiene 1s y 0s, representando las dos clases.
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
print(X_xor[1])
print(y_xor[1])

# VISUALIZACIÓN DE LOS DATOS
# Se grafican los puntos de datos en un gráfico de dispersión (scatter plot).
# Los puntos de clase 1 (etiquetados como 1 en y_xor) se representan con cuadrados de color 
# azul ('royalblue').
# Los puntos de clase 0 (etiquetados como 0 en y_xor) se representan con círculos de color 
# rojo ('tomato').
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue',
            marker='s',
            label='Class 1')
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato',
            marker='o',
            label='Class 0')

# CONFIGURACIÓN DE LOS EJES Y LEYENDA
# Se establecen los límites de los ejes x e y.
# Se etiquetan los ejes.
# Se añade una leyenda para identificar las clases.
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()

# VISUALIZACIÓN FINAL
# Se descomenta la línea (si se desea) para guardar el gráfico como una imagen PNG.
# Finalmente, se muestra el gráfico con plt.show().
# plt.savefig('figures/03_12.png', dpi=300)
plt.show()




# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_13.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_13.png', que es una ruta relativa al directorio actual.
# * width=700
# Ajusta el ancho de la imagen a 700 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## Using the kernel trick to find separating hyperplanes in higher dimensional space



# CREACIÓN DEL CLASIFICADOR SVM
# Se crea un objeto SVC (Support Vector Classification) con los siguientes parámetros:
# * kernel='rbf'
# Utiliza un núcleo radial (Radial Basis Function), que es adecuado para problemas no lineales 
# como el XOR. Este tipo de núcleo permite que el SVM encuentre límites de decisión más complejos.
# * random_state=1
# Establece una semilla para el generador de números aleatorios para asegurar la reproducibilidad 
# del modelo.
# * gamma=0.10
# Este parámetro controla la influencia de un solo ejemplo de entrenamiento. Un valor más alto 
# de gamma implica que el modelo considerará puntos más cercanos a los vectores de soporte, lo 
# que puede llevar a una mayor complejidad en el modelo.
# * C=10.0
# Este parámetro controla la penalización por errores de clasificación. Un valor alto de C 
# significa que se penaliza más fuertemente los errores, lo que puede resultar en un modelo 
# más complejo.
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)

# ENTRENAMIENTO DEL MODELO
# Se entrena el clasificador SVM utilizando el conjunto de datos X_xor (las características) 
# y y_xor (las etiquetas de clase). El modelo ajusta sus parámetros internos para poder separar 
# las dos clases de datos.
svm.fit(X_xor, y_xor)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions, que probablemente fue definida en un bloque de 
# código anterior. Esta función genera un gráfico que muestra las regiones de decisión del 
# clasificador SVM entrenado. Cada región en el gráfico representa una clase diferente basada 
# en las predicciones del modelo. Los puntos de datos se grafican junto con las regiones 
# coloreadas que indican las clases predichas.
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

# CONFIGURACIÓN DE LA LEYENDA Y VISUALIZACIÓN
# Se añade una leyenda en la esquina superior izquierda del gráfico para identificar las clases.
# plt.tight_layout() ajusta los elementos del gráfico para que no se superpongan.
# Finalmente, se muestra el gráfico con plt.show(). Si se descomenta la línea plt.savefig(...), 
# se guardaría el gráfico como un archivo PNG.
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_14.png', dpi=300)
plt.show()




# IMPORTACIÓN DE LA CLASE SVC
# Se importa la clase SVC de sklearn.svm, que se utiliza para crear un clasificador de soporte 
# vectorial.

# CREACIÓN DEL CLASIFICADOR SVM
# Se inicializa un objeto de la clase SVC con los siguientes parámetros:
# * kernel='rbf'
# Especifica que se utilizará un núcleo de función base radial (RBF). Este tipo de núcleo es 
# adecuado para problemas no lineales, ya que permite que el modelo encuentre límites de 
# decisión más complejos.
# * random_state=1
# Establece una semilla para el generador de números aleatorios, lo que garantiza que los 
# resultados sean reproducibles.
# * gamma=0.2
# Este parámetro controla el alcance de la influencia de un solo vector de soporte. Un valor de 
# gamma más bajo implica que el modelo será más suave y puede ser más general, mientras que un 
# valor más alto puede dar lugar a un modelo más ajustado que se adapta mejor a los datos de 
# entrenamiento.
# * C=1.0
# Este parámetro determina la penalización por errores de clasificación. Un valor de C más alto 
# tiende a crear un modelo que se ajusta más a los datos de entrenamiento, mientras que un valor 
# más bajo puede permitir más errores, favoreciendo una mayor generalización.
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)

# ENTRENAMIENTO DEL MODELO
# Se entrena el modelo SVM utilizando el conjunto de datos de entrenamiento X_train_std 
# (características estandarizadas) y y_train (etiquetas de clase). El modelo ajusta sus 
# parámetros internos en función de los datos de entrenamiento.
svm.fit(X_train_std, y_train)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions, que visualiza las regiones de decisión del 
# clasificador SVM entrenado. Se utiliza X_combined_std y y_combined, que probablemente 
# contienen tanto los datos de entrenamiento como los de prueba, y test_idx se usa para 
# resaltar ejemplos específicos de prueba en el gráfico.
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))

# CONFIGURACIÓN DE ETIQUETAS Y VISUALIZACIÓN
# Se añaden etiquetas a los ejes X e Y del gráfico para indicar qué características se están 
# representando (longitud y ancho de los pétalos).
# Se añade una leyenda en la esquina superior izquierda para identificar las clases.
# plt.tight_layout() ajusta los elementos del gráfico para evitar superposiciones.
# Finalmente, plt.show() muestra el gráfico. La línea que está comentada (plt.savefig(...)) 
# se utilizaría para guardar el gráfico como un archivo PNG si se descomenta.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_15.png', dpi=300)
plt.show()




# CREACIÓN DEL CLASIFICADOR SVM
# Se inicializa un objeto de la clase SVC (Support Vector Classifier) con los siguientes 
# parámetros:
# * kernel='rbf'
# Utiliza un núcleo de función base radial (RBF). Este núcleo es adecuado para problemas no 
# lineales, permitiendo que el clasificador encuentre límites de decisión complejos.
# * random_state=1
# Fija la semilla del generador de números aleatorios para garantizar que los resultados sean 
# reproducibles.
# * gamma=100.0
# Este parámetro determina el alcance de la influencia de un solo vector de soporte. Un valor 
# alto de gamma, como 100.0, hace que el modelo se ajuste más a los datos de entrenamiento, 
# pudiendo resultar en un sobreajuste (overfitting) si no se maneja adecuadamente.
# * C=1.0
# Este parámetro controla la penalización por errores de clasificación. Un valor de C más 
# alto hace que el modelo penalice más los errores, buscando un ajuste más exacto a los datos 
# de entrenamiento.
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)

# ENTRENAMIENTO DEL MODELO
# Se entrena el clasificador SVM usando el conjunto de datos de entrenamiento X_train_std 
# (que ha sido estandarizado) y y_train (las etiquetas de clase correspondientes). Durante este 
# proceso, el modelo ajusta sus parámetros internos según los datos proporcionados.
svm.fit(X_train_std, y_train)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions, que crea un gráfico que visualiza las regiones 
# de decisión del clasificador SVM entrenado. Utiliza X_combined_std y y_combined, que 
# probablemente contengan tanto los datos de entrenamiento como los de prueba, y test_idx se 
# usa para destacar ejemplos específicos de prueba en el gráfico.
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))

# CONFIGURACIÓN DE ETIQUETAS Y VISUALIZACIÓN
# Se añaden etiquetas a los ejes X e Y del gráfico para indicar qué características se están 
# representando (longitud y ancho de los pétalos).
# Se añade una leyenda en la esquina superior izquierda para identificar las clases 
# representadas en el gráfico.
# plt.tight_layout() ajusta automáticamente el espacio del gráfico para evitar superposiciones.
# Finalmente, plt.show() muestra el gráfico en pantalla. La línea comentada (plt.savefig(...)) 
# podría utilizarse para guardar el gráfico como un archivo PNG si se descomenta.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_16.png', dpi=300)
plt.show()


# # Decision tree learning



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_17.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_17.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# DEFINICIÓN DE LA FUNCIÓN DE ENTROPÍA
# Esta función calcula la entropía H de una distribución binaria dada una probabilidad p de que 
# un evento pertenezca a la clase 1. La entropía se mide en bits y se utiliza para cuantificar 
# la incertidumbre de una distribución:
# La fórmula tiene en cuenta tanto la probabilidad p de que un evento pertenezca a la clase 1 
# como la probabilidad 1-p de que pertenezca a la clase 0.
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# GENERACIÓN DE VALORES DE PROBABILIDAD
# Se crea un array x que contiene valores desde 0.0 hasta 1.0 (sin incluir 1.0), con un paso 
# de 0.01. Esto representa diferentes probabilidades p para la clase 1.
x = np.arange(0.0, 1.0, 0.01)

# CÁLCULO DE LA ENTROPÍA PARA CADA PROBABILIDAD
# Se utiliza una lista por comprensión para calcular la entropía para cada valor de p en el 
# array x.
# Si p es igual a 0, se asigna None para evitar el cálculo de logaritmos de cero, ya que 
# log(0) no está definido.
ent = [entropy(p) if p != 0 else None 
       for p in x]

# VISUALIZACIÓN DE LA ENTROPÍA
# Se añaden etiquetas a los ejes del gráfico:
# El eje Y representa la entropía.
# El eje X representa la probabilidad de pertenencia a la clase 1 (p(i=1)).
# Se utiliza plt.plot(x, ent) para graficar la entropía en función de las probabilidades.
# Finalmente, plt.show() muestra el gráfico en pantalla. La línea comentada (plt.savefig(...)) 
# podría ser utilizada para guardar el gráfico como un archivo PNG si se descomenta.
plt.ylabel('Entropy')
plt.xlabel('Class-membership probability p(i=1)')
plt.plot(x, ent)
# plt.savefig('figures/03_26.png', dpi=300)
plt.show()




# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_18.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_18.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## Maximizing information gain - getting the most bang for the buck



# IMPORTACIÓN DE BIBLIOTECAS
# Se importan matplotlib.pyplot para crear gráficos y numpy para realizar cálculos numéricos.

# DEFINICIÓN DE FUNCIONES DE IMPUREZA

# GINI
# Calcula la impureza de Gini, que mide la probabilidad de que un elemento se clasifique 
# incorrectamente cuando se elige aleatoriamente.
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

# ENTROPÍA
# Calcula la entropía, que mide la incertidumbre en la clasificación.
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# ERROR DE CLASIFICACIÓN
# Calcula el error de clasificación, que es 1 menos la probabilidad máxima de pertenencia a 
# cualquiera de las clases.
def error(p):
    return 1 - np.max([p, 1 - p])

# GENERACIÓN DE VALORES DE PROBABILIDAD
# Se crea un array x que contiene valores de probabilidad que van de 0.0 a 1.0 (sin incluir 1.0), 
# con un paso de 0.01.
x = np.arange(0.0, 1.0, 0.01)

# CÁLCULO DE LOS ÍNDICES DE IMPUREZA
# Se calcula la entropía para cada valor de p, escalada a la mitad (sc_ent).
# También se calcula el error de clasificación para cada valor de p.
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

# VISUALIZACIÓN DE LOS RESULTADOS
# Se crea una figura y un eje para el gráfico.
# Se utilizan un bucle for para graficar cada índice de impureza en función de p. Cada línea se 
# etiquetará y se le asignará un estilo de línea y un color.
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini impurity', 'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# CONFIGURACIÓN DEL GRÁFICO
# Se añade una leyenda al gráfico que muestra qué línea corresponde a qué índice de impureza.
# Se añaden líneas horizontales en y=0.5 y y=1.0 como referencia.
# Se ajustan los límites del eje Y y se etiquetan los ejes.
# Finalmente, se muestra el gráfico utilizando plt.show(). La línea comentada (plt.savefig(...)) 
# podría ser utilizada para guardar el gráfico como un archivo PNG si se descomenta.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity index')
#plt.savefig('figures/03_19.png', dpi=300, bbox_inches='tight')
plt.show()


# ## Building a decision tree



# IMPORTACIÓN DEL MODELO DE ÁRBOL DE DECISIÓN
# Se importa DecisionTreeClassifier, que es el clasificador de árbol de decisión de la biblioteca 
# scikit-learn.

# CREACIÓN DEL MODELO DE ÁRBOL DE DECISIÓN
# Se instancia un objeto de DecisionTreeClassifier llamado tree_model con los siguientes 
# parámetros:
# * criterion='gini'
# Utiliza la impureza de Gini como criterio para la división de los nodos.
# * max_depth=4
# Limita la profundidad máxima del árbol a 4, lo que ayuda a evitar el sobreajuste.
# * random_state=1
# Establece una semilla para el generador de números aleatorios, asegurando que los 
# resultados sean reproducibles.
tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)

# ENTRENAMIENTO DEL MODELO
# Se entrena el modelo con los datos de entrenamiento (X_train como características y y_train 
# como etiquetas).
tree_model.fit(X_train, y_train)

# PREPARACIÓN DE LOS DATOS COMBINADOS
# Se combinan los conjuntos de datos de entrenamiento y prueba. X_combined contiene todas las 
# características de ambos conjuntos, y y_combined contiene las etiquetas correspondientes.
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions, que se utiliza para visualizar las regiones 
# de decisión del clasificador.
# classifier=tree_model especifica el modelo a visualizar.
# test_idx=range(105, 150) resalta los puntos de datos en el conjunto de prueba, que se 
# encuentran en el rango de índices 105 a 150.
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree_model,
                      test_idx=range(105, 150))

# ETIQUETADO Y FORMATO DEL GRÁFICO
# Se etiquetan los ejes X e Y para indicar que representan la longitud y el ancho de los 
# pétalos, respectivamente.
# Se añade una leyenda en la parte superior izquierda.
# plt.tight_layout() se asegura de que los elementos del gráfico se ajusten bien sin superponerse.
# La línea comentada (plt.savefig(...)) se puede utilizar para guardar el gráfico como un 
# archivo PNG si se descomenta.
# Finalmente, plt.show() muestra el gráfico en pantalla.
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_20.png', dpi=300)
plt.show()




# IMPORTACIÓN DEL MÓDULO DE ÁRBOLES
# Se importa el módulo tree de scikit-learn, que contiene funciones y clases para trabajar 
# con árboles de decisión.

# DEFINICIÓN DE NOMBRES DE CARACTERÍSTICAS
# Se crea una lista llamada feature_names que contiene los nombres de las características que 
# se utilizaron para entrenar el modelo. Estos nombres corresponden a las medidas de las flores 
# en el conjunto de datos, probablemente el conjunto de datos Iris.
feature_names = ['Sepal length', 'Sepal width',
                 'Petal length', 'Petal width']

# VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
# Se llama a la función plot_tree para visualizar el árbol de decisión representado por tree_model.
# * tree_model
# El modelo de árbol de decisión que fue entrenado previamente.
# * feature_names=feature_names
# Se pasan los nombres de las características para que aparezcan en el gráfico, lo que facilita 
# la interpretación del árbol.
# * filled=True
# Este parámetro indica que se debe llenar los nodos con colores basados en las clases predichas, 
# lo que ayuda a visualizar la decisión del árbol de manera más clara.
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)

# MOSTRAR EL GRÁFICO
# La línea comentada (plt.savefig(...)) sugiere que el gráfico se podría guardar como un archivo 
# PDF si se descomenta.
# plt.show() se utiliza para mostrar el gráfico en pantalla, permitiendo a los usuarios 
# visualizar el árbol de decisión.
# plt.savefig('figures/03_21_1.pdf')
plt.show()


# ## Combining weak to strong learners via random forests



# IMPORTACIÓN DEL CLASIFICADOR DE BOSQUE ALEATORIO
# Se importa la clase RandomForestClassifier del módulo ensemble de scikit-learn. 
# Este clasificador utiliza múltiples árboles de decisión para mejorar la precisión y reducir 
# el sobreajuste.

# CREACIÓN DEL MODELO DE BOSQUE ALEATORIO
# Se instancia un objeto de la clase RandomForestClassifier llamado forest.
# * n_estimators=25
# Este parámetro indica que se crearán 25 árboles de decisión en el bosque. Más árboles 
# generalmente mejoran la precisión del modelo, pero aumentan el tiempo de entrenamiento.
# * random_state=1
# Se establece una semilla aleatoria para asegurar la reproducibilidad de los resultados; 
# es decir, el modelo producirá los mismos resultados cada vez que se ejecute con esta semilla.
# * n_jobs=2
# Este parámetro permite utilizar 2 núcleos de procesamiento para entrenar el modelo en 
# paralelo, lo que puede acelerar el proceso, especialmente con un gran conjunto de datos.
forest = RandomForestClassifier(n_estimators=25, 
                                random_state=1,
                                n_jobs=2)

# ENTRENAMIENTO DEL MODELO
# Se entrena el modelo forest utilizando los datos de entrenamiento X_train (características) y 
# y_train (etiquetas o clases).
forest.fit(X_train, y_train)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# Se llama a la función plot_decision_regions para visualizar las regiones de decisión del 
# clasificador en el espacio de características.
# * X_combined
# Contiene tanto los datos de entrenamiento como de prueba.
# * y_combined
# Contiene las etiquetas correspondientes para los datos combinados.
# * classifier=forest
# Se pasa el modelo de bosque aleatorio entrenado para que se tracen sus regiones de decisión.
# * test_idx=range(105, 150)
# Se especifica un rango de índices para resaltar las instancias de prueba en el gráfico, 
# lo que permite observar cómo el modelo clasifica estos puntos.
plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

# ETIQUETADO Y PRESENTACIÓN DEL GRÁFICO
# Se establecen las etiquetas de los ejes X e Y.
# Se añade una leyenda en la parte superior izquierda del gráfico.
# plt.tight_layout() ajusta automáticamente los parámetros del gráfico para que se vea bien en 
# la figura.
# La línea comentada plt.savefig(...) sugiere que se podría guardar el gráfico como un archivo 
# de imagen si se descomenta.
# plt.show() se utiliza para mostrar el gráfico en pantalla.
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_2.png', dpi=300)
plt.show()


# # K-nearest neighbors - a lazy learning algorithm



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/03_23.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/03_23.png', que es una ruta relativa al directorio actual.
# * width=400
# Ajusta el ancho de la imagen a 400 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# IMPOTACIÓN DEL CLASIFICADOR KNN
# Se importa la clase KNeighborsClassifier del módulo neighbors de scikit-learn, que se utiliza 
# para crear un modelo de clasificación basado en el algoritmo KNN.

# CREACIÓN DEL MODELO KNN
# Se instancia un objeto de la clase KNeighborsClassifier llamado knn con los siguientes 
# parámetros:
# * n_neighbors=5
# Este parámetro indica que el modelo considerará los 5 vecinos más cercanos al realizar la 
# clasificación.
# * p=2
# Este parámetro especifica la distancia que se utilizará. Con p=2, se utiliza la distancia 
# euclidiana (que es una forma de la métrica de Minkowski).
# * metric='minkowski'
# Se define que la métrica utilizada para calcular la distancia entre puntos será la métrica 
# de Minkowski.
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')

# ENTRENAMIENTO DEL MODELO
# Se entrena el modelo knn utilizando los datos de entrenamiento X_train_std (características 
# estandarizadas) y y_train (etiquetas o clases). En el contexto de KNN, no se realiza un 
# entrenamiento explícito, pero se almacena la información de los datos de entrenamiento para 
# la clasificación futura.
knn.fit(X_train_std, y_train)

# VISUALIZACIÓN DE LAS REGLAS DE DECISIÓN
# Se llama a la función plot_decision_regions para visualizar las regiones de decisión del 
# clasificador KNN en el espacio de características.
# * X_combined_std
# Contiene tanto los datos de entrenamiento como de prueba (ya estandarizados).
# * y_combined
# Contiene las etiquetas correspondientes para los datos combinados.
# * classifier=knn
# Se pasa el modelo KNN entrenado para que se tracen sus regiones de decisión.
# * test_idx=range(105, 150)
# Se especifica un rango de índices para resaltar las instancias de prueba en el gráfico, 
# permitiendo observar cómo el modelo clasifica estos puntos.
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

# ETIQUETADO Y PRESENTACIÓN DEL GRÁFICO
# Se establecen las etiquetas de los ejes X e Y para indicar las características que se están 
# visualizando.
# Se añade una leyenda en la parte superior izquierda del gráfico para identificar las clases.
# plt.tight_layout() ajusta automáticamente los parámetros del gráfico para que se vea bien 
# en la figura.
# La línea comentada plt.savefig(...) sugiere que se podría guardar el gráfico como un archivo 
# de imagen si se descomenta.
# plt.show() se utiliza para mostrar el gráfico en pantalla.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_24_figures.png', dpi=300)
plt.show()


# # Summary

# ---
# 
# Readers may ignore the next cell.



# Ejecuta un comando en la terminal desde un entorno de Python (como un Jupyter Notebook o un 
# script que permite comandos de sistema) para convertir un notebook de Jupyter en un archivo 
# de script de Python. 
# * !
# Este símbolo se utiliza en entornos como Jupyter Notebooks para ejecutar comandos del sistema 
# operativo directamente desde el notebook. En este caso, el comando es una ejecución de un 
# script de Python.
# * python ../.convert_notebook_to_script.py
# Este comando ejecuta un script de Python llamado convert_notebook_to_script.py. Este archivo 
# se encuentra en el directorio anterior (../ indica que está un nivel arriba en el sistema de 
# archivos). El propósito de este script es convertir un notebook de Jupyter (.ipynb) en un 
# archivo de script de Python (.py).
# * --input ch03.ipynb
# Esta es una opción o argumento que le indica al script cuál es el archivo de entrada, en este 
# caso, el notebook ch03.ipynb.
# * --output ch03.py
# Esta opción le indica al script que guarde la salida (el archivo convertido) con el nombre 
# ch03.py, que es un script de Python.


