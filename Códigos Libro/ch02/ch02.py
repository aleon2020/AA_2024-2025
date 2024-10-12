# coding: utf-8


import sys
# * from python_environment_check import check_packages
from python_environment_check import check_packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
# * from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap

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
# Este módulo, por su nombre, está diseñado para verificar que el entorno de Python 
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
    'pandas': '1.3.2'
}
check_packages(d)


# # Chapter 2 - Training Machine Learning Algorithms for Classification

# ### Overview
# 

# - [Artificial neurons – a brief glimpse into the early history of machine learning](#Artificial-neurons-a-brief-glimpse-into-the-early-history-of-machine-learning)
#     - [The formal definition of an artificial neuron](#The-formal-definition-of-an-artificial-neuron)
#     - [The perceptron learning rule](#The-perceptron-learning-rule)
# - [Implementing a perceptron learning algorithm in Python](#Implementing-a-perceptron-learning-algorithm-in-Python)
#     - [An object-oriented perceptron API](#An-object-oriented-perceptron-API)
#     - [Training a perceptron model on the Iris dataset](#Training-a-perceptron-model-on-the-Iris-dataset)
# - [Adaptive linear neurons and the convergence of learning](#Adaptive-linear-neurons-and-the-convergence-of-learning)
#     - [Minimizing cost functions with gradient descent](#Minimizing-cost-functions-with-gradient-descent)
#     - [Implementing an Adaptive Linear Neuron in Python](#Implementing-an-Adaptive-Linear-Neuron-in-Python)
#     - [Improving gradient descent through feature scaling](#Improving-gradient-descent-through-feature-scaling)
#     - [Large scale machine learning and stochastic gradient descent](#Large-scale-machine-learning-and-stochastic-gradient-descent)
# - [Summary](#Summary)



# * from IPython.display
# Importa desde el submódulo display del paquete IPython. Este módulo está diseñado para mostrar 
# y renderizar diferentes tipos de datos dentro de entornos interactivos, como Jupyter Notebooks.
# * import Image
# Importa la clase Image desde el módulo display. La clase Image se utiliza para mostrar 
# imágenes en el entorno interactivo (por ejemplo, en una celda de Jupyter Notebook).



# # Artificial neurons - a brief glimpse into the early history of machine learning



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_01.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_01.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## The formal definition of an artificial neuron



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_02.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_02.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## The perceptron learning rule



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_03.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_03.png', que es una ruta relativa al directorio actual.
# * width=600
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_04.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_04.png', que es una ruta relativa al directorio actual.
# * width=600
# Ajusta el ancho de la imagen a 600 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API




# CLASE PERCEPTRÓN
# Es un modelo de aprendizaje supervisado. El perceptrón ajusta los pesos de un modelo lineal 
# utilizando un conjunto de datos de entrenamiento y un algoritmo de aprendizaje.
class Perceptron:
    
    # PARÁMETROS DE INICIALIZACIÓN
    # - eta: Tasa de aprendizaje (un valor entre 0 y 1). Controla el tamaño de los ajustes en los 
    # pesos durante el entrenamiento.
    # - n_iter: Número de iteraciones (o épocas) sobre los datos de entrenamiento.
    # - random_state: Semilla para generar números aleatorios, asegurando la reproducibilidad al 
    # inicializar los pesos de forma aleatoria.
    # ATRIBUTOS
    # - w_: Vector de pesos que será ajustado durante el entrenamiento.
    # - b_: Sesgo (bias), que también se ajusta durante el entrenamiento.
    # - errors_: Lista que almacena el número de errores de clasificación (actualizaciones) 
    # en cada época del entrenamiento.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # MÉTODO FIT
    # - X: Matriz con los datos de entrenamiento (características de los ejemplos).
    # - y: Vector con las etiquetas o clases objetivo para cada ejemplo.
    # El método ajusta los pesos (w_) y el sesgo (b_) durante varias iteraciones sobre los 
    # datos de entrenamiento.
    # En cada iteración, se calculan las predicciones para cada muestra, se evalúa el error, y se 
    # actualizan los pesos y el sesgo en función de ese error.
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # MÉTODO NET_INPUT
    # Calcula el "input neto", que es la combinación lineal de las características de entrada 
    # ponderadas por los pesos más el sesgo (esto es lo que evalúa el perceptrón para hacer 
    # predicciones).
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # MÉTODO PREDICT
    # Devuelve la predicción de la clase para una entrada. Si el input neto es mayor o igual a 0, 
    # se predice la clase 1, y si no, se predice la clase 0. Esto equivale a aplicar una función 
    # de paso (unidad escalón).
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)




# DEFINICIÓN DE VECTORES
# * v1 = np.array([1, 2, 3])
# Se crea un vector v1 de tres componentes [1, 2, 3] usando numpy.
# * v2 = 0.5 * v1
# Se crea un nuevo vector v2, que es el resultado de multiplicar cada componente de v1 por 0.5, 
# obteniendo el vector [0.5, 1.0, 1.5].
v1 = np.array([1, 2, 3])
v2 = 0.5 * v1

# CÁLCULO DEL ÁNGULO ENTRE V1 Y V2
# * v1.dot(v2)
# Calcula el producto punto (o producto escalar) entre los vectores v1 y v2.
# * np.linalg.norm(v1) y np.linalg.norm(v2)
# Calculan las normas (magnitudes) de los vectores v1 y v2, respectivamente.
# * v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
# Calcula el coseno del ángulo entre los vectores usando la fórmula del producto punto 
# normalizado.
# * np.arccos(...)
# Aplica la función arco coseno (arccos) para obtener el ángulo en radianes entre v1 y v2.
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ## Training a perceptron model on the Iris dataset

# ### Reading-in the Iris data



# IMPORTACIÓN DE BIBLIOTECAS
# * import os
# Módulo para interactuar con el sistema operativo (aunque no se utiliza en este código).
# * import pandas as pd
# Importa la biblioteca pandas, que se usa para la manipulación de datos en estructuras 
# como DataFrame.

# BLOQUE TRY-EXCEPT
# Se intenta cargar el archivo de datos desde una URL.
# Dentro del bloque try:
# * URL del dataset
# La variable s almacena la URL del dataset Iris, proporcionado por la UCI Machine Learning 
# Repository.
# * pd.read_csv
# Se utiliza para leer el archivo CSV desde la URL. La función read_csv descarga y carga el 
# archivo como un DataFrame de pandas.
# * header=None
# Indica que el archivo no tiene encabezados.
# * encoding='utf-8'
# Especifica la codificación de caracteres.
# * except HTTPError
# Si ocurre un error (por ejemplo, si no se puede acceder a la URL), el código intentará cargar 
# el archivo desde una ruta local (iris.data). Esto permite que el programa siga funcionando 
# aunque no se pueda acceder a la URL.
try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')

# MOSTRAR LAS ÚLTIMAS FILAS
# * df.tail()
# Una vez cargado el archivo (ya sea desde la URL o el archivo local), esta línea muestra las 
# últimas 5 filas del conjunto de datos.
df.tail()


# ### Plotting the Iris data



# CONFIGURACIÓN DE MATPLOTLIB
# * %matplotlib inline
# Esta línea es un comando específico para entornos como Jupyter Notebook, que permite mostrar 
# los gráficos directamente en el cuaderno de trabajo.
# * import matplotlib.pyplot as plt
# Importa la biblioteca matplotlib.pyplot como plt para crear gráficos.
# * import numpy as np
# Importa numpy, que es útil para manipulación de arreglos numéricos.

# SELECCIÓN DE CLASES DE FLORES
# * y = df.iloc[0:100, 4].values
# Extrae las etiquetas de clase (Iris-setosa o Iris-versicolor) de las primeras 100 filas del 
# DataFrame df (que contiene el Iris dataset) y la quinta columna (que tiene los nombres de las 
# especies).
# * y = np.where(y == 'Iris-setosa', 0, 1)
# Convierte las etiquetas en valores numéricos: 0 para Iris-setosa y 1 para Iris-versicolor.
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# EXTRACCIÓN DE CARACTERÍSTICAS
# * X = df.iloc[0:100, [0, 2]].values
# Extrae dos características numéricas (la longitud del sépalo y la longitud del pétalo) de las 
# primeras 100 filas, seleccionando la primera columna (longitud del sépalo) y la tercera 
# columna (longitud del pétalo). Esto crea una matriz X de 100 filas y 2 columnas.
X = df.iloc[0:100, [0, 2]].values

# CREACIÓN DEL GRÁFICO DE DISPERSIÓN
# * plt.scatter(...)
# Crea dos gráficos de dispersión (uno para cada clase de flor).
# * Primer scatter
# Plotea los datos de Iris-setosa (primeras 50 filas de X), usando puntos rojos 
# (color='red', marker='o').
# * Segundo scatter
# Plotea los datos de Iris-versicolor (filas 50 a 100 de X), usando cuadrados azules 
# (color='blue', marker='s').
# * plt.xlabel('Sepal length [cm]')
# Etiqueta el eje X como "Sepal length [cm]".
# * plt.ylabel('Petal length [cm]')
# Etiqueta el eje Y como "Petal length [cm]".
# * plt.legend(loc='upper left')
# Agrega una leyenda para identificar las clases, colocada en la esquina superior izquierda.
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
# plt.savefig('images/02_06.png', dpi=300)

# MOSTRAR EL GRÁFICO
# * plt.show()
# Muestra el gráfico en pantalla.
plt.show()


# ### Training the perceptron model



# CREACIÓN DEL PERCEPTRÓN
# * ppn = Perceptron(eta=0.1, n_iter=10)
# Se crea una instancia del modelo de Perceptrón con una tasa de aprendizaje (eta) de 0.1 y 
# un número de iteraciones (épocas) igual a 10. Este perceptrón usará el algoritmo de 
# aprendizaje para ajustar los pesos durante el entrenamiento.
ppn = Perceptron(eta=0.1, n_iter=10)

# ENTRENAMIENTO DEL MODELO
# * ppn.fit(X, y)
# Se entrena el perceptrón con los datos de entrada X (características) y las etiquetas y 
# (clases objetivo). Durante el entrenamiento, el perceptrón ajusta sus pesos en función de 
# los errores de clasificación en cada época.
ppn.fit(X, y)

# VISUALIZACIÓN DEL NÚMERO DE ERRORES
# * plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# Se genera un gráfico que muestra el número de errores de clasificación (actualizaciones) en 
# cada época.
# * range(1, len(ppn.errors_) + 1)
# Representa las épocas, es decir, el número de iteraciones desde la 1 hasta la última.
# * ppn.errors_
# Es una lista que contiene el número de errores en cada época del entrenamiento.
# * marker='o'
# Usa círculos para marcar los puntos en el gráfico.
# * plt.xlabel('Epochs')
# Etiqueta el eje X como "Epochs" (épocas).
# * plt.ylabel('Number of updates')
# Etiqueta el eje Y como "Number of updates" 
# (número de actualizaciones o errores de clasificación).
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
# plt.savefig('images/02_07.png', dpi=300)

# MOSTRAR EL GRÁFICO
# * plt.show()
# Muestra el gráfico en pantalla.
plt.show()


# ### A function for plotting decision regions



# IMPORTACIÓN DE MÓDULOS
# Importa ListedColormap de matplotlib, que permite crear mapas de colores personalizados para 
# los gráficos.

# DEFINICIÓN DE LA FUNCIÓN
# * def plot_decision_regions(X, y, classifier, resolution=0.02)
# Define la función que toma como argumentos:
# * X
# Matriz de características (dos dimensiones).
# * y
# Vector de etiquetas de clase.
# * classifier
# Un modelo de clasificador entrenado (por ejemplo, un Perceptrón).
# * resolution
# Resolución del gráfico (el paso entre puntos en la malla).
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # CONFIGURACIÓN DE MARCADORES Y COLORES
    # * markers y colors
    # Se definen listas de símbolos (marcadores) y colores para representar diferentes clases 
    # en el gráfico.
    # * cmap
    # Se crea un mapa de colores a partir de la lista de colores, limitándolo al número de 
    # clases únicas en y.
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # CÁLCULO DEL ÁREA DE DECISIÓN
    # Se determina el rango de valores para las características en X (mínimos y máximos), 
    # ajustándolos para agregar un margen.
    # * np.meshgrid(...)
    # Crea una cuadrícula de puntos (malla) sobre el espacio definido por los límites de las 
    # características, usando una resolución especificada.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # PREDICCIÓN DE LA CLASE EN LA MALLA
    # * classifier.predict(...)
    # Utiliza el clasificador para predecir las clases para todos los puntos de la malla.
    # * lab = lab.reshape(xx1.shape)
    # Reshapea las predicciones para que coincidan con la forma de la malla.
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    # VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
    # * plt.contourf(...)
    # Dibuja un gráfico de contornos rellenos que representa las regiones de decisión 
    # en el espacio, utilizando el mapa de colores definido.
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # VISUALIZACIÓN DE EJEMPLOS DE CLASE
    # Un bucle for recorre las clases únicas en y y utiliza plt.scatter(...) 
    # para dibujar los puntos de datos reales en el gráfico, asignando a cada clase 
    # su respectivo color y marcador.
    # * label=f'Class {cl}'
    # Etiqueta cada clase en la leyenda.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')




# LLAMADA A LA FUNCIÓN
# * plot_decision_regions(X, y, classifier=ppn)
# Esta línea llama a la función que se definió previamente para mostrar cómo el clasificador ppn 
# (Perceptrón) separa las diferentes clases en el espacio definido por las características X y 
# las etiquetas y.
# * X
# Contiene las características (por ejemplo, longitud del sépalo y longitud del pétalo).
# * y 
# Contiene las etiquetas de clase (por ejemplo, las clases de flores).
plot_decision_regions(X, y, classifier=ppn)

# ETIQUETAS DE LOS EJES
# * plt.xlabel('Sepal length [cm]')
# Establece la etiqueta del eje X como "Sepal length [cm]", indicando que se está graficando 
# la longitud del sépalo.
# * plt.ylabel('Petal length [cm]')
# Establece la etiqueta del eje Y como "Petal length [cm]", indicando que se está graficando 
# la longitud del pétalo.
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')

# LEYENDA
# * plt.legend(loc='upper left')
# Agrega una leyenda al gráfico en la esquina superior izquierda, que ayuda a identificar 
# las clases representadas en el gráfico.
plt.legend(loc='upper left')
#plt.savefig('images/02_08.png', dpi=300)

# MOSTRAR EL GRÁFICO
# * plt.show()
# Muestra el gráfico generado en la interfaz de visualización 
# (por ejemplo, un cuaderno Jupyter o una ventana de gráficos).
plt.show()


# # Adaptive linear neurons and the convergence of learning

# ## Minimizing cost functions with gradient descent



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_09.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_09.png', que es una ruta relativa al directorio actual.
# * width=600
# Ajusta el ancho de la imagen a 600 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_10.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_10.png', que es una ruta relativa al directorio actual.
# * width=500
# Ajusta el ancho de la imagen a 500 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## Implementing an adaptive linear neuron in Python



# CLASE Y DOCUMENTACIÓN
# * AdalineGD
# Define la clase para el modelo Adaline.
# La documentación (docstring) describe los parámetros, atributos y métodos de la clase.
class AdalineGD:

    # PARÁMETROS DE INICIALIZACIÓN
    # * eta
    # Tasa de aprendizaje, controla el tamaño de los ajustes a los pesos (valor entre 0.0 y 1.0).
    # * n_iter
    # Número de iteraciones (épocas) sobre el conjunto de datos de entrenamiento.
    # * random_state
    # Semilla para inicializar los pesos aleatorios, lo que permite la reproducibilidad.
    # ATRIBUTOS
    # * w_
    # Array unidimensional que almacena los pesos ajustados después del entrenamiento.
    # * b_
    # Escalar que representa el sesgo (bias) ajustado.
    # * losses_
    # Lista que almacena los valores de la función de pérdida (error cuadrático medio) en cada 
    # época.

    # MÉTODO __INIT__
    # Inicializa los parámetros del modelo y establece los pesos y el sesgo.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # MÉTODO FIT
    # - Entrenamiento: Este método entrena el modelo utilizando los datos X (características) y 
    # y (etiquetas).
    # - Genera pesos iniciales aleatorios y un sesgo.
    # En cada época:
    # - Calcula la entrada neta usando net_input.
    # - Calcula la salida usando la función de activación (en este caso, lineal).
    # - Calcula el error como la diferencia entre las etiquetas verdaderas y las predicciones.
    # - Actualiza los pesos y el sesgo en función del error.
    # - Calcula la pérdida (error cuadrático medio) y la almacena.
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    # MÉTODO NET_INPUT
    # Calcula la entrada neta como el producto punto de X y los pesos, más el sesgo.
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # MÉTODO ACTIVATION
    # Define la función de activación. En este caso, simplemente devuelve la entrada sin 
    # cambios (función identidad).
    def activation(self, X):
        return X

    # MÉTODO PREDICT
    # Devuelve la etiqueta de clase para nuevas muestras. Utiliza un umbral de 0.5 para 
    # clasificar las salidas en 0 o 1.
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




# CREACIÓN DE LA FIGURA Y LOS EJES
# * fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# Crea una figura (fig) con dos subgráficos (ax), organizados en una fila y dos columnas. 
# El tamaño de la figura es de 10x4 pulgadas.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# ENTRENAMIENTO CON LA PRIMERA TASA DE APRENDIZAJE
# * ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
# Se crea una instancia del modelo Adaline con 15 épocas y una tasa de aprendizaje (eta) de 0.1. 
# Luego, se entrena el modelo con los datos X y y.
# * ax[0].plot(...)
# Se grafica la evolución de la función de pérdida (error cuadrático medio) en función de las 
# épocas. Se utiliza np.log10(ada1.losses_) para mostrar el logaritmo en base 10 de la pérdida, 
# lo que ayuda a visualizar mejor los cambios en rangos amplios.
# * ax[0].set_xlabel(...)
# Establece la etiqueta del eje X como "Epochs".
# * ax[0].set_ylabel(...)
# Establece la etiqueta del eje Y como "log(Mean squared error)".
# * ax[0].set_title(...)
# Establece el título del primer gráfico como "Adaline - Learning rate 0.1".
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

# ENTRENAMIENTO CON LA SEGUNDA TASA DE APRENDIZAJE
# * ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
# Se crea otra instancia del modelo Adaline con 15 épocas y una tasa de aprendizaje de 0.0001, 
# y se entrena con los mismos datos.
# * ax[1].plot(...)
# Se grafica la función de pérdida directamente (sin logaritmo) en función de las épocas.
# * ax[1].set_xlabel(...)
# Establece la etiqueta del eje X como "Epochs".
# * ax[1].set_ylabel(...)
# Establece la etiqueta del eje Y como "Mean squared error".
# * ax[1].set_title(...)
# Establece el título del segundo gráfico como "Adaline - Learning rate 0.0001".
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
# plt.savefig('images/02_11.png', dpi=300)

# MOSTRAR EL GRÁFICO
# * plt.show()
# Muestra la figura con ambos gráficos en pantalla.
plt.show()




# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_12.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_12.png', que es una ruta relativa al directorio actual.
# * width=700
# Ajusta el ancho de la imagen a 700 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).



# ## Improving gradient descent through feature scaling



# * Image(...)
# Usa la clase Image (probablemente importada desde IPython.display, como en el ejemplo anterior) 
# para mostrar una imagen en un entorno interactivo como Jupyter Notebook.
# * filename='./figures/02_13.png'
# Especifica la ruta de la imagen que se desea mostrar. En este caso, la imagen se encuentra en el
# archivo './figures/02_13.png', que es una ruta relativa al directorio actual.
# * width=700
# Ajusta el ancho de la imagen a 700 píxeles. Esto redimensiona la imagen para que ocupe ese 
# espacio de ancho, mientras que su altura se ajusta proporcionalmente (si no se especifica una 
# altura).





# COPIA DEL ARRAY
# * X_std = np.copy(X)
# Crea una copia del array X, llamada X_std, para que las transformaciones no modifiquen el 
# conjunto de datos original.
X_std = np.copy(X)

# NORMALIZACIÓN DE LA PRIMERA CARACTERÍSTICA
# * X[:, 0].mean()
# Calcula la media de la primera característica.
# * X[:, 0].std()
# Calcula la desviación estándar de la primera característica.
# Se resta la media de cada valor y se divide por la desviación estándar, 
# lo que transforma la primera columna de X a una nueva escala donde tendrá media 0 
# y desviación estándar 1.
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

# Normalización de la segunda característica
# X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
# Realiza el mismo proceso de normalización para la segunda característica 
# (segunda columna de X).
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()




# CREACIÓN Y ENTRENAMIENTO DEL MODELO
# * ada_gd = AdalineGD(n_iter=20, eta=0.5)
# Se crea una instancia del modelo Adaline con 20 épocas y una tasa de aprendizaje (eta) de 0.5.
# * ada_gd.fit(X_std, y)
# Se entrena el modelo utilizando el conjunto de datos estandarizado X_std y las etiquetas y.
ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# * plot_decision_regions(X_std, y, classifier=ada_gd)
# Se llama a la función plot_decision_regions para visualizar las regiones de decisión del 
# clasificador Adaline en el espacio de características estandarizadas.
# * plt.title('Adaline - Gradient descent')
# Se establece el título del gráfico.
# * plt.xlabel('Sepal length [standardized]')
# Se etiqueta el eje X, indicando que se está graficando la longitud del sépalo estandarizada.
# * plt.ylabel('Petal length [standardized]')
# Se etiqueta el eje Y para la longitud del pétalo estandarizada.
# * plt.legend(loc='upper left')
# Se agrega una leyenda al gráfico en la esquina superior izquierda.
# * plt.tight_layout()
# Ajusta el diseño del gráfico para que los elementos no se superpongan.
# * plt.show()
# Muestra el gráfico con las regiones de decisión.
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

# VISUALIZACIÓN DE LA FUNCIÓN DE PÉRDIDA
# * plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
# Se grafica la evolución del error cuadrático medio a lo largo de las épocas.
# * range(1, len(ada_gd.losses_) + 1)
# Genera el rango de épocas desde 1 hasta el número de épocas entrenadas.
# * ada_gd.losses_
# Contiene los valores de la función de pérdida para cada época.
# * plt.xlabel('Epochs')
# Se etiqueta el eje X como "Epochs".
# * plt.ylabel('Mean squared error')
# Se etiqueta el eje Y como "Mean squared error".
# * plt.tight_layout()
# Ajusta el diseño del gráfico.
# * plt.show()
# Muestra el gráfico de la función de pérdida.
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
#plt.savefig('images/02_14_2.png', dpi=300)
plt.show()


# ## Large scale machine learning and stochastic gradient descent



# CLASE ADALINESGD
# Implementa un clasificador de neurona lineal adaptativa (Adaline) utilizando el método de 
# gradiente descendente estocástico (SGD) para entrenar el modelo.
class AdalineSGD:

    # PARÁMETROS
    # * eta
    # La tasa de aprendizaje, que controla el tamaño de los pasos de ajuste de los pesos en 
    # cada iteración.
    # * n_iter
    # Número de épocas o iteraciones sobre el conjunto de datos de entrenamiento.
    # * shuffle
    # Si es True, mezcla los datos en cada época para evitar ciclos repetitivos que puedan 
    # afectar al aprendizaje.
    # * random_state
    # Semilla para la inicialización de los pesos y el mezclado aleatorio, lo que garantiza 
    # resultados reproducibles.
    # ATRIBUTOS
    # * w_
    # Los pesos del modelo, que son ajustados durante el entrenamiento.
    # * b_
    # El sesgo o "bias" del modelo, que también se ajusta durante el entrenamiento.
    # * losses_
    # Almacena los errores medios cuadrados por cada época, que indican qué tan bien se está 
    # ajustando el modelo.
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    # MÉTODO FIT
    # Ajusta los pesos y el sesgo del modelo a los datos de entrenamiento X (características) y 
    # y (etiquetas). Por cada época, actualiza los pesos utilizando el gradiente descendente 
    # estocástico.
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    # MÉTODO PARTIAL_FIT
    # Similar a fit, pero no reinicializa los pesos, lo que permite hacer entrenamiento 
    # incremental.
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    # MÉTODO _SHUFFLE
    # Reorganiza aleatoriamente los datos en cada iteración para evitar patrones repetitivos 
    # que puedan interferir en el aprendizaje.
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    # MÉTODO INITIALIZE_WEIGHTS
    # Inicializa los pesos y el sesgo con números pequeños aleatorios.
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
    
    # MÉTODO _UPDATE_WEIGHTS
    # Actualiza los pesos basándose en un solo ejemplo xi y su objetivo target. 
    # Calcula el error y ajusta los pesos y el sesgo usando la regla de aprendizaje de Adaline.
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
    
    # MÉTODO NET_INPUT
    # Calcula la suma ponderada de las entradas más el sesgo.
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # MÉTODO ACTIVATION
    # Devuelve el valor de activación, que en este caso es lineal (Adaline utiliza activación 
    # lineal).
    def activation(self, X):
        return X

    # MÉTODO PREDICT
    # Predice la clase (0 o 1) para una nueva entrada basándose en la salida de la activación.
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




# INICIALIZACIÓN DEL MODELO
# Se crea una instancia del clasificador Adaline usando gradiente descendente estocástico 
# (AdalineSGD) con 15 iteraciones (n_iter=15), una tasa de aprendizaje de 0.01 (eta=0.01), 
# y una semilla aleatoria (random_state=1) para garantizar resultados reproducibles.
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)

# ENTRENAMIENTO DEL MODELO
# El modelo se entrena utilizando los datos X_std (características estandarizadas) y 
# y (etiquetas de clase). Aquí, X_std probablemente contiene características estandarizadas 
# de la longitud del sépalo y la longitud del pétalo de flores, ya que estos atributos se 
# mencionan más adelante en los ejes.
ada_sgd.fit(X_std, y)

# VISUALIZACIÓN DE LAS REGIONES DE DECISIÓN
# La función plot_decision_regions dibuja las regiones de decisión generadas por el modelo 
# Adaline en los datos X_std y y. Las etiquetas de los ejes indican que las características 
# corresponden a la longitud del sépalo y la longitud del pétalo estandarizadas.
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')

# AJUSTE DEL DISEÑO DE LA FIGURA Y GUARDADO
# plt.tight_layout() ajusta automáticamente los márgenes de la gráfica para evitar 
# solapamientos, luego se guarda la figura en la ruta 'figures/02_15_1.png' con una resolución 
# de 300 DPI. Finalmente, se muestra la gráfica.
plt.tight_layout()
plt.savefig('figures/02_15_1.png', dpi=300)
plt.show()

# GRÁFICA DE LA FUNCIÓN DE PÉRDIDA
# Se genera una gráfica que muestra la evolución de la pérdida promedio (error cuadrático medio) 
# a lo largo de las 15 épocas de entrenamiento, lo que ayuda a visualizar cómo mejora el modelo 
# durante el aprendizaje.
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')

# GUARDADO Y VISUALIZACIÓN DE LA GRÁFICA DE PÉRDIDA
# Esta gráfica se guarda como 'figures/02_15_2.png' con una resolución de 300 DPI y luego se 
# muestra.
plt.savefig('figures/02_15_2.png', dpi=300)
plt.show()




# * ada_sgd
# Es una instancia previamente entrenada o inicializada del clasificador Adaline que utiliza 
# gradiente descendente estocástico (SGD).
# * X_std[0, :]
# Selecciona el primer ejemplo de las características estandarizadas (X_std). Aquí, X_std[0, :] 
# toma la primera fila de X_std, que corresponde a las características de un solo ejemplo 
# (una flor en este caso).
# * y[0]
# Toma el valor objetivo o la etiqueta asociada con ese primer ejemplo.
# El método partial_fit ajusta el modelo usando únicamente este ejemplo (X_std[0, :] y y[0]), 
# lo que significa que actualiza los pesos del modelo sin reinicializarlos. Este tipo de 
# ajuste incremental es útil en situaciones donde los datos llegan en secuencia o cuando no 
# se desea reentrenar el modelo desde cero.

ada_sgd.partial_fit(X_std[0, :], y[0])


# # Summary

# --- 
# 
# Readers may ignore the following cell



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
# * --input ch02.ipynb
# Esta es una opción o argumento que le indica al script cuál es el archivo de entrada, en este 
# caso, el notebook ch02.ipynb.
# * --output ch02.py
# Esta opción le indica al script que guarde la salida (el archivo convertido) con el nombre 
# ch02.py, que es un script de Python.


