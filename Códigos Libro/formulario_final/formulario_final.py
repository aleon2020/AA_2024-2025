# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sb
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# # FORMULARIO EXAMEN FINAL

# # ÍNDICE

# - [TEMA 1: Configuración y Visualización del Entorno](#tema-1-configuración-y-visualización-del-entorno)
#     - [1.1 Configuración de las Rutas de Importación](#11-configuración-de-las-rutas-de-importación)
#     - [1.2 Verificación de las Versiones de los Paquetes](#12-verificación-de-las-versiones-de-los-paquetes)
#     - [1.3 Visualización de Imágenes](#13-visualización-de-imágenes)
#     - [1.4 Importación de Paquetes](#14-importación-de-paquetes)
# - [TEMA 2: Análisis Exploratorio de Datos (Compresión de Datos y Reducción Dimensional)](#tema-2-análisis-exploratorio-de-datos-compresión-de-datos-y-reducción-dimensional)
#     - [2.1 Carga y Exploración Inicial del Dataset](#21-carga-y-exploración-inicial-del-dataset)
#     - [2.2 Anonimización y Cálculo de la Correlación entre Características](#22-anonimización-y-cálculo-de-la-correlación-entre-características)
#     - [2.3 División de Variables Independientes y Dependiente](#23-división-de-variables-independientes-y-dependiente)
#     - [2.4 Mapa de Calor de Correlaciones](#24-mapa-de-calor-de-correlaciones)
#     - [2.5 Histogramas de Distribución de las Características](#25-histogramas-de-distribución-de-las-características)
# - [TEMA 3: Métodos de Compresión de Datos y Reducción Dimensional](#tema-3-métodos-de-compresión-de-datos-y-reducción-dimensional)
#     - [3.1 Reducción Dimensional No Supervisada mediante Análisis de Componentes Principales (PCA)](#31-reducción-dimensional-no-supervisada-mediante-análisis-de-componentes-principales-pca)
#         - [3.1.1 Paso 1: Estandarización del Conjunto de Datos D-Dimensional](#311-paso-1-estandarización-del-conjunto-de-datos-d-dimensional)
#         - [3.1.2 Paso 2: Construcción de la Matriz de Covarianza](#312-paso-2-construcción-de-la-matriz-de-covarianza)
#         - [3.1.3 Paso 3: Descomposición de la Matriz de Covarianza en Vectores y Valores Propios](#313-paso-3-descomposición-de-la-matriz-de-covarianza-en-vectores-y-valores-propios)
#         - [3.1.4 Paso 4: Ordenación de los Valores Propios en Orden Decreciente](#314-paso-4-ordenación-de-los-valores-propios-en-orden-decreciente)
#         - [3.1.5 Paso 5: Selección de k Vectores Propios correspondientes a los k Valores Propios Más Grandes](#315-paso-5-selección-de-k-vectores-propios-correspondientes-a-los-k-valores-propios-más-grandes)
#         - [3.1.6 Paso 6: Contrucción de la Matriz de Proyección W](#316-paso-6-contrucción-de-la-matriz-de-proyección-w)
#         - [3.1.7 Paso 7: Transformación del Dataset mediante la Matriz de Proyección W](#317-paso-7-transformación-del-dataset-mediante-la-matriz-de-proyección-w)
#         - [3.1.8 Visualización del Nuevo Espacio de Características](#318-visualización-del-nuevo-espacio-de-características)
#         - [3.1.9 Clasificación y Visualización de las Regiones de Decisión](#319-clasificación-y-visualización-de-las-regiones-de-decisión)
#         - [3.1.10 Explicación de la Varianza Total](#3110-explicación-de-la-varianza-total)
#         - [3.1.11 Carga de los Componentes Principales](#3111-carga-de-los-componentes-principales)
#     - [3.2 Compresión de Datos Supervisada mediante Análisis Discriminante Lineal (LDA)](#32-compresión-de-datos-supervisada-mediante-análisis-discriminante-lineal-lda)
#         - [3.2.1 Paso 1: Estandarización del Conjunto de Datos D-Dimensional](#321-paso-1-estandarización-del-conjunto-de-datos-d-dimensional)
#         - [3.2.2 Paso 2: Cálculo del Vector Medio D-Dimensional para cada Clase](#322-paso-2-cálculo-del-vector-medio-d-dimensional-para-cada-clase)
#         - [3.2.3 Paso 3: Construcción de las Matrices de Dispersión dentro de clases (S_W) y entre Clases (S_B)](#323-paso-3-construcción-de-las-matrices-de-dispersión-dentro-de-clases-s_w-y-entre-clases-s_b)
#         - [3.2.4 Paso 4: Cálculo de Vectores y Valores Propios de (S_W)^-1 * S_B](#324-paso-4-cálculo-de-vectores-y-valores-propios-de-s_w-1--s_b)
#         - [3.2.5 Paso 5: Ordenación de los Valores Propios en Orden Decreciente](#325-paso-5-ordenación-de-los-valores-propios-en-orden-decreciente)
#         - [3.2.6 Paso 6: Selección de los k Vectores Propios Más Grandes para Contruir la Matriz de Tranformación W](#326-paso-6-selección-de-los-k-vectores-propios-más-grandes-para-contruir-la-matriz-de-tranformación-w)
#         - [3.2.7 Paso 7: Proyección de Ejemplos en el Nuevo Subespacio usando la Matriz de Transformación W](#327-paso-7-proyección-de-ejemplos-en-el-nuevo-subespacio-usando-la-matriz-de-transformación-w)
#         - [3.2.8 Clasificación y Visualización de las Regiones de Decisión en el Subespacio LDA](#328-clasificación-y-visualización-de-las-regiones-de-decisión-en-el-subespacio-lda)
#     - [3.3 Técnicas de Reducción Dimensional No Lineal](#33-técnicas-de-reducción-dimensional-no-lineal)
#         - [3.3.1 Carga y Visualización de Imágenes de Dígitos](#331-carga-y-visualización-de-imágenes-de-dígitos)
#         - [3.3.2 Obtención de Dimensiones del Dataset y Separación de Características y Etiquetas](#332-obtención-de-dimensiones-del-dataset-y-separación-de-características-y-etiquetas)
#         - [3.3.3 Aplicación de t-SNE para Reducción Dimensional No Lineal](#333-aplicación-de-t-sne-para-reducción-dimensional-no-lineal)
#         - [3.3.4 Definición y Aplicación de la Función de Visualización](#334-definición-y-aplicación-de-la-función-de-visualización)
# - [TEMA 4: Métodos de Análisis de Agrupamiento](#tema-4-métodos-de-análisis-de-agrupamiento)
#     - [4.1 Clustering con K-means](#41-clustering-con-k-means)
#         - [4.1.1 Preparación de los Datos](#411-preparación-de-los-datos)
#         - [4.1.2 Método del Codo para Determinar el Número Óptimo de Clusters](#412-método-del-codo-para-determinar-el-número-óptimo-de-clusters)
#         - [4.1.3 Cálculo del Coeficiente de Silueta por Cluster](#413-cálculo-del-coeficiente-de-silueta-por-cluster)
#         - [4.1.4 Estandarización de los Datos](#414-estandarización-de-los-datos)
#         - [4.1.5 Reducción Dimensional con PCA](#415-reducción-dimensional-con-pca)
#         - [4.1.6 Visualización Final de los Clusters](#416-visualización-final-de-los-clusters)
#     - [4.2 Organización de clusters con Árboles Jerárquicos](#42-organización-de-clusters-con-árboles-jerárquicos)
#         - [4.2.1 Preparación de los Datos](#421-preparación-de-los-datos)
#         - [4.2.2 Cálculo del Linkage para Clustering Jerárquico](#422-cálculo-del-linkage-para-clustering-jerárquico)
#         - [4.2.3 Visualización de los Clusters con un Dendrograma](#423-visualización-de-los-clusters-con-un-dendrograma)
#         - [4.2.4 Reorganización de los Datos según el Dendrograma](#424-reorganización-de-los-datos-según-el-dendrograma)
#         - [4.2.5 Determinación de Clusters y Etiquetas](#425-determinación-de-clusters-y-etiquetas)
#         - [4.2.6 Clustering Aglomerativo para Etiquetas](#426-clustering-aglomerativo-para-etiquetas)
#         - [4.2.7 Escalado de Datos y Reducción de Dimensionalidad con PCA](#427-escalado-de-datos-y-reducción-de-dimensionalidad-con-pca)
#         - [4.2.8 Visualización de Clusters en el Espacio PCA](#428-visualización-de-clusters-en-el-espacio-pca)
#     - [4.3 Localización de regiones de alta densidad con DBSCAN](#43-localización-de-regiones-de-alta-densidad-con-dbscan)
#         - [4.3.1 Carga de Datos y Escalado de las Características](#431-carga-de-datos-y-escalado-de-las-características)
#         - [4.3.2 Visualización de Agrupamiento con K-means y Clustering Jerárquico](#432-visualización-de-agrupamiento-con-k-means-y-clustering-jerárquico)
#         - [4.3.3 Reducción de Dimensionalidad y Clustering con DBSCAN](#433-reducción-de-dimensionalidad-y-clustering-con-dbscan)
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





# # TEMA 2: Análisis Exploratorio de Datos (Compresión de Datos y Reducción Dimensional)

# ## 2.1 Carga y Exploración Inicial del Dataset

# En primer lugar, se carga el dataset 'dataset_compression.csv' y se configura la visualización de todas sus columnas. Además, se exploran las características más importantes del dataset como las columnas, su forma, una muestra de los primeros registros, así como un resumen de la información general y otras estadísticas descriptivas.



# PÁGINA 142
df = pd.read_csv('dataset_compression.csv')
# ----------

pd.set_option('display.max_columns', len(df.columns))

df.columns
df.shape
df.head(1)
df.info()
df.describe()
# dataset.tail()


# ANÁLISIS DE LOS RESULTADOS
# 
# El dataset contiene un total de 9 columnas y 768 filas. La exploración inicial revela la media, la desviación estándar, el mínimo, el máximo y los cuartiles de cada columna, lo que permite observar detalladamente la dispersión y la distribución de los datos, además de poder detectar problemas como valores nulos o distribuciones atípicas que puedan afectar al análsis posterior.

# ## 2.2 Anonimización y Cálculo de la Correlación entre Características

# Una vez realizado el primer análisis general de los datos, se elimina la columna 'Target' para crear un dataset anonimizado (dataset_compression_anonymized), el cual se guarda en un fichero con el mismo nombre. Después, se calcula la matriz de correlación, que permite evaluar relaciones lineales entre las columnas.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
dataset_compression_anonymized = df.drop(["Target"], axis=1)
dataset_compression_anonymized.to_csv('dataset_compression_anonymized.csv', index=False)
dataset_compression_anonymized.corr()

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# dataset_compression_anonymized = df.drop(["Target"], axis=1)
# dataset_N_characteristics = dataset_compression_anonymized.drop(['Col1', 'Col2', ..., 'ColN'], axis=1)
# dataset_N_characteristics.to_csv('dataset_N_characteristics.csv', index=False)
# dataset_N_characteristics.corr()


# ## 2.3 División de Variables Independientes y Dependiente

# Después de anonimizar los datos, se separan las variables independientes (X) de la variable objetivo (y), en este caso 'Target', y se imprimen las etiquetas de clase únicas, las cuales permiten preparar el dataset para su uso en modelos de clasificación.



# USANDO TODAS LAS CARACTERÍSTICAS DE LOS DATOS
X = dataset_compression_anonymized
y = df.get("Target")

# PÁGINA 55
print('Class labels:', np.unique(y))
# ---------

# USANDO N CARACTERÍSTICAS DE LOS DATOS
# X = dataset_N_characteristics
# y = df.get("Target")

# PÁGINA 55
# print('Class labels:', np.unique(y))
# ---------


# ANÁLISIS DE LOS RESULTADOS
# 
# El dataset contiene 2 clases (0 y 1), lo que implica un problema de clasificación con múltiples etiquetas. El balance entre clases se verifica a través de la distribución de etiquetas, permitiendo conocer si existe un desbalance significativo que pudiera requerir estrategias adicionales de ajuste o balanceo en el modelo.

# ## 2.4 Mapa de Calor de Correlaciones

# Con las variables X e y ya definidas, se genera un mapa de calor que representa gráficamente a la matriz de correlación de todas las características del dataset, lo que facilita la detección visual de las relaciones entre variables.



fig, ax = plt.subplots(figsize=(9,9))
sb.heatmap(df.corr(), linewidth = 0.5, annot=True)


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el mapa de calor muestra varias relaciones significativas entre algunas características.
# 
# Para otros apartados, habrá que seleccionar N de las mejores columnas entre todas las posibles, respectivamente. Para saber qué columnas poseen una mayor/mejor correlación, se seleccionan todos los valores de la fila 'Target' y pasarlos a valor absoluto, y una vez hecho esto, seleccionar los N valores más altos.
# 
# - Col7:   |0.29|  =   0.29
# - Col5:   |0.17|  =   0.17
# - Col2:   |0.24|  =   0.24
# - Col4:   |0.22|  =   0.22
# - Col8:   |0.075| =   0.075
# - Col3:   |0.47|  =   0.47
# - Col6:   |0.065| =   0.065
# - Col1:   |0.13|  =   0.13

# ## 2.5 Histogramas de Distribución de las Características

# El siguiente bloque de código crea un histograma para cada una de las 8 columnas que componen el dataset anonimizado, donde se visualiza la distribución de cada característica en función de su frecuencia.



columns = dataset_compression_anonymized.columns
fig = plt.figure(figsize=(12,12))
for i in range(0,8):
  ax = plt.subplot(3,3,i+1)
  ax.hist(dataset_compression_anonymized[columns[i]],bins = 20, color='blue', edgecolor='black')
  ax.set_title(dataset_compression_anonymized.head(0)[columns[i]].name)
plt.tight_layout()
plt.show()


# # TEMA 3: Métodos de Compresión de Datos y Reducción Dimensional

# ## 3.1 Reducción Dimensional No Supervisada mediante Análisis de Componentes Principales (PCA)

# ### 3.1.1 Paso 1: Estandarización del Conjunto de Datos D-Dimensional

# La estandarización es un paso previo necesario para el PCA (Principal Component Analysis), ya que asegura que todas las variables contribuyan de manera equitativa. Esto implica escalar los datos para que cada característica tenga media 0 y desviación estándar 1.



# PÁGINA 143
X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# ----------




# PÁGINA 55
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
# ---------


# ### 3.1.2 Paso 2: Construcción de la Matriz de Covarianza

# La matriz de covarianza refleja la relación de covarianza entre cada par de características en el conjunto de datos estandarizado. Esta matriz es crucial en PCA ya que permite identificar direcciones en el espacio de características con una mayor varianza.



# PÁGINA 144
cov_mat = np.cov(X_train_std.T)
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# La matriz de covarianza tiene dimensiones dxd (8x8 en este caso), lo que implica que cada elemento muestra cómo van variando 2 características juntas. Si se tienen valores grandes en la diagonal, se sugiere que la varianza de las características es alta, mientras que el resto de valores hacen referencia a la covarianza entre cada par de características.

# ### 3.1.3 Paso 3: Descomposición de la Matriz de Covarianza en Vectores y Valores Propios

# La descomposición en valores propios produce un conjunto de valores y vectores propios que describen la dirección y la magnitud de la varianza en todo el dataset.



# PÁGINA 144
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', eigen_vals)
# ----------

print('\nEigenvectors \n', eigen_vecs)


# ANÁLISIS DE LOS RESULTADOS
# 
# Los valores propios (Eigenvalues) obtenidos indican la cantidad de varianza explicada por cada componente principal. 
# 
# Por otro lado, los vectores propios (Eigenvectors) asociados muestran la dirección de las nuevas componentes principales. Un valor propio grande implica que la componente asociada explica una parte significativa de la variación en los datos.

# ### 3.1.4 Paso 4: Ordenación de los Valores Propios en Orden Decreciente

# Este paso permite clasificar los vectores propios de mayor a menor, dependiendo de la cantidad de varianza que comprenda cada uno.



# PÁGINA 145
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,9), var_exp, align='center', label='Individual explained variance')
plt.step(range(1,9), cum_var_exp, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Este gráfico muestra la varianza de cada componente y su varianza acumulada. Si las dos primeras componentes comprenden, por ejemplo, el 80% de la varianza, significa que una gran parte de la información de los datos originales se retiene en esas componentes, lo que sugeriría reducir el dataset a esas 2 dimensiones.

# ### 3.1.5 Paso 5: Selección de k Vectores Propios correspondientes a los k Valores Propios Más Grandes

# En este paso, se seleccionan las componentes principales que posean una mayor varianza.



# PÁGINA 146
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Se seleccionan los primeros k vectores propios, lo que implica elegir aquellas componentes que más información conserven del dataset original. Esto permite ayudar a reducir la dimensionalidad de una forma más efectiva.

# ### 3.1.6 Paso 6: Contrucción de la Matriz de Proyección W

# Se crea la matriz de proyección W usando los vectores propios seleccionados en el paso anterior. Esta matriz se utilizará para transformar los datos originales al nuevo espacio de características reducido.



# PÁGINA 146
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, la matriz W es de tamaño dxk (8x2 en este caso). Los valores de W obtenidos representan las direcciones de las 2 componentes principales más importantes, lo que facilitará la proyección del dataset a un subespacio de menor dimensionalidad.

# ### 3.1.7 Paso 7: Transformación del Dataset mediante la Matriz de Proyección W



# PÁGINA 147
X_train_std[0].dot(w)
# ----------




# PÁGINA 147
X_train_pca = X_train_std.dot(w)
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Se utiliza W para proyectar los datos originales en un espacio de menor dimensión.
# 
# El conjunto de datos proyectado (X_train_pca) se ha convertido en una versión reducida de X_train_std, donde se han mantenido las 2 componentes principales. Este subespacio mantiene una gran parte de la información que tenía inicialmente, lo que permite visualizar y clasificar los datos de una forma más efectiva.

# ### 3.1.8 Visualización del Nuevo Espacio de Características

# El gráfico resultante de este código muestra la separación de cada clase en función de las 2 componentes principales. Si se muestra una clara separación de las clases, se sugiere que PCA ha capturado bien las características discriminativas.



# PÁGINA 148
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# ----------




# PÁGINA 149
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=f'Class {cl}',
        edgecolor='black')
# ----------


# ### 3.1.9 Clasificación y Visualización de las Regiones de Decisión

# Este código entrena un clasificador de Regresión Logística en los datos proyectados y visualiza las regiones de decisión en el espacio reducido.



# PÁGINA 150
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# ----------




# PÁGINA 151
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Las regiones de decisión indican cómo el modelo clasifica los datos en el espacio de las componentes principales. Si se observan márgenes claros entre las clases, se tiene una señal de que el clasificador logra una buena separación en el espacio reducido.

# ### 3.1.10 Explicación de la Varianza Total

# Se aplica PCA con todas las componentes para ver la varianza explicada por cada una y cómo contribuyen al total. En este caso, los valores de pca.explained_variance_ratio_ indican qué proporción de la varianza comprende cada componente. Esto ayuda a decidir cuántas componentes se deben usar para mantener un nivel de información suficiente en los datos.



# PÁGINA 151
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
# ----------


# ### 3.1.11 Carga de los Componentes Principales

# Y por último, se calculan las cargas de las componentes principales, las cuales indican cómo cada característica original contribuye a los nuevos componentes.



# PÁGINA 152
loadings = eigen_vecs * np.sqrt(eigen_vals)
# ----------




# PÁGINA 152
fig, ax = plt.subplots()
ax.bar(range(8), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(8))
ax.set_xticklabels(df.columns[0:-1], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()
# ----------




# PÁGINA 153
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# ----------




# PÁGINA 153
fig, ax = plt.subplots()
ax.bar(range(8), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(8))
ax.set_xticklabels(df.columns[0:-1], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Las cargas cercanas a 1 o -1 indican qué características tienen una fuerte relación con cada una de las componentes principales. Esto permite interpretar qué variables originales tienen más influencia en cada componente.

# ## 3.2 Compresión de Datos Supervisada mediante Análisis Discriminante Lineal (LDA)

# ### 3.2.1 Paso 1: Estandarización del Conjunto de Datos D-Dimensional

# La estandarización es un paso previo necesario para el LDA (Linear Discriminant Analysis), ya que éste depende de la varianza relativa de las características. Esto implica escalar los datos para que cada característica tenga media 0 y desviación estándar 1.



# PÁGINA 143
X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# ----------




# PÁGINA 55
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
# ---------


# ### 3.2.2 Paso 2: Cálculo del Vector Medio D-Dimensional para cada Clase

# Este paso calcula el vector medio para cada clase, lo que permite representar cada clase en el espacio de características.



# PÁGINA 156
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(0, 2):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Cada uno de los vectores medios obtenidos (MV 0 y MV 1 en este caso) representan el centroide de cada clase en el espacio de características estandarizado. Los vectores medios obtenidos son fundamentales para poder calcular las matrices de dispersión entre clases y dentro de las mismas, lo que ayuda a maximizar la separabilidad entre clases.

# ### 3.2.3 Paso 3: Construcción de las Matrices de Dispersión dentro de clases (S_W) y entre Clases (S_B)

# Estas matrices de dispersión son claves para LDA. La matriz S_W representa la dispersión de cada muestra respecto al centroide de su clase, mientras que S_B mide la dispersión entre los centroides de las diferentes clases.



# PÁGINA 157
d = 8
S_W = np.zeros((d, d))
for label, mv in zip(range(0, 2), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: 'f'{S_W.shape[0]}x{S_W.shape[1]}')
# ----------




# PÁGINA 157
print('Class label distribution:', np.bincount(y_train[1:]))
# ----------




# PÁGINA 158
d = 8
S_W = np.zeros((d, d))
for label, mv in zip(range(0, 2), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: 'f'{S_W.shape[0]}x{S_W.shape[1]}')
# ----------




# PÁGINA 158
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)
d = 8
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot(
        (mean_vec - mean_overall).T)
print('Between-class scatter matrix: 'f'{S_B.shape[0]}x{S_B.shape[1]}')
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# La matriz de dispersión dentro de clases (S_W) representa la variabilidad dentro de cada clase. Si los valores obtenidos en S_W son bajos, se sugiere que la dispersión dentro de las clases es menor.
# 
# La matriz de dispersión entre clases (S_B) indica la variabilidad entre cada clase. Si la S_B obtenida es grande en comparación con S_W, se sugiere que las clases están bien clasificadas, lo que facilita la clasificación.

# ### 3.2.4 Paso 4: Cálculo de Vectores y Valores Propios de (S_W)^-1 * S_B



# PÁGINA 158
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Al resolver la ecuación de valores propios para (S_W)^-1 * S_B se obtienen vectores y valores propios que definen las direcciones y magnitudes de máxima separabilidad entre las clases.
# 
# Los valores propios obtenidos indican la capacidad de separabilidad de cada dirección encontrada. Un valor propio grande sugiere que su correspondiente vector propio proporciona una buena discriminación entre clases.

# ### 3.2.5 Paso 5: Ordenación de los Valores Propios en Orden Decreciente

# Se ordenan los valores propios en orden descendente para seleccionar aquellas direcciones que maximicen la separabilidad entre clases.



# PÁGINA 158
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# ----------

# PÁGINA 159
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])
# ----------




# PÁGINA 159
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,9), discr, align='center', label='Individual discriminability')
# ----------

# PÁGINA 160
plt.step(range(1,9), cum_discr, where='mid', label='Cumulative discriminability')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Los valores propios ordenados indican la importancia de cada dirección en términos de discriminabilidad, donde los valores más grandes muestran que las primeras direcciones seleccionadas en LDA capturan mejor la variabilidad entre clases, lo cual es útil para reducir dimensiones sin perder información importante.

# ### 3.2.6 Paso 6: Selección de los k Vectores Propios Más Grandes para Contruir la Matriz de Tranformación W

# Se eligen los vectores propios correspondientes a los valores propios más altos para construir la matriz W, que proyectará los datos al subespacio de menor dimensión.



# PÁGINA 160
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
# ----------


# ANÁLISIS DE LOS RESULTAODS
# 
# La matriz de transformación W es de tamaño dxk (8x2 en este caso), donde k es el número de dimensiones seleccionadas. Esto permite reducir el espacio de características y maximizar la discriminación entre clases en el espacio transformado.

# ### 3.2.7 Paso 7: Proyección de Ejemplos en el Nuevo Subespacio usando la Matriz de Transformación W

# En el gráfico resultante, las clases deberían aparecer separadas si LDA ha capturado bien las diferencias entre ellas. Una buena separación visual en las primeras dos componentes discriminantes sugiere que el subespacio reducido es efectivo para clasificar las clases.



# PÁGINA 161
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l, 1], # * (-1)
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
# ----------


# ### 3.2.8 Clasificación y Visualización de las Regiones de Decisión en el Subespacio LDA

# Y por último, se entrena un clasificador de Regresión Logística en el subespacio LDA y se visualizan las regiones de decisión tanto para el conjunto de entrenamiento como para el conjunto de prueba.



# PÁGINA 162
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_std, y_train)
# ----------




# PÁGINA 162
# lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# lr = lr.fit(X_train_lda, y_train)
# plot_decision_regions(X_train_lda, y_train, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()
# ----------




# PÁGINA 163
# X_test_lda = lda.transform(X_test_std)
# plot_decision_regions(X_test_lda, y_test, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# La visualización de las regiones de decisión en el subespacio LDA muestra cómo el clasificador separa las clases. Si las regiones están bien definidas y se ajustan a las clases, se suugiere que el modelo es efectivo para discriminar en el subespacio reducido.

# ## 3.3 Técnicas de Reducción Dimensional No Lineal

# ### 3.3.1 Carga y Visualización de Imágenes de Dígitos

# Este bloque de código carga el conjunto de datos 'digits' y visualiza las primeras imágenes, lo que permite entender mejor la estructura de los datos antes de aplicar la reducción dimensional.



# PÁGINA 166
digits = load_digits()
# ----------




# PÁGINA 166
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Se muestran las primeras 3 imágenes de dígitos en escala de grises, representando cada imagen como una cuadrícula de píxeles de 8x8. Esto permite verificar que se trata de muestras de dígitos manuscritos con distintas formas y tamaños, lo que da una primera idea de la variabilidad en el conjunto.

# ### 3.3.2 Obtención de Dimensiones del Dataset y Separación de Características y Etiquetas

# En este paso, se obtiene la forma de los datos mediante digits.data.shape, y se separan las características (X_digits) y las etiquetas (y_digits), las cuales representan imágenes de dígitos y sus valores reales, respectivamente.



# PÁGINA 166
digits.data.shape
# ----------




# PÁGINA 167
y_digits = digits.target
X_digits = digits.data
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# El tamaño de los datos es 1797 filas y 64 columnas (una por cada píxel de las imágenes de 8x8). Cada conjunto tiene 64 características, que representan los niveles de intensidad de cada píxel en la imagen. Este alto número de características sugiere la necesidad de reducción dimensional para visualizar mejor los patrones.

# ### 3.3.3 Aplicación de t-SNE para Reducción Dimensional No Lineal

# Este bloque de código aplica el algoritmo t-SNE para reducir la dimensionalidad de las imágenes de 64 a 2 dimensiones, usando como inicialización PCA. Esto permite visualizar la estructura de los datos en un plano bidimensional.



# PÁGINA 167
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# La proyección t-SNE transforma los datos a 2 dimensiones, permitiendo visualizar las relaciones no lineales en el espacio reducido. Esto crea un mapa bidimensional en el que cada grupo representa un dígito diferente, idealmente formando grupos compactos y bien definidos.

# ### 3.3.4 Definición y Aplicación de la Función de Visualización

# Este paso define una función plot_projection para visualizar los dígitos en el espacio reducido de 2 dimensiones. Cada dígito tiene un color distinto, y se añade una etiqueta en el centro de cada grupo, indicando el número correspondiente.



# PÁGINA 167
def plot_projection(x, colors):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])
    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
# ----------




# PÁGINA 168
plot_projection(X_digits_tsne, y_digits)
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# El gráfico resultante muestra los dígitos agrupados en el plano bidimensional, con cada dígito identificado mediante un color y una etiqueta numérica. Si los grupos de dígitos aparecen bien separados, esto indica que t-SNE ha capturado con éxito la estructura no lineal de los datos, lo que permite ver cómo los dígitos similares están más próximos entre sí en el espacio reducido. En caso de solapamiento entre algunos dígitos, se podría ajustar mediante perplexity o explorar otras técnicas de reducción no lineal para mejorar la separación.

# # TEMA 4: Métodos de Análisis de Agrupamiento

# ## 4.1 Clustering con K-means

# ### 4.1.1 Preparación de los Datos



# PÁGINA 143
X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
# ----------


# ### 4.1.2 Método del Codo para Determinar el Número Óptimo de Clusters

# Este snippet de código implementa el método del codo para estimar el número óptimo de clusters en X. Para ello, se ejecuta KMeans para valores de clusters entre 1 y 14 y se calcula la distorsión/inercia encargada de medir la suma de las distancias cuadradas entre cada punto y su centroide.



# PÁGINA 313
distortions = []
for i in range(1, 15):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
# ----------

# PÁGINA 314
plt.plot(range(1,15), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, se busca el punto de codo. En este caso, la distorsión deja de decrecer considerablemente a partir de 4 clusters, por lo que se sugiere que el número óptimo de clusters es 4.

# ### 4.1.3 Cálculo del Coeficiente de Silueta por Cluster

# En este snippet de código se calcula y se representa el coeficiente de silueta para cada punto en X, todos ellos agrupados en 5 clusters (n_clusters=5).
# 
# El coeficiente de silueta mide la cohesión dentro del cluster y la separación entre todos ellos. Por último, cada cluster se grafica con barras horizontales para observar la calidad de la agrupación y la media del coeficiente de silueta para todos los puntos.



# PÁGINA 315
km = KMeans(n_clusters=5,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(
    X, y_km, metric='euclidean'
)
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
# ----------

# PÁGINA 316
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color="red",
            linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el coeficiente medio de silueta es 0.4, lo que indica que los clusters son levemente coherentes.
# 
# Si el coeficiente de silueta es cercano a 1, se tienen clusters compactos y bien separados. 
# 
# Sin embargo, si el coeficiente de silueta es negativo o bajo, es posible que el número de clusters no sea el adecuado o que la estructura de los datos no es fácil de agrupar.

# ### 4.1.4 Estandarización de los Datos

# Se estandarizan las características en X para que tengan una media de 0 y una desviación estándar de 1, además de los centroides de los clusters previamente calculados. Esto es importante para métodos de agrupamiento como KMeans, que dependen de la escala de las características.



# PÁGINA 143
sc = StandardScaler()
X_std = sc.fit_transform(X)
Cluster_std = sc.transform(km.cluster_centers_)
# ----------


# ### 4.1.5 Reducción Dimensional con PCA

# Se utiliza el análisis de componentes principales (PCA) para reducir las dimensiones de X a dos componentes principales. Esto facilita la visualización de los datos y de los centroides en un espacio bidimensional.
# 
# Si los puntos de cada cluster están agrupados y separados visualmente, significa que el algoritmo KMeans ha logrado una buena agrupación.



# PÁGINA 150
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
X_clusters = pca.transform(Cluster_std)
# ----------


# ### 4.1.6 Visualización Final de los Clusters

# Y por último, se visualizan los clusters en el espacio reducido a 2 dimensiones, donde cada cluster se muestra en un color distinto y los centroides están marcados con estrellas rojas.



# PÁGINA 317
km = KMeans(n_clusters=5,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X_pca[y_km == 0, 0],
            X_pca[y_km == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X_pca[y_km == 1, 0],
            X_pca[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')
plt.scatter(X_pca[y_km == 2, 0],
            X_pca[y_km == 2, 1],
            s=50,
            c='yellow',
            edgecolor='black',
            marker='o',
            label='Cluster 3')
plt.scatter(X_pca[y_km == 3, 0],
            X_pca[y_km == 3, 1],
            s=50,
            c='red',
            edgecolor='black',
            marker='^',
            label='Cluster 4')
plt.scatter(X_pca[y_km == 4, 0],
            X_pca[y_km == 4, 1],
            s=50,
            c='blue',
            edgecolor='black',
            marker='^',
            label='Cluster 5')
plt.scatter(X_clusters[:, 0],
            X_clusters[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# La visualización de los clusters permite evaluar la separación y dispersión de los grupos. Si los clusters están bien separados, se sugiere tener una buena estructura de agrupamiento. Además, la posición de los centroides puede ayudar a interpretar las características principales de cada grupo.

# ## 4.2 Organización de clusters con Árboles Jerárquicos

# ### 4.2.1 Preparación de los Datos



# PÁGINA 143
X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
# ----------


# ### 4.2.2 Cálculo del Linkage para Clustering Jerárquico

# Se utiliza la función linkage para calcular el dendrograma jerárquico con el método de enlace completo y la métrica euclidiana.



# PÁGINA 322
# ----------

# PÁGINA 323
row_clusters = linkage(df.values,
                       method='complete',
                       metric='euclidean')
# ----------


# ### 4.2.3 Visualización de los Clusters con un Dendrograma

# Se crea un dendrograma que muestra visualmente cómo los elementos se agrupan jerárquicamente.



# PÁGINA 323
pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
             index=[f'cluster {(i + 1)}' for i in
                    range(row_clusters.shape[0])])
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Los resultados obtenidos en row_clusters contienen una matriz de tamaño 767 filas x 4 columnas, donde cada fila describe un paso en la unión de clusters, cuyo valor más importante es el obtenido de la columna 'distance'.
# 
# La columna 'distance' muestra qué tan similares son los clusters unidos en cada paso.

# Una vez hecho esto, se crea un dendrograma que muestra visualmente cómo los elementos se agrupan jerárquicamente.



# PÁGINA 324
row_dendr = dendrogram(
    row_clusters
)   
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, se visualiza el dendrograma jerárquico, que facilita la identificación del número óptimo de clusters y muestra las distancias entre cada elemento y los clusters.

# ### 4.2.4 Reorganización de los Datos según el Dendrograma

# Se reorganizan las filas del DataFrame df según el orden jerárquico del dendrograma para visualizar los datos, permitiendo así visualizar similitudes entre cada fila usando un mapa de calor con un esquema de colores 'hot_r'.



# PÁGINA 326
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, 
                  interpolation='nearest',
                  cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()
# ----------


# ### 4.2.5 Determinación de Clusters y Etiquetas

# Se determina un número específico de clusters y se obtienen las etiquetas para los datos, además de identificar los clusters al cortar el dendrograma con un umbral definido por el número máximo de clusters (n_clusters).



n_clusters = 3
clusters = fcluster(row_clusters, n_clusters, criterion='maxclust')
print(clusters)


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el resultado obtenido contiene las etiquetas asignadas a cada dato. En este caso, la mayoría de los clusters contienen 2 datos.

# ### 4.2.6 Clustering Aglomerativo para Etiquetas

# Se utiliza la clase AgglomerativeClustering para implementar un modelo de clustering aglomerativo con las mismas configuraciones que el dendrograma para agrupar los datos y comparar los resultados.



# PÁGINA 328
ac = AgglomerativeClustering(n_clusters=3,
                             metric='euclidean',
                             linkage='complete')

labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Los labels obtenidos pueden coincidir o diferir con los resultados obtenidos en el dendrograma, lo que puede reflejar posibles variaciones en la implementación.

# ### 4.2.7 Escalado de Datos y Reducción de Dimensionalidad con PCA

# Se escalan los datos y se reducen las dimensiones para facilitar la visualización en un plano bidimensional.
# 
# Además, se normalizan los datos y se proyectan en dos dimensiones con PCA para una mejor visualización.



# PÁGINA 143
sc = StandardScaler()
X_std = sc.fit_transform(X)
# ----------

# PÁGINA 150
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
# ----------


# ### 4.2.8 Visualización de Clusters en el Espacio PCA

# Se grafican los clusters resultantes en el espacio PCA reducido, coloreando cada grupo para una mejor distinción.



# PÁGINA 317
plt.scatter(X_pca[labels == 0, 0],
            X_pca[labels == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')

plt.scatter(X_pca[labels == 1, 0],
            X_pca[labels == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')

plt.scatter(X_pca[labels == 2, 0],
            X_pca[labels == 2, 1],
            s=50,
            c='yellow',
            edgecolor='black',
            marker='o',
            label='Cluster 3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# La gráfica muestra la distrución de los clusters. En este caso, el cluster 1 se encuentra ligeramente disperso, mientras que los clusters 2 y 3 muestran una ligera superposición entre ellos, lo que podría sugerir relaciones de interés entre los datos.

# ## 4.3 Localización de regiones de alta densidad con DBSCAN

# ### 4.3.1 Carga de Datos y Escalado de las Características



# PÁGINA 143
X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
# ----------

# PÁGINA 143
sc = StandardScaler()
X = sc.fit_transform(X)
# ----------


# ### 4.3.2 Visualización de Agrupamiento con K-means y Clustering Jerárquico

# En primer lugar, se aplica el algoritmo K-means, que minimiza las distancias internas de cada cluster, para agrupar los datos en 3 clusters y graficar los resultados, donde se muestra la distribución de los datos según los clusters asignados.
# 
# Después, se aplica el clustering aglomerativo para agrupar los datos y graficar los resultados. Para ello, se utiliza un enfoque jerárquico para dividir los datos en 3 clusters usando enlace completo y métrica euclidiana, lo que facilita la comparación con el método de K-means.



# PÁGINA 330
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
km = KMeans(n_clusters=3, 
            random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='cluster 1')
# ----------

# PÁGINA 331
ax1.scatter(X[y_km == 1, 0], 
            X[y_km == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 2')
ax1.scatter(X[y_km == 2, 0], 
            X[y_km == 2, 1],
            c='green',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 3')
ax1.set_title('K-means clustering')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ac = AgglomerativeClustering(n_clusters=2,
                             metric='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0],
            X[y_ac == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Cluster 1')
ax2.scatter(X[y_ac == 1, 0],
            X[y_ac == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 2')
ax2.scatter(X[y_ac == 2, 0],
            X[y_ac == 2, 1],
            c='green',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 3')
ax2.set_title('Agglomerative clustering')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver en la gráfica de la izquierda, los 3 clusters muestran superposición entre la gran mayoría de los elementos de otros clusters. Sin embargo, en la gráfica de la derecha, se tiene una situación similar, con la diferencia de que los datos del cluster 1 se encuentran algo más dispersos de los clusters 2 y 3.

# ### 4.3.3 Reducción de Dimensionalidad y Clustering con DBSCAN

# En primer lugar, se proyectan los datos a 2 dimensiones mediante PCA, el cual reduce las dimensiones de los datos a un espacio bidimensional y manteniendo la mayor varianza posible, todo ello para visualizar el clustering. Esto es útil para la visualización de los datos y para algoritmos sensibles a las dimensiones.
# 
# Una vez hecho esto, se aplica el algoritmo DBSCAN para identificar regiones de alta densidad y detectar posibles outliers. Este algoritmo identifica regiones densas y separa datos dispersos como outliers. Sus parámetros 'eps' (radio) y 'min_samples' (mínimo de puntos) controlan la densidad mínima necesaria para formar un cluster.



# PÁGINA 150
pca = PCA(n_components=2)
X = pca.fit_transform(X)
# ----------

# PÁGINA 332
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == -1, 0],
            X[y_db == -1, 1],
            c='black',
            edgecolor='black',
            marker='*',
            s=40,
            label='Outliers')
plt.scatter(X[y_db == 0, 0],
            X[y_db == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Cluster 1')
plt.scatter(X[y_db == 1, 0],
            X[y_db == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 2')
plt.scatter(X[y_db == 2, 0],
            X[y_db == 2, 1],
            c='green',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()
# ----------


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, las estrellas negras muestran aquellos puntos clasificados como outliers (-1), mientras que el resto de puntos se encuentran distribuidos en 3 clusters.

# # ANEXO: Convertir Jupyter Notebook a Fichero Python

# ## A.1 Script en el Directorio Actual





# ## A.2 Script en el Directorio Padre




