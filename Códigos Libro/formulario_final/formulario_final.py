# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# # FORMULARIO EXAMEN FINAL

# # ÍNDICE

# - [TEMA 1: Configuración y Visualización del Entorno](#tema-1-configuración-y-visualización-del-entorno)
#     - [1.1 Configuración de las Rutas de Importación](#11-configuración-de-las-rutas-de-importación)
#     - [1.2 Verificación de las Versiones de los Paquetes](#12-verificación-de-las-versiones-de-los-paquetes)
#     - [1.3 Visualización de Imágenes](#13-visualización-de-imágenes)
#     - [1.4 Importación de Paquetes](#14-importación-de-paquetes)
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
# X = df.iloc[:, 0:-1].values
# y = df.iloc[:, -1].values

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


# ## 2.6 División de Datos y Estandarización

# Y por último, se dividen los datos en conjuntos de entrenamiento y prueba, donde las características se estandarizan mediante la función StandardScaler(), de tal forma que se garantice que todas las características tengan media 0 y desviación estándar 1, algo muy importante en algoritmos sensibles a la escala de datos.



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


# ANÁLISIS DE LOS RESULTADOS
# 
# Como se puede ver, el dataset ha quedado dividido en un 30% de datos de prueba y un 70% de datos de entrenamiento. Es importante mencionar que la estandarización asegura que todas las características se encuentren en una escala comparable, algo esencial para algoritmos como Redes Neuronales o Máquinas de Soporte Vectorial (SVM, Support Vector Machines), lo que les permite mejorar la estabilidad y la precisión del modelo.

# # TEMA 3: Métodos de Compresión de Datos y Reducción Dimensional

# ## 3.1 Reducción Dimensional No Supervisada mediante Análisis de Componentes Principales (PCA)



# PÁGINA 144
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', eigen_vals)
# ----------

print('\nEigenvectors \n', eigen_vecs)




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




# PÁGINA 146
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# ----------




# PÁGINA 146
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
# ----------




# PÁGINA 147
X_train_std[0].dot(w)
# ----------




# PÁGINA 147
X_train_pca = X_train_std.dot(w)
# ----------




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




# PÁGINA 151
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
# ----------




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


# ## 3.2 Compresión de Datos Supervisada mediante Análisis Discriminante Lineal (LDA)



# PÁGINA 156
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(0, 2):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')
# ----------




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




# PÁGINA 158
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# ----------




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




# PÁGINA 160
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
# ----------




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


# ## 3.3 Técnicas de Reducción Dimensional No Lineal



# PÁGINA 166
digits = load_digits()
# ----------




# PÁGINA 166
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()
# ----------




# PÁGINA 166
digits.data.shape
# ----------




# PÁGINA 167
y_digits = digits.target
X_digits = digits.data
# ----------




# PÁGINA 167
tsne = TSNE(n_components=2, init='pca',
            random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)
# ----------




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


# # ANEXO: Convertir Jupyter Notebook a Fichero Python

# ## A.1 Script en el Directorio Actual





# ## A.2 Script en el Directorio Padre




