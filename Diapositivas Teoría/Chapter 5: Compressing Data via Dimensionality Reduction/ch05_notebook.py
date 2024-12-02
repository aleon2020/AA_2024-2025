# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# # Capítulo 5: Compresión de datos mediante reducción de dimensionalidad




# Image(filename='./figures/01_01.png', width=500)

# display(HTML("""
# <div style="display: flex; justify-content: center;">
#     <img src="./figures/01_01.png" width="500" height="300" format="png">
# </div>
# """))


# ## Introducción

# Existen diferentes métodos para reducir la dimensionalidad de los conjuntos de datos:
# 
# • Las técnicas de selección de características son un enfoque para lograrlo.
# 
# • Una alternativa a la selección de características para la reducción de dimensionalidad es la extracción de características, que implica transformar el conjunto de datos en un nuevo subespacio de características con menor dimensionalidad.
# 
# La reducción de dimensionalidad o la compresión de datos es importante en algunos casos en el Aprendizaje Automático. ¿Por qué?
# 
# • Al transformar datos de alta dimensión en un espacio de menor dimensión, la complejidad del conjunto de datos se reduce, lo que puede resultar en tiempos de entrenamiento y un menor uso de recursos informáticos.
# 
# • Al descartar datos irrelevantes o ruidosos se ayuda a crear datos y modelos más sólidos.
# 
# • Al reducir el número de dimensiones, permite visualizar datos (2 o 3 dimensiones), lo que facilita la identificación de patrones y clusters.
# 
# La diferencia entre la selección y la extracción de funciones es la siguiente:
# 
# • En la selección de características, seleccionamos un subconjunto de características original del conjunto de datos. El objetivo es identificar y mantener las características más relevantes que contribuyan al poder predictivo del modelo.
# 
# • En la extracción de características, se transforma y se proyecta el conjunto de datos original en un nuevo espacio de características. Esto implica crear nuevas características combinando las existentes, dando como resultado una
# dimensionalidad reducida que puede capturar información esencial del conjunto de datos.

# ## Extracción de características

# La extracción de características puede entenderse como un enfoque para la compresión de datos con el objetivo de mantener la mayor parte de la información relevante.
# 
# En la práctica, la extracción de características se utiliza para:
# 
# • Mejorar el espacio de almacenamiento.
# 
# • Mejorar la eficiencia computacional del aprendizaje.
# 
# • Permitir una visualización de los datos (2D o 3D).
# 
# • Mejorar el rendimiento predictivo descartando información irrelevante o ruidosa.
# 
# • Reducir la maldición de la dimensionalidad.
# 
# ¿Qué problemas son causados ​​por una alta dimensionalidad (la maldición de dimensionalidad)?
# 
# • En dimensiones altas, los datos tienden a ser más escasos. Esto significa que los puntos de datos están más separados (la mayor parte del espacio de alta dimensión está vacío), dificultando la identificación de patrones (haciendo que la agrupación y tareas de clasificación desafiantes).
# 
# • Entonces, en dimensiones altas se requieren más datos para llenar el espacio vacío y obtener resultados significativos.
# 
# • Si no tiene más datos, los algoritmos son propensos a sobreajustarse (no generaliza correctamente). En un intento por capturar toda la variabilidad del conjunto de datos, el modelo puede volverse complejo y ajustarse a datos irrelevantes y detalles que no se generalizan bien en datos nuevos.

# ## Análisis de Componentes Principales (PCA)

# El Análisis de Componentes Principales (PCA) es una técnica de transformación de análisis lineal no supervisado ampliamente utilizada en diferentes campos para la extracción de características y la reducción de dimensionalidad.
# 
# PCA tiene como objetivo encontrar las direcciones de máxima varianza en datos de alta dimensión y proyecta los datos en un nuevo subespacio con iguales o menores dimensiones que el original.
# 
# Los ejes ortogonales (componentes principales) del nuevo subespacio pueden interpretarse como las direcciones de máxima varianza dada la restricción de que los nuevos ejes de características son ortogonales entre sí.
# 
# IMAGEN 05_01
# 
# W es una matriz de transformación d×k-dimensional que nos permite mapear un vector d-dimensional de las características del ejemplo de entrenamiento, x, en un nuevo subespacio de características k-dimensional que tiene menos dimensiones que el espacio de características d-dimensional original (normalmente, k << d).
# 
# IMAGEN 05_02
# 
# Como resultado de transformar el conjunto de datos d-dimensional original en este nuevo subespacio k-dimensional, el primer componente principal tendrá la mayor variación posible.
# 
# Todos los componentes principales consiguientes tendrán la mayor varianza dada la restricción de que estos componentes son no correlacionados (ortogonal) con los otros componentes principales.
# 
# Incluso si las características de entrada están correlacionadas, los componentes principales resultantes serán mutuamente ortogonales (no correlacionados).
# 
# Las direcciones PCA son muy sensibles al escalamiento de datos y debemos estandarizar las características antes de la PCA si las características se midieron en diferentes escalas y queremos asignar igual importancia a todas las características.
# 
# La estandarización normalmente implica:
# 
# • Centrar los datos: Restar la media de cada característica del conjunto de datos para que cada característica tenga una media de cero.
# 
# • Escalar los datos: Dividir cada característica por su desviación estándar para que cada característica tiene una desviación estándar de uno. 
# 
# Este proceso a menudo se denomina normalización de puntuación z. Al estandarizar los datos, se garantiza que cada característica contribuya por igual al cálculo de los componentes principales, permitiendo a PCA identificar las direcciones de máxima varianza sin verse sesgadas por la escala de las características.
# 
# El enfoque se resumen en 7 sencillos pasos:
# 
# 1. Estandarizar el conjunto de datos d-dimensional.
# 
# 2. Construir la matriz de covarianza.
# 
# 3. Descomponer la matriz de covarianza en sus vectores propios y valores propios.
# 
# 4. Ordenar los valores propios en orden decreciente para clasificar los correspondientes vectores propios.
# 
# 5. Seleccionar k vectores propios, que corresponden a los k valores propios más grandes, donde k es la dimensionalidad del nuevo subespacio de características (k <= d).
# 
# 6. Construir una matriz de proyección, W, a partir de los k vectores propios “superiores”.
# 
# 7. Transformar el conjunto de datos de entrada d-dimensional, X, usando la matriz de proyección, W, para obtener el nuevo subespacio de características k-dimensional.
# 
# IMAGEN 05_03
# 
# La matriz de covarianza es un caso especial de matriz cuadrada (matriz simétrica), lo que significa que la matriz es igual a su transpuesta (A = A^T).
# 
# Cuando descomponemos propiamente una matriz simétrica de este tipo, los valores propios son números reales (en lugar de complejos) y los vectores propios son ortogonales (perpendiculares) a cada uno.
# 
# Además, los valores propios y los vectores propios vienen en pares. Si descomponemos una matriz de covarianza en sus vectores propios y valores propios, los vectores propios asociados con el valor propio más alto corresponden a la dirección de varianza máxima en el conjunto de datos.



df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None
)




X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# Split dataset in train and test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)

# Standarize the features and the d-dimensional dataset
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# Después de la estandarización, se construye la matriz de covarianza.
# 
# Es una matriz simétrica d×d-dimensional, donde d es el número de características en el conjunto de datos.
# 
# Almacena las covarianzas por pares entre las diferentes características.
# 
# Por ejemplo, se puede calcular la covarianza entre dos características mediante la siguiente ecuación:
# 
# IMAGEN 05_04
# 
# Por ejemplo, la matriz de covarianza de tres características puede entonces escribirse de la siguiente manera:
# 
# IMAGEN 05_05
# 
# Los vectores propios de la matriz de covarianza representan los principales componentes (las direcciones de máxima varianza), mientras que los valores propios correspondientes definen su magnitud.
# 
# IMAGEN 05_06




# Compute the covariance matrix
cov_mat = np.cov(X_train_std.T)

# linalg.eig() function
# Calculate the eigenvalues and eigenvectors of the covariance matrix
# We use the linalg.eig function from NumPy to compute the eigenvectors and eigenvalues
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n', eigen_vals)
print('\nEigenvectors \n', eigen_vecs)


# Como queremos reducir la dimensionalidad de nuestro conjunto de datos, solo seleccionamos el subconjunto de los vectores propios (componentes principales) que contiene la mayor parte de la información (varianza).
# 
# Los valores propios definen la magnitud de los vectores propios, por lo que tenemos que ordenar los valores propios por magnitud decreciente.
# 
# Estamos interesados ​​en los k vectores propios superiores en función de sus valores propios correspondientes.
# 
# A continuación, se trazan los ratios de varianza explicados de los valores propios. La relación de varianza explicada de un valor propio es simplemente la fracción de un valor propio y la suma total de los valores propios:
# 
# IMAGEN 05_07



tot = sum(eigen_vals)
var_exp = [(i / tot) for i in 
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 14), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Usando la función NumPy cumsum, podemos entonces calcular la suma acumulada de varianzas explicadas.
# 
# El gráfico resultante indica que el primer componente principal por sí solo representa aproximadamente el 40% de la varianza en el conjunto de datos.
# 
# Además, podemos ver que los dos primeros componentes principales combinados explican casi el 60% de la varianza en el conjunto de datos.
# 
# Hemos descompuesto la matriz de covarianza en pares propios (vectores propios y valores propios). Ahora hay que:
# 
# • Ordenar los valores propios en orden decreciente.
# 
# • Seleccionar k vectores propios, que corresponden a los k valores propios más grandes, donde k es la dimensionalidad del nuevo subespacio de características (k <= d).
# 
# • Construir una matriz de proyección, W, a partir de los k vectores propios.
# 
# • Transformar el conjunto de datos de entrada d-dimensional, X, usando la matriz de proyección, W, para obtener el nuevo subespacio de características k-dimensional.



# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# Dos dimensiones
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)




# Una muestra (x' = xW)
X_train_std[0].dot(w)




# El dataset completo (X' = XW)
X_train_pca = X_train_std.dot(w)


# Se visualiza el conjunto de datos de entrenamiento transformado en un diagrama de dispersión en un formato bidimensional.
# 
# Aunque codificamos la información de la etiqueta de clase con fines ilustrativos en el diagrama de dispersión anterior, debemos tener en cuenta que PCA es una técnica no supervisada que no utiliza ninguna clase de información de la etiqueta.



colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()





def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
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




# Principal component analysis in scikit-learn

# Inicialización del transformador de PCA
# y del estimador de Regresión Logística
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',
                        random_state=1,
                        solver='lbfgs')

# Reducción de dimensionalidad
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Ajuste del modelo de Regresión Logística en el dataset reducido
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()




plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# Si nos interesan los ratios de varianza explicados de los diferentes componentes principales, podemos simplemente inicializar la clase PCA con el parámetro n_components establecido como None, por lo que todos los componentes principales se mantienen y luego se puede acceder a la relación de varianza explicada a través del atributo explained_variance_ratio.



pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# Las nuevas características representan combinaciones lineales de las características originales con las
# componentes principales, donde cada Wij son las contribuciones de las funciones originales a la nueva función.
# 
# IMAGEN 05_08
# 
# Sin embargo, para medir las contribuciones de las características originales a la nueva característica, utilizamos versiones escaladas de W, donde cada vector propio se multiplica por la raíz cuadrada de su valor propio, cuyo resultado suele denominarse 'carga'.
# 
# ¿Por qué escalar?
# 
# Los valores resultantes pueden entonces interpretarse como la correlación entre las características originales y las nuevas características.



loadings = eigen_vecs * np.sqrt(eigen_vals)


# A continuación, se trazan las cargas para el primer componente principal (loadings[:, 0]), que es la primera columna de esta matriz.
# 
# Por ejemplo, el alcohol tiene una correlación negativa con la primera característica nueva (aproximadamente –0,3), mientras que el ácido málico tiene una correlación positiva (aproximadamente 0,54).



fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()


# Podemos obtener las cargas de un objeto PCA de scikit-learn ajustado en un manera similar, donde pca.components_ representa los vectores propios y pca.explained_variance_ representa los valores propios.



sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots()
ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()


# ## Análisis discriminante lineal (LDA)

# Después de explorar PCA como una técnica de extracción de características no supervisada, presentamos el Análisis Discriminante Lineal (LDA).
# 
# LDA es una técnica de transformación lineal que toma información y tiene en cuenta de las etiquetas de cada clase (algoritmo supervisado).
# 
# Es una técnica de transformación lineal que se puede utilizar para reducir el número de dimensiones (a veces también se denomina LDA de Fisher).
# 
# LDA busca encontrar una proyección lineal de los datos que maximice la separabilidad entre diferentes clases.
# 
# En otras palabras, LDA intenta identificar una o más direcciones en el espacio de características (discriminadores lineales) que maximizan la distancia entre medias de clase.
# 
# El número máximo de discriminantes lineales que puede ser obtenido con LDA está determinado por el número de clases y el número de características en su conjunto de datos.
# 
# Específicamente, el número máximo de discriminantes lineales es el menor de:
# 
# • C-1, donde C es el número de clases en el conjunto de datos.
# 
# • d, donde d es el número de características originales.
# 
# Por tanto, el número máximo de discriminantes lineales es mín(C-1,d).
# 
# IMAGEN 05_09
# 
# Resumamos brevemente los pasos principales que se requieren para realizar LDA:
# 
# 1. Estandarizar el conjunto de datos d-dimensional (d es el número de características).
# 
# 2. Para cada clase, calcular el vector medio d-dimensional.
# 
# 3. Construya la matriz de dispersión entre clases, Sb, y la dispersión dentro de clases matriz, Sw.
# 
# 4. Clacular los vectores propios y los valores propios correspondientes de la matriz Sw ^ -1 * Sb.
# 
# 5. Ordenar los valores propios en orden decreciente para clasificar los vectores propios correspondientes.
# 
# 6. Elegir los k vectores propios que correspondan a los k valores propios más grandes para construir una matriz de transformación d×k-dimensional, W; donde los vectores propios son las columnas de esta matriz.
# 
# 7. Proyectar los ejemplos en el nuevo subespacio de características usando la matriz de transformación, W.
# 
# Construimos la matriz de dispersión dentro de clases y entre clases.
# 
# Cada vector medio, mi, almacena el valor medio de la característica, 𝜇𝑚, com respecto a los ejemplos de la clase i:
# 
# IMAGEN 05_10



np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(
                X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')


# Ahora podemos calcular la matriz de dispersión dentro de clases, Sw, compuesta por matrices de dispersión individuales, Si, de cada clase individual:
# 
# IMAGEN 05_11



d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')


# La suposición que hacemos cuando calculamos el matrices de dispersión es que las etiquetas de clase en el conjunto de datos de entrenamiento son distribuidas uniformemente.
# 
# Sin embargo, si imprimimos el número de etiquetas de clase, vemos que esto viola el supuesto:



print('Class label distribution:',  
      np.bincount(y_train)[1:])


# Queremos escalar las matrices de dispersión individuales, Si, antes de nosotros resumirlos como la matriz de dispersión, Sw.
# 
# Cuando dividimos las matrices de dispersión por el número de ejemplos de clase, ni, podemos ver que calcular la matriz de dispersión es lo mismo que calcular la matriz de covarianza, Σ𝑖 (la matriz de covarianza es una versión normalizada de la matriz de dispersión):
# 
# IMAGEN 05_12



d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')


# Después de calcular la matriz de dispersión dentro de la clase escalada (o matriz de covarianza), podemos pasar al siguiente paso y calcular la matriz de dispersión entre clases Sb, donde m es la media general que se calcula incluyendo ejemplos de todas las clases c:
# 
# IMAGEN 05_13



mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1) 

d = 13  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot(
        (mean_vec - mean_overall).T)
print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')


# Realizamos la descomposición propia en la matriz Sw ^ -1 * Sb:



eigen_vals, eigen_vecs =\
    np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# Después de calcular los pares propios, podemos ordenar los valores propios en orden descendente:



eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, 
                     key=lambda k: k[0], reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])


# Apilemos ahora las dos columnas de vectores propios más discriminativas para crear la matriz de transformación, W:



w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)


# Usando la matriz de transformación W, ahora podemos transformar el conjunto de datos de entrenamiento multiplicando las matrices (X' = XW):



X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# Usando scikit-learn:



lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression(multi_class='ovr', random_state=1, 
                        solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()




X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# ## Incrustación de vecinos estocásticos distribuidos en t (t-SNE)

# PCA y LDA son técnicas de transformación lineal para funciones de extracción.
# 
# Sin embargo, hay técnicas de reducción de dimensionalidad no lineal.
# 
# Uno es la incrustación de vecinos estocásticos distribuidos en t (t-SNE), muy utilizado para visualizar conjuntos de datos de alta dimensión en dos o tres dimensiones.
# 
# ¿Por qué considerar la reducción de dimensionalidad no lineal?
# 
# • Muchos algoritmos de aprendizaje automático hacen suposiciones sobre la separabilidad lineal de los datos de entrada.
# 
# • Sin embargo, si estamos tratando con problemas no lineales, que podemos encontrar con bastante frecuencia en aplicaciones del mundo real, existen técnicas de transformación lineal para la reducción de dimensionalidad, como PCA, aunque LDA puede no ser la mejor opción.
# 
# IMAGEN 05_14
# 
# t-SNE aprende a incrustar puntos de datos en una dimensión inferior espacio tal que las distancias por pares en el espacio original se conservan.
# 
# t-SNE es una técnica destinada a fines de visualización, ya que requiere todo el conjunto de datos para la proyección.
# 
# Dado que proyecta los puntos directamente (a diferencia de PCA, no involucran una matriz de proyección), no podemos aplicar t-SNE a nuevos puntos de datos.
# 
# A continuación, se muestra una demostración rápida de cómo se puede aplicar t-SNE al conjunto de datos de dígitos de 64 dimensiones:



digits = load_digits()
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show() 




digits.data.shape




y_digits = digits.target
X_digits = digits.data




tsne = TSNE(n_components=2, init='pca',
            random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)




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
        
plot_projection(X_digits_tsne, y_digits)
plt.show()


# ## Convertir Jupyter Notebook a Fichero Python




