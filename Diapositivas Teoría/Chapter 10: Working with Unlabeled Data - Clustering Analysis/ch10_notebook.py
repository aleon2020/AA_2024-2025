# coding: utf-8


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
# from scipy.cluster.hierarchy import set_link_color_palette
from packaging import version
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# # Capítulo 10: Trabajar con datos sin etiquetar: análisis de agrupación

# # Índice

# - [Introducción](#introducción)
# - [Agrupar objetos por similitud usando k-means](#agrupar-objetos-por-similitud-usando-k-means)
# - [Organizar clústeres como un árbol jerárquico](#organizar-clústeres-como-un-árbol-jerárquico)
# - [Localización de regiones de alta densidad mediante DBSCAN](#localización-de-regiones-de-alta-densidad-mediante-dbscan)
# - [Convertir Jupyter Notebook a Fichero Python](#convertir-jupyter-notebook-a-fichero-python)




# Image(filename='./figures/01_01.png', width=500)

# display(HTML("""
# <div style="display: flex; justify-content: center;">
#     <img src="./figures/01_01.png" width="500" height="300" format="png">
# </div>
# """))


# ## Introducción

# Las técnicas de aprendizaje supervisado crean modelos de aprendizaje automático utilizando datos cuya respuesta ya se conocía. Por ejemplo, en clasificación, las etiquetas de clase están disponibles en capacitación de datos.
# 
# Las técnicas de aprendizaje no supervisadas construyen el aprendizaje automático modelos que nos permiten descubrir estructuras ocultas en los datos donde no sabemos la respuesta correcta de antemano. Por ejemplo, en la agrupación, el modelo intenta encontrar una relación natural agrupar datos para que los elementos del mismo grupo sean más similares a entre sí que con aquellos de diferentes grupos.

# ## Agrupar objetos por similitud usando k-means

# El algoritmo k-means es uno de los algoritmos de agrupamiento más populares, que se utilizan ampliamente tanto en el mundo académico como en industria.
# 
# Intenta encontrar grupos de objetos similares que estén más relacionados entre sí, excepto a objetos de otros grupos.
# 
# Ejemplos de aplicaciones de clustering orientadas a los negocios incluyen la agrupación de documentos, música y películas por diferentes temas o encontrar clientes que compartan intereses similares basados sobre comportamientos de compra comunes como base para la recomendación.
# 
# El algoritmo k-means es extremadamente fácil de implementar y computacionalmente muy eficiente en comparación con otros algoritmos de agrupación.
# 
# El algoritmo k-means pertenece a la categoría de agrupamiento basado en prototipos. Existen otras categorías de agrupamiento, como el jerárquico y el agrupamiento basado en densidad.
# 
# La agrupación basada en prototipos significa que cada grupo está representado por un prototipo, que suele ser el centroide (promedio) de puntos similares con características continuas, o el medoide (el punto más representativo o que minimiza la distancia a todos los demás puntos que pertenecen a un grupo particular) en el caso de los rasgos categóricos.
# 
# K-means es muy bueno para identificar grupos con una forma esférica. Esto se debe a que el algoritmo minimiza la suma de la distancia de los cuadrados entre los puntos de datos y el centroide del grupo. En un espacio euclidiano, esta minimización tiende a formar grupos que son esféricos alrededor de sus centroides.
# 
# Uno de los inconvenientes de este algoritmo de agrupamiento es que tenemos para especificar el número de grupos, k, a priori. Una inapropiada elección de k puede dar como resultado un rendimiento de agrupación deficiente.
# 
# El método del codo y los gráficos de silueta son técnicas útiles para evaluar la calidad de una agrupación para ayudarnos a determinar la número óptimo de grupos, k.
# 
# Aunque la agrupación de k-means se puede aplicar a datos en dimensiones superiores, analizaremos en siguientes ejemplos que utilizan un conjunto de datos bidimensional simple con fines de visualización.



X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)
plt.scatter(X[:, 0], 
            X[:, 1], 
            c='white', 
            marker='o', 
            edgecolor='black', 
            s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.tight_layout()
plt.show()


# El objetivo es agrupar los ejemplos según similitud entre las características utilizando el algoritmo k-means, como se resume en los siguientes pasos:
# 
# 1. Elegir aleatoriamente k centroides de los ejemplos como grupo inicial.
# 
# 2. Asignar cada ejemplo al centroide más cercano (más similar), 𝜇(𝑗), 𝑗 ∈ {1, … , 𝑘}.
# 
# 3. Mover los centroides al centro de los ejemplos que fueron asignados.
# 
# 4. Repetir los pasos 2 y 3 hasta que las asignaciones de clúster no cambien o la tolerancia de cambio definida por el usuario o el número máximo de iteraciones haya sido alcanzado.
# 
# ¿Cómo medimos la similitud entre objetos?
# 
# Podemos definir la similitud como lo opuesto a la distancia, y un término comúnmente utilizado para agrupar ejemplos con características continuas es la distancia euclidiana al cuadrado entre dos puntos, x e y, en un espacio m-dimensional:
# 
# Nota: En esta ecuación, el índice j se refiere a la j-ésima dimensión (columna de características) del entradas de ejemplo, x e y.
# 
# IMAGEN 10_01
# 
# Con base en esta métrica de distancia euclidiana, podemos describir el algoritmo k-means como un problema de optimización simple. Un enfoque iterativo para minimizar la suma de cuadrados dentro del grupo de errores (SSE) o también llamado inercia de clúster:
# 
# IMAGEN 10_02
# 
# Establecemos el número de grupos deseados en 3 (tener que especificar el número de grupos a priori es una de las limitaciones de k-means).
# 
# Configuramos n_init=10 para ejecutar los algoritmos de agrupamiento de k-means 10 veces de forma independiente, con diferentes centroides aleatorios para elegir el modelo final como el que tiene el SSE más bajo.
# 
# A través del parámetro max_iter, especificamos el número máximo de iteraciones para cada ejecución (aquí, 300). Hay que tener en cuenta que la implementación de k-means en scikit-learn se detiene antes de que el tiempo converja y antes de alcanzar el número máximo de iteraciones.
# 
# Sin embargo, es posible que k-means no alcance la convergencia para una ejecución en particular, lo que puede ser problemático (computacionalmente costoso) si elegimos valores relativamente grandes para max_iter. Una forma de abordar los problemas de convergencia es elegir valores mayores para tol, que es un parámetro que controla la tolerancia con respecto a los cambios en el SSE dentro del clúster para declarar convergencia. En el código anterior, elegimos una tolerancia de 1e-04 (=0,0001).



km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)


# Escalado de funciones: Cuando aplicamos k-means a datos del mundo real usando una ecuación euclidiana métrica de distancia, queremos asegurarnos de que las características se midan en la misma escala y aplicar la estandarización de puntuación z o min-max escalando si es necesario.



plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


# Tenemos el inconveniente de tener que especificar el número de clusters, k, a priori. El número de clusters a elegir puede no ser siempre el mismo en aplicaciones del mundo real, especialmente si estamos trabajando con un conjunto de datos de dimensiones superiores que no se pueden visualizar.
# 
# Además, el algoritmo k-means utiliza una semilla aleatoria para colocar centroides, que a veces pueden resultar en clústeres defectuosos, lentos o convergentes si los centroides iniciales se eligen mal. Una forma de abordar este problema es ejecutar el algoritmo k-means varias veces en un conjunto de datos y elegir el modelo de mejor rendimiento en términos de ESS. Otra estrategia es colocar los centroides iniciales lejos unos de otros a través del algoritmo k-means++, que conduce a resultados mejores y más consistentes que las clásicas k-means.
# 
# La inicialización en k-means++ se puede resumir de la siguiente manera:
# 
# IMAGEN 10_03
# 
# El clustering duro describe una familia de algoritmos donde cada ejemplo en un conjunto de datos se asigna exactamente a un grupo, como en los algoritmos k-means y k-means++ que analizamos anteriormente en este capítulo.
# 
# Por el contrario, los algoritmos para clustering suave (a veces también llamados de clustering difuso) asignan un ejemplo a uno o más clusters. Un ejemplo de agrupamiento suave es el algoritmo difuso de medias C (FCM), también llamados k-means suaves o k-means difuso.
# 
# Uno de los principales desafíos del aprendizaje no supervisado es que no se sabe la respuesta definitiva. No tenemos etiquetas de clase reales en nuestro conjunto de datos que nos permitan evaluar el desempeño del modelo.
# 
# Por lo tanto, para cuantificar la calidad de la agrupación, necesitamos utilizar métricas intrínsecas, como la SSE (inercia) dentro del clúster, para comparar el rendimiento de diferentes agrupaciones de modelos de k-means. En scikit-learn ya se puede acceder a través del atributo inertia_ después del ajuste un modelo KMeans.



print(f'Distortion: {km.inertia_:.2f}')


# Basado en el SSE dentro del clúster, podemos usar una herramienta gráfica, el llamado método del codo, para estimar el número óptimo de grupos, k, para una tarea determinada.
# 
# Podemos decir que si k aumenta, la inercia disminuirá. Esto es porque los ejemplos estarán más cerca de los centroides que se han asignado.
# 
# La idea detrás del método del codo es identificar el valor de k donde la distorsión comienza a aumentar más rápidamente.



distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# Otra métrica intrínseca para evaluar la calidad de un clustering es el análisis de silueta, que también se puede aplicar a la agrupación de algoritmos distintos de k-means.
# 
# El análisis de silueta se puede utilizar como herramienta gráfica para trazar una medida de qué tan estrechamente agrupados están los ejemplos en los grupos (mida qué tan similar es un objeto a su propio grupo en comparación con otros grupos).
# 
# Calcular el coeficiente de silueta de un solo ejemplo en nuestro conjunto de datos, podemos aplicar los siguientes tres pasos:
# 
# IMAGEN 10_04
# 
# Puntuación cercana a 1:
# 
# • Un valor cercano a 1 indica que el punto de datos está bien agrupado dentro de su propio grupo y está claramente separado de otros grupos.
# 
# • Esto sugiere que el punto de datos es similar a otros puntos en su grupo y diferente de puntos en grupos vecinos. Es señal de que el cúmulo es compacto y bien definido.
# 
# Puntuación de 0:
# 
# • Un valor de 0 indica que el punto de datos está en el límite entre dos grupos.
# 
# • Esto significa que el punto de datos está equidistante entre su propio grupo y el punto de datos más cercano. En este caso, el punto podría pertenecer a cualquiera de los grupos, lo que sugiere que los racimos no están bien separados.
# 
# Puntuación cercana a -1:
# 
# • Un valor cercano a -1 indica que el punto de datos probablemente esté mal asignado a su grupo actual.
# 
# • Esto sugiere que el punto de datos es más similar a un grupo vecino que al cluster al que ha sido asignado. Es una señal de que el cluster está borroso o que los datos del punto podrían estar en el grupo equivocado.
# 
# El coeficiente de silueta está disponible como silhouette_samples del módulo métrico scikit-learn.
# 
# La función silhouette_scores calcula el promedio coeficiente de silueta en todos los ejemplos, que es equivalente a numpy.mean (silhouette_samples (...)).




km = KMeans(n_clusters=3, 
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
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, 
            color="red", 
            linestyle="--") 
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()


# Los coeficientes de silueta no están cerca de 0 y están aproximadamente igualmente alejados del promedio de puntuación de silueta, que es, en este caso, un indicador de una buena agrupación. Además, para resumir la bondad de nuestra agrupación, agregamos el coeficiente de silueta promedio a la gráfica (línea de puntos).
# 
# Ejemplo de una mala agrupación:



km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], 
            km.cluster_centers_[:, 1],
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




cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
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
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()


# Las siluetas son visiblemente diferentes en cuanto a longitudes y anchos, lo que es evidencia de un agrupamiento relativamente malo o subóptimo.

# ## Organizar clústeres como un árbol jerárquico

# Examinaremos un enfoque alternativo al clustering basado en prototipos: agrupamiento jerárquico. Organiza los datos en una estructura jerárquica, similar a un árbol, donde cada nivel de la jerarquía representa una agrupación diferente de los datos. Esta estructura permite observar cómo se agrupan los datos en diferentes niveles de semejanza.
# 
# Una ventaja del algoritmo de agrupamiento jerárquico es que permite trazar dendrogramas (visualizaciones de una jerarquía binaria de agrupamiento), que puede ayudar con la interpretación de los resultados.
# 
# Otra ventaja de este enfoque jerárquico es que no necesitamos para especificar el número de grupos por adelantado.
# 
# Los dos enfoques principales para la agrupación jerárquica son agrupamiento jerárquico aglomerativo y divisivo.
# 
# En el agrupamiento jerárquico divisivo, comenzamos con un grupo que abarca el conjunto de datos completo, y dividimos iterativamente el agruparse en grupos más pequeños hasta que cada grupo solo contenga un ejemplo.
# 
# En el agrupamiento jerárquico aglomerativo, tomamos lo opuesto. Comenzamos con cada ejemplo como un grupo individual y fusionamos los pares de clusters más cercanos hasta que solo quede un cluster.
# 
# Los dos algoritmos estándar para la jerarquía aglomerativa y la agrupación son el enlace simple y el enlace completo.
# 
# Utilizando un enlace simple, calculamos las distancias entre los miembros más similares (o más cercanos) para cada par de grupos y fusionar los dos grupos para los cuales la distancia entre los los miembros más cercanos son los más pequeños.
# 
# El enfoque de vinculación completa es similar al de vinculación única pero, en lugar de comparar los miembros más cercanos en cada par de grupos, comparamos los miembros más diferentes (o más lejanos) con realizar la fusión.
# 
# IMAGEN 10_05
# 
# Nos centraremos en la agrupación aglomerativa utilizando el sistema completo de enfoques de vinculación. Es un procedimiento iterativo que se puede resumir en los siguiente pasos:
# 
# 1. Calcular una matriz de distancias por pares de todos los ejemplos.
# 
# 2. Representar cada punto de datos como un grupo único.
# 
# 3. Fusionar los dos grupos más cercanos según la distancia entre los miembros más diferentes (distantes).
# 
# 4. Actualizar la matriz de vinculación del clúster.
# 
# 5. Repetir los pasos 2 a 4 hasta que quede un solo grupo.



np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
df


# Calcular la matriz de distancias de todos los ejemplos equivale a calcular la distancia euclidiana entre cada par de ejemplos de entrada en nuestro conjunto de datos.



row_dist = pd.DataFrame(squareform(
                        pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)
row_dist


# La matriz de vinculación consta de varias filas donde cada fila representa una fusión. Las primeras dos columnas denotan los miembros más diferentes en cada grupo, y la tercera columna informa la distancia entre esos miembros. La última columna devuelve el recuento de miembros de cada grupo.
# 
# IMAGEN 10_06



row_clusters = linkage(df.values, 
                       method='complete', 
                       metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['row label 1', 
                      'row label 2',
                      'distance', 
                      'no. of items in clust.'],
             index=[f'cluster {(i + 1)}' for i in 
                    range(row_clusters.shape[0])])


# Ahora que hemos calculado la matriz de vinculación, podemos visualizar los resultados en forma de dendrograma, donde la altura de las uniones indica la distancia o disimilitud entre los clústeres fusionados



# make dendrogram black (part 1/2)
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters, 
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


# En aplicaciones prácticas, los dendrogramas de agrupamiento jerárquico se utilizan a menudo en combinación con un mapa de calor, que nos permite representar los valores individuales en la matriz o matriz de datos que contiene nuestros ejemplos de entrenamiento con un código de color.



fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, 
                       orientation='left')
# note: for matplotlib < v1.5.1, please use
# orientation='right'
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()


# Usando scikit-learn:



ac = AgglomerativeClustering(n_clusters=3,
                             metric='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')




ac = AgglomerativeClustering(n_clusters=2,
                             metric='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')


# ## Localización de regiones de alta densidad mediante DBSCAN

# Vamos a incluir un enfoque más para la agrupación: agrupación espacial de aplicaciones con ruido basada en densidad (DBSCAN).
# 
# Como su nombre lo indica, la agrupación basada en densidad asigna etiquetas basadas en regiones densas de puntos.
# 
# En DBSCAN, la noción de densidad se define como el número de puntos dentro de un radio específico, 𝜀.
# 
# Según el algoritmo DBSCAN, se asigna una etiqueta especial a cada ejemplo (punto de datos) utilizando los siguientes criterios:
# 
# • Un punto se considera un punto central si al menos un número específico (MinPts) de puntos vecinos se encuentran dentro del radio especificado, 𝜀.
# 
# • Un punto fronterizo es un punto que tiene menos vecinos que MinPts dentro de 𝜀 pero se encuentra dentro del radio 𝜀 de un punto central.
# 
# • Se consideran todos los demás puntos que no son centrales ni puntos de ruido fronterizos.
# 
# Después de etiquetar los puntos como núcleo, borde o ruido, el algoritmo DBSCAN se puede resumir en dos simples pasos:
# 
# 1. Forma un grupo separado para cada punto central o grupo conectado de núcleos (los puntos centrales están conectados si no están más lejos que 𝜀.)
# 
# 2. Asigna cada punto fronterizo al grupo de su punto central correspondiente.
# 
# IMAGEN 10_07



X, y = make_moons(n_samples=200, 
                  noise=0.05, 
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()




f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0], 
            X[y_km == 0, 1],
            edgecolor='black',
            c='lightblue', 
            marker='o', 
            s=40, 
            label='cluster 1')
ax1.scatter(X[y_km == 1, 0], 
            X[y_km == 1, 1],
            edgecolor='black',
            c='red', 
            marker='s', 
            s=40, 
            label='cluster 2')
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
ax2.set_title('Agglomerative clustering')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()




db = DBSCAN(eps=0.2, 
            min_samples=5, 
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], 
            X[y_db == 0, 1],
            c='lightblue', 
            marker='o', 
            s=40,
            edgecolor='black', 
            label='Cluster 1')
plt.scatter(X[y_db == 1, 0], 
            X[y_db == 1, 1],
            c='red', 
            marker='s', 
            s=40,
            edgecolor='black', 
            label='Cluster 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()


# ## Convertir Jupyter Notebook a Fichero Python




