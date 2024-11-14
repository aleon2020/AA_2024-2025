# coding: utf-8


import sys
# * from python_environment_check import check_packages
from python_environment_check import check_packages
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



# * import sys
# Import the sys module, which is a Python standard library module.
# This module provides access to variables and functions that interact strongly with the
# Python interpreter, such as manipulating module search path and input/output
# standard, among others.
# * sys.path
# It is a list containing the paths in which the Python interpreter looks for modules when
# you use import. When you try to import a module, Python searches the paths specified in this
# list.
# * sys.path.insert(0, '..')
# Insert the path '..' (representing the parent directory) at the beginning of the sys.path list.
# Adding it in position 0 ensures that when Python looks for modules to import,
# first check in the parent directory before continuing with the default paths.

sys.path.insert(0, '..')


# Check recommended package versions:



# Import the check_packages function from the python_environment_check module. 
# This module, from its name, appears to be designed to verify that the Python environment 
# have the correct versions of certain packages installed.
# * d = {...}
# Defines a dictionary d that contains the names of several packages as keys 
# (e.g. numpy, scipy, matplotlib, etc.) and as values ​​the minimum versions 
# required from those packages.
# * check_packages(d)
# The check_packages function takes as input the dictionary d and probably performs a 
# check on current Python environment to ensure installed versions 
# of these packages are at least those specified in the dictionary. If any of the packages 
# is not installed or has the wrong version, the function may throw an error or 
# suggest installing/updating the packages.

d = {
    'numpy': '1.21.2',
    'scipy': '1.7.0',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2',
}
check_packages(d)


# # Example 12: Hierarchical Tree clustering Exercise

# ### Overview

# - [Importing libraries for data analysis and scaling](#importing-libraries-for-data-analysis-and-scaling)
# - [Extracting the principal components step by step](#extracting-the-principal-components-step-by-step)
# - [Performing hierarchical clustering on a distance matrix](#performing-hierarchical-clustering-on-a-distance-matrix)
# - [Attaching dendrograms to a heat map](#attaching-dendrograms-to-a-heat-map)
# - [Applying agglomerative clustering via scikit-learn](#applying-agglomerative-clustering-via-scikit-learn)
# - [Summary](#summary)

# STATEMENT
# 
# In this exercise, you will perform an unsupervised analysis using agglomerative clustering with complete linkage on the data set provided in the 'dataset1.csv' file.
# 
# To do this, you will use the following instructions:
# 
# - Data normalization: Before applying the algorithm, make sure to normalize the data to improve the accuracy of the analysis.
# 
# - Calculation of clusters with scipy or scikit-learn.
# 
# - Graphic representation of dendrogram and heat map.
# 
# - Graphical representation with PCA: Perform a PCA to reduce the dimensionality of the data and represent the clusters in a two-dimensional graph.



# * from IPython.display
# Import from the display submodule of the IPython package. This module is designed to display 
# and render different types of data within interactive environments, such as Jupyter Notebooks.
# * import Image
# Import the Image class from the display module. The Image class is used to display 
# images in the interactive environment (for example, in a Jupyter Notebook cell).
# * %matplotlib inline
# This is a magic command specific to IPython/Jupyter Notebook.
# Enables display of matplotlib plots directly within cells of the 
# notebook. Graphics are rendered "inline" (within the same notebook) without the need 
# to open pop-up windows.



# ## Importing libraries for data analysis and scaling





# ## Extracting the principal components step by step



# Download the wine dataset from the UCI Machine Learning Repository
df = pd.read_csv('dataset1.csv')




# Show the first five rows of the dataset
df.head()




# Show the shape of the dataset
df.shape




# Splitting the dataset into features and target variable
X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values # Target variable is the first column


# ## Performing hierarchical clustering on a distance matrix



row_clusters = linkage(df.values,
                       method='complete',
                       metric='euclidean')




pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
                      index=[f'cluster {(i + 1)}' for i in range(row_clusters.shape[0])])




row_dendr = dendrogram(
    row_clusters,
    )   

plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


# ## Attaching dendrograms to a heat map



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


# ## Applying agglomerative clustering via scikit-learn



n_clusters = 3
clusters = fcluster(row_clusters, n_clusters, criterion='maxclust')
print(clusters)




ac = AgglomerativeClustering(n_clusters=3,
                             metric='euclidean',
                             linkage='complete')

labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')




sc = StandardScaler()
X_std = sc.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_std)

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


# # Summary

# ---
# 
# Readers may ignore the next cell.



# Run a command in the terminal from a Python environment (such as a Jupyter Notebook or a 
# script that allows system commands to convert a Jupyter notebook to a file Python script. 
# * !
# This symbol is used in environments such as Jupyter Notebooks to execute system commands 
# operational directly from the notebook. In this case, the command is an execution of a 
# Python Script.
# * python ../.convert_notebook_to_script.py
# This command runs a Python script called convert_notebook_to_script.py. This file 
# is located in the previous directory (../ indicates that it is one level up in the system 
# files). The purpose of this script is to convert a Jupyter notebook (.ipynb) into a 
# Python script file (.py).
# * --input ex12.ipynb
# This is an option or argument that tells the script what the input file is, in this 
# case, the notebook ex12.ipynb.
# * --output ex12.py
# This option tells the script to save the output (the converted file) with the name
# ex12.py, which is a Python script.


