# coding: utf-8


import sys
# * from python_environment_check import check_packages
from python_environment_check import check_packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

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
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# # Chapter 5 - Compressing Data via Dimensionality Reduction

# ### Overview

# - [Unsupervised dimensionality reduction via principal component analysis](#Unsupervised-dimensionality-reduction-via-principal-component-analysis)
#   - [The main steps behind principal component analysis](#The-main-steps-behind-principal-component-analysis)
#   - [Extracting the principal components step-by-step](#Extracting-the-principal-components-step-by-step)
#   - [Total and explained variance](#Total-and-explained-variance)
#   - [Feature transformation](#Feature-transformation)
#   - [Principal component analysis in scikit-learn](#Principal-component-analysis-in-scikit-learn)
#   - [Assessing feature contributions](#Assessing-feature-contributions)
# - [Supervised data compression via linear discriminant analysis](#Supervised-data-compression-via-linear-discriminant-analysis)
#   - [Principal component analysis versus linear discriminant analysis](#Principal-component-analysis-versus-linear-discriminant-analysis)
#   - [The inner workings of linear discriminant analysis](#The-inner-workings-of-linear-discriminant-analysis)
#   - [Computing the scatter matrices](#Computing-the-scatter-matrices)
#   - [Selecting linear discriminants for the new feature subspace](#Selecting-linear-discriminants-for-the-new-feature-subspace)
#   - [Projecting examples onto the new feature space](#Projecting-examples-onto-the-new-feature-space)
#   - [LDA via scikit-learn](#LDA-via-scikit-learn)
# - [Nonlinear dimensionality reduction techniques](#Nonlinear-dimensionality-reduction-techniques)
#   - [Visualizing data via t-distributed stochastic neighbor embedding](#Visualizing-data-via-t-distributed-stochastic-neighbor-embedding)
# - [Summary](#Summary)



# * from IPython.display
# Import from the display submodule of the IPython package. This module is designed to display 
# and render different types of data within interactive environments, such as Jupyter Notebooks.
# * import Image
# Import the Image class from the display module. The Image class is used to display 
# images in the interactive environment (for example, in a Jupyter Notebook cell).
# * %matplotlib inline
# This is a magic command specific to IPython/Jupyter Notebook.
# Enables display of matplotlib plots directly within cells of the 
#notebook. Graphics are rendered "inline" (within the same notebook) without the need 
# to open pop-up windows.



# # Unsupervised dimensionality reduction via principal component analysis

# ## The main steps behind principal component analysis



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/05_01.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/05_01.png', which is a relative path to the current directory.
# * width=400
# Set the image width to 400 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Extracting the principal components step-by-step




# Download the wine dataset from the UCI Machine Learning Repository.
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# If the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

# Show the first five rows of the dataset.
df_wine.head()




# Show the shape of the dataset.
df_wine.shape



# Splitting the data into 70% training and 30% test subsets.




# Splitting the dataset into features and target variable, where the target variable is the first column.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# Splitting the dataset into training (70%) and test (30%) sets.
# stratify=y: Makes sure that the training and test datasets have the same class proportions.
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)


# Standardizing the data.




# Standarize the features.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# ---
# 
# **Note**
# 
# Accidentally, I wrote `X_test_std = sc.fit_transform(X_test)` instead of `X_test_std = sc.transform(X_test)`. In this case, it wouldn't make a big difference since the mean and standard deviation of the test set should be (quite) similar to the training set. However, as you remember from Chapter 3, the correct way is to re-use parameters from the training set if we are doing any kind of transformation -- the test set should basically stand for "new, unseen" data.
# 
# My initial typo reflects a common mistake which is that some people are *not* re-using these parameters from the model training/building and standardize the new data "from scratch." Here is a simple example to explain why this is a problem.
# 
# Let's assume we have a simple training set consisting of 3 examples with 1 feature (let's call this feature "length"):
# 
# - train_1: 10 cm -> class_2
# - train_2: 20 cm -> class_2
# - train_3: 30 cm -> class_1
# 
# mean: 20, std.: 8.2
# 
# After standardization, the transformed feature values are
# 
# - train_std_1: -1.21 -> class_2
# - train_std_2: 0 -> class_2
# - train_std_3: 1.21 -> class_1
# 
# Next, let's assume our model has learned to classify examples with a standardized length value < 0.6 as class_2 (class_1 otherwise). So far so good. Now, let's say we have 3 unlabeled data points that we want to classify:
# 
# - new_4: 5 cm -> class ?
# - new_5: 6 cm -> class ?
# - new_6: 7 cm -> class ?
# 
# If we look at the "unstandardized "length" values in our training datast, it is intuitive to say that all of these examples are likely belonging to class_2. However, if we standardize these by re-computing standard deviation and mean you would get similar values as before in the training set and your classifier would (probably incorrectly) classify examples 4 and 5 as class_2.
# 
# - new_std_4: -1.21 -> class_2
# - new_std_5: 0 -> class_2
# - new_std_6: 1.21 -> class_1
# 
# However, if we use the parameters from your "training set standardization," we'd get the values:
# 
# - example5: -18.37 -> class_2
# - example6: -17.15 -> class_2
# - example7: -15.92 -> class_2
# 
# The values 5 cm, 6 cm, and 7 cm are much lower than anything we have seen in the training set previously. Thus, it only makes sense that the standardized features of the "new examples" are much lower than every standardized feature in the training set.
# 
# ---

# Eigendecomposition of the covariance matrix.




# Calculate the covariance matrix.
cov_mat = np.cov(X_train_std.T)

# Calculate the eigenvalues and eigenvectors of the covariance matrix.
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n', eigen_vals)
print('\nEigenvectors \n', eigen_vecs)


# **Note**: 
# 
# Above, I used the [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) function to decompose the symmetric covariance matrix into its eigenvalues and eigenvectors.
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     This is not really a "mistake," but probably suboptimal. It would be better to use [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html) in such cases, which has been designed for [Hermetian matrices](https://en.wikipedia.org/wiki/Hermitian_matrix). The latter always returns real  eigenvalues; whereas the numerically less stable `np.linalg.eig` can decompose nonsymmetric square matrices, you may find that it returns complex eigenvalues in certain cases. (S.R.)
# 

# ## Total and explained variance



# Calculate the explained variance ratio.
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)






plt.bar(range(1, 14), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('figures/05_02.png', dpi=300)
plt.show()


# ## Feature transformation



# Make a list of (eigenvalue, eigenvector) tuples.
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low.
eigen_pairs.sort(key=lambda k: k[0], reverse=True)




# Construct the projection matrix.
# In this case, we will reduce the high-dimensional feature space to a 2-dimensional feature subspace.
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)


# **Note**
# Depending on which version of NumPy and LAPACK you are using, you may obtain the Matrix W with its signs flipped. Please note that this is not an issue: If $v$ is an eigenvector of a matrix $\Sigma$, we have
# 
# $$\Sigma v = \lambda v,$$
# 
# where $\lambda$ is our eigenvalue,
# 
# 
# then $-v$ is also an eigenvector that has the same eigenvalue, since
# $$\Sigma \cdot (-v) = -\Sigma v = -\lambda v = \lambda \cdot (-v).$$



# Transform the first element of the training dataset using the projection matrix.
X_train_std[0].dot(w)




# Transform the entire training dataset using the projection matrix.
X_train_pca = X_train_std.dot(w)

# Visualize the transformed training dataset in a two-dimensional scatterplot.
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
# plt.savefig('figures/05_03.png', dpi=300)
plt.show()


# ## Principal component analysis in scikit-learn

# **NOTE**
# 
# The following four code cells have been added in addition to the content to the book, to illustrate how to replicate the results from our own PCA implementation in scikit-learn:




pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_




plt.bar(range(1, 14), pca.explained_variance_ratio_, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()




pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)




plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()





# Auxiliar function to plot the decision regions.
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # Setup marker generator and color map.
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

    # Plot class examples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')


# Training logistic regression classifier using the first 2 principal components.




# Initializing the PCA transformer.
pca = PCA(n_components=2)

# Dimensionality reduction.
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Initializing the logistic regression eliminator:
# - ovr = One-vs-Rest.
# - lbfgs = Limited-memory Broyden-Fletcher-Goldfarb-Shanno.
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

# Fitting the logistic regression model on the reducted dataset.
lr = lr.fit(X_train_pca, y_train)




# Plotting the decision regions in the reduced space for the training dataset.
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_04.png', dpi=300)
plt.show()




# Plotting the decision regions in the reduced space for the test dataset.
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_05.png', dpi=300)
plt.show()




# Explained variance ratios.
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# ## Assessing feature contributions



# Calculate the loadings.
loadings = eigen_vecs * np.sqrt(eigen_vals)

# Plot the loadings for the first principal component.
fig, ax = plt.subplots()

ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig('figures/05_05_02.png', dpi=300)
plt.show()




loadings[:, 0]




# Calculate the loadings.
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Plot the loadings for the first principal component.
fig, ax = plt.subplots()

ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig('figures/05_05_03.png', dpi=300)
plt.show()


# # Supervised data compression via linear discriminant analysis

# ## Principal component analysis versus linear discriminant analysis



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/05_06.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/05_06.png', which is a relative path to the current directory.
# * width=400
# Set the image width to 400 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## The inner workings of linear discriminant analysis

# ## Computing the scatter matrices

# Calculate the mean vectors for each class:



np.set_printoptions(precision=4)

# Compute one mean vector for each label.
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')


# Compute the within-class scatter matrix:



# Compute the within-class scatter matrix
d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices

print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')


# Better: covariance matrix since classes are not equally distributed:



print('Class label distribution:',  
      np.bincount(y_train)[1:])




d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
    
print('Scaled within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')


# Compute the between-class scatter matrix:



# Compute the between-class scatter matrix.
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)  # make column vector

d = 13  # number of features
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')


# ## Selecting linear discriminants for the new feature subspace

# Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$:



# Calculate the eigenvalues and eigenvectors of the matrix 𝑺𝑊−1·𝑺𝐵.
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# **Note**:
#     
# Above, I used the [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html) function to decompose the symmetric covariance matrix into its eigenvalues and eigenvectors.
#     <pre>>>> eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)</pre>
#     This is not really a "mistake," but probably suboptimal. It would be better to use [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html) in such cases, which has been designed for [Hermetian matrices](https://en.wikipedia.org/wiki/Hermitian_matrix). The latter always returns real  eigenvalues; whereas the numerically less stable `np.linalg.eig` can decompose nonsymmetric square matrices, you may find that it returns complex eigenvalues in certain cases. (S.R.)
# 

# Sort eigenvectors in descending order of the eigenvalues:



# Make a list of (eigenvalue, eigenvector) tuples.
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low.
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues.
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])




tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, align='center',
        label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid',
         label='Cumulative discriminability')
plt.ylabel('Discriminability ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('figures/05_07.png', dpi=300)
plt.show()




# Construct the projection matrix.
# In this case, we will reduce the high-dimensional feature space to a 2-dimensional feature subspace.
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)


# ## Projecting examples onto the new feature space



# Transform the first element of the training dataset using the projection matrix.
X_train_std[0].dot(w)




# Transform the entire training dataset using the projection matrix.
X_train_lda = X_train_std.dot(w)

# Visualize the transformed training dataset in a two-dimensional scatterplot.
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
plt.savefig('figures/05_08.png', dpi=300)
plt.show()


# ## LDA via scikit-learn




# Initializing the LDA transformer.
lda = LDA(n_components=2)

# Dimensionality reduction.
X_train_lda = lda.fit_transform(X_train_std, y_train)





# Initializing the logistic regression estimator:
# - ovr = One-vs-Rest.
# - lbfgs = Limited-memory Broyden-Fletcher-Goldfarb-Shanno.
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

# Fitting the logistic regression model on the reduced dataset.
lr = lr.fit(X_train_lda, y_train)

# Plotting the decision regions in the reduced space for the training dataset.
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_09.png', dpi=300)
plt.show()




# Dimensionality reduction.
X_test_lda = lda.transform(X_test_std)

# Plotting the decision regions in the reduced space for the test dataset.
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_10.png', dpi=300)
plt.show()


# # Nonlinear dimensionality reduction techniques



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/05_11.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/05_11.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ### Visualizing data via t-distributed stochastic neighbor embedding




# Load the digits dataset.
digits = load_digits()

fig, ax = plt.subplots(1, 4)

for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
    
# plt.savefig('figures/05_12.png', dpi=300)
plt.show() 




# Print the shape of the digits data.
digits.data.shape




# Get the data and the labels.
y_digits = digits.target
X_digits = digits.data





# Fit and transform with a TSNE.
tsne = TSNE(n_components=2,
            init='pca',
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
# plt.savefig('figures/05_13.png', dpi=300)
plt.show()


# # Summary

# ---
# 
# Readers may ignore the next cell.



# Run a command in the terminal from a Python environment (such as a Jupyter Notebook or a 
# script that allows system commands to convert a Jupyter notebook to a file 
# Python script. 
# * !
# This symbol is used in environments such as Jupyter Notebooks to execute system commands 
# operational directly from the notebook. In this case, the command is an execution of a 
# python script.
# * python ../.convert_notebook_to_script.py
# This command runs a Python script called convert_notebook_to_script.py. This file 
# is located in the previous directory (../ indicates that it is one level up in the system 
# files). The purpose of this script is to convert a Jupyter notebook (.ipynb) into a 
# Python script file (.py).
# * --input ch05.ipynb
# This is an option or argument that tells the script what the input file is, in this 
# case, the notebook ch05.ipynb.
# * --output ch05.py
# This option tells the script to save the output (the converted file) with the name 
# ch05.py, which is a Python script.


