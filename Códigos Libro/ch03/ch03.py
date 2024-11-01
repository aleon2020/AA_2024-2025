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



# # Choosing a classification algorithm

# # First steps with scikit-learn

# Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower examples. The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.



# LIBRARY IMPORT
# * sklearn.datasets
# The scikit-learn datasets module is imported, which contains several classic data sets, 
# like Iris, which is used in classification examples.
# * numpy (np)
# The numpy library is imported under the alias np to handle numerical operations.

# LOADING THE IRIS DATA SET
# * load_iris() 
# Load the Iris dataset, a classic dataset that contains features of three types 
# of iris flowers: Setosa, Versicolor, and Virginica. The data set includes 150 
# examples, with 4 characteristics for example (length and width of sepal and petal), and the 
# class labels corresponding to the species.
iris = datasets.load_iris()

# FEATURE SELECTION
# * iris.data
# Contains a matrix of 150 rows and 4 columns, where each row is an example with its 4 
# characteristics.
# * [:, [23]]
# Only columns 2 and 3 are selected, which correspond to the length and width
# of the petal, respectively. Thus, X contains only these two characteristics.
X = iris.data[:, [2, 3]]

# CLASS LABEL ASSIGNMENT
# * iris.target
# Contains the class labels for each example in the data set. The labels 
# are 0, 1, and 2, which correspond to the three species of iris flowers (Setosa, Versicolor, 
# and Virginica).
y = iris.target

# PRINTING THE UNIQUE CLASS LABELS
# * np.unique(y)
# This NumPy function returns the unique values ​​of the labels in y. In this case, 
# will print the unique class labels: [0, 1, 2], which represent the three species of 
# flowers in the Iris dataset.
print('Class labels:', np.unique(y))


# Splitting data into 70% training and 30% test data:



# MODULE IMPORT
# The train_test_split function is imported from the model_selection module of scikit-learn. 
# This function is used to split a data set into two subsets: 
# one for training and one for testing.

# DIVISION OF DATA
# * X
# They are the characteristics (independent variables) of the data set.
# * y
# They are the labels (the dependent or class variable).
# The train_test_split function splits this data into four subsets:
# * X_train
# Set of features for training.
# * X_test
# Feature set for testing.
# * y_train
# Labels corresponding to the training set.
# * y_test
# Labels corresponding to the test set.

# PARAMETERS
# * test_size=0.3
# Indicates that 30% of the data (0.3) will be reserved for the test set, and 70% 
# remaining will be used for training.
# * random_state=1
# Set a random seed to make the split reproducible, i.e. when using 
# the same seed (1), the same division will be obtained every time it is executed.
# * stratify=y
# Ensures that the proportion of classes in y (the labels) is the same in both sets 
# (training and testing). This is useful when classes are unbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)




# PRINTING Y LABEL COUNT
# * np.bincount(y)
# This function counts the number of times each integer value appears in the y array. 
# In this case, y contains the class labels (0, 1, 2) of the entire data set.
# The result is a list in which each position corresponds to a tag, and the value in 
# that position indicates how many times that tag appears in y.
# * Example
# If y has 50 examples of class 0, 50 of class 1, and 50 of class 2, the result 
# will be something like [50, 50, 50].
print('Labels counts in y:', np.bincount(y))

# PRINTING TAG COUNT ON Y_TRAIN
# Similar to the previous case, this line counts the occurrences of each tag in the set 
# training (y_train), which was obtained after splitting the data with train_test_split.
# Since the stratify parameter was used when splitting the data, the tag count in y_train 
# will maintain the proportions of the original classes.
print('Labels counts in y_train:', np.bincount(y_train))

# PRINT LABEL COUNT IN Y_TEST
# Here we count how many times each label appears in the test set (y_test).
# Again, the proportions of the classes in y_test will be the same as in the data 
# originals due to the use of stratify.
print('Labels counts in y_test:', np.bincount(y_test))


# Standardizing the features:



# MODULE IMPORT
# The StandardScaler class is imported from the sklearn.preprocessing module. This class is used 
# to standardize features, that is, make the data have a mean of 0 and a 
# standard deviation of 1.

# CREATION OF A STANDARD SCALER OBJECT
# A StandardScaler instance called sc is created. This object will be used to adjust 
# (calculate the parameters) and transform the data.
sc = StandardScaler()

# FITTING THE CLIMBER ON THE TRAINING SET
# The fit(X_train) method fits the scaler using the training data (X_train). 
# During this process, the StandardScaler calculates the mean and standard deviation of each 
# feature in X_train. These values ​​are stored and will be used later to 
# transform both training and test data.
sc.fit(X_train)

# TRAINING SET TRANSFORMATION
# The transform(X_train) method uses the mean and standard deviation values 
# calculated in the previous step to transform the features of X_train. Each value 
# of X_train features are standardized by subtracting the mean and dividing by the 
# standard deviation. The result is a new data set X_train_std, where the 
# features have a mean of 0 and a standard deviation of 1.
X_train_std = sc.transform(X_train)

#TEST SET TRANSFORMATION
# The same transformation is applied to the test data (X_test) using the parameters 
# of standardization (mean and standard deviation) calculated from the set of 
# training. This ensures that transformations on the test data are 
# consistent with those of the training set. The result is the standardized set 
# X_test_std.
X_test_std = sc.transform(X_test)


# ## Training a perceptron via scikit-learn



#PERCEPTRON IMPORT
# The Perceptron class from the sklearn.linear_model module is imported. The Perceptron is an algorithm 
# Linear classification that updates weights using perceptron learning rule.

# CREATION OF A PERCEPTRON MODEL
# An instance of the Perceptron model is created with the following parameters:
# * eta0=0.1
# Learning rate, which controls how much the weights are adjusted in each iteration.
# * random_state=1
# Set a random seed to ensure reproducibility of results.
ppn = Perceptron(eta0=0.1, random_state=1)

# MODEL TRAINING
# The ppn model is trained with the standardized training data set 
# (X_train_std) and its corresponding tags (y_train).
# The fit method adjusts the model by calculating the weights that best separate the classes in the 
# training data.
ppn.fit(X_train_std, y_train)




# PREDICTION ON THE TEST SET
# The trained model is used to make predictions on the test set 
# standardized (X_test_std), generating the y_pred class predictions.
y_pred = ppn.predict(X_test_std)

# CALCULATION OF MISCLASSIFIED EXAMPLES
# Compare the actual labels (y_test) with the model predictions (y_pred) to count 
# how many examples were misclassified. This is achieved with the expression 
# (y_test != y_pred).sum(), which counts how many values ​​are different between y_test and y_pred.
print('Misclassified examples: %d' % (y_test != y_pred).sum())




# ACCURACY CALCULATION
# Import the accuracy_score function from the sklearn.metrics module to calculate accuracy 
# (the percentage of correctly classified examples) comparing y_test with y_pred.
# Precision is printed to three decimal places using the expression '%.3f'.
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))




# PRECISION USING THE PERCEPTRON SCORE METHOD
# Alternatively, precision is calculated using the Perceptron (ppn) score method, which 
# also returns the accuracy of the model on the test set (X_test_std and y_test).
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))




# IMPORT OF LIBRARIES
# * ListedColormap
# Allows you to create a custom color map for the display.
# * matplotlib.pyplot
# Provides functions for generating graphs.
# * numpy
# It is the library to perform numerical calculations.
# * LooseVersion
# Imported to check matplotlib version compatibility 
# (although not used in this snippet).

# DEFINITION OF THE PLOT_DECISION_REGIONS FUNCTION
# * X
# The data set with two features (columns) to display.
# * y
# The class labels associated with the data.
# * classifier
# The classification model that will be used to predict the classes.
# * test_idx
# Optional indexes to highlight test set examples.
# * resolution
# The resolution of the mesh to draw for the decision regions.
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # CONFIGURATION OF COLORS AND MARKERS
    # Various marker types (o, s, ^, etc.) and colors are defined for each class.
    # The cmap color map is generated so that each class has a unique color, in 
    # function of the number of classes in the data (np.unique(y)).
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # GENERATION OF THE DECISION MESH
    # The range of values ​​of the two characteristics (X[:, 0] and X[:, 1]) is calculated for 
    # create a mesh (grid) of points that covers the entire two-dimensional space. The 
    # limits are extended by 1 unit so that data is not clipped.
    # np.meshgrid is used to create coordinate arrays that cover space.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # PREDICTION ON THE MESH
    # The classifier predictions are applied to each point of the generated mesh.
    # The results are transformed and rearranged to match the shape of the mesh, 
    # allowing you to create a continuous graph of the decision regions.
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    # DRAWING OF THE DECISION REGIONS
    # * contourf
    # Used to draw the decision regions, coloring each region according to the 
    # classifier predictions.
    # The axis limits are adjusted to match the range of values ​​of the 
    # characteristics.
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # VIEWING DATA POINTS
    # The original data points are drawn over the decision regions. each class 
    # has a different color and marker, and the points have a black border to highlight them.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # HIGHLIGHTING TEST SET EXAMPLES
    # If indexes of the test examples (test_idx) are provided, these points are highlighted 
    # on the graph using a large circle with a black border and no fill color (c='none').
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



# COMBINATION OF DATASETS
# * np.vstack
# Used to vertically stack standardized training data sets 
# (X_train_std) and test (X_test_std). This generates a single X_combined_std array containing 
# all data standardized.
# * np.hstack
# Used to horizontally stack class labels in the set 
# training (y_train) and test set (y_test). This creates a single y_combined array 
# containing all class tags.
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function (defined previously) is called to display the 
# decision regions of the Perceptron (ppn) model.
# The parameters passed are:
# * X=X_combined_std
# The standardized data combined.
# * y=y_combined
# The combined tags.
# * classifier=ppn
# The classifier used to predict the classes.
# * test_idx=range(105, 150)
# Indicates which examples from the test set should be highlighted in the visualization. 
#Indices 105 to 149 are being highlighted here.
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

# AXLE LABELING
# Labels are added to the X and Y axes of the graph, representing the characteristics of 
# length and width of standardized petals.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')

# LEGEND
# A legend is added to the chart in the upper left corner to identify the 
# different classes.
plt.legend(loc='upper left')

# DESIGN SETTINGS
# This function automatically adjusts the spacing of the chart so that elements do not overlap. 
# overlap and look more organized.
plt.tight_layout()

# SHOW THE GRAPH
# The line that saves the graph as a PNG file is commented out. If it is uncommented, the 
# graphic would be saved to the specified path with a resolution of 300 dpi.
# * plt.show()
# Display the generated graph in a popup window.
# plt.savefig('figures/03_01.png', dpi=300)
plt.show()


# # Modeling class probabilities via logistic regression

# ### Logistic regression intuition and conditional probabilities



# LIBRARY IMPORT
# * matplotlib.pyplot
# Used to create graphs in Python.
# * numpy
# Used to perform numerical calculations and manipulate arrays.

# DEFINITION OF THE SIGMOID FUNCTION
# The sigmoid function is defined, which takes a value (or an array of values) z and applies the 
# formula for the sigmoid function. The output is in the range of (0, 1), which makes it useful 
# to interpret as probabilities.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# GENERATION OF VALUES FOR Z
# An array z containing values ​​from -7 to 7 (not including 7) is created, with a step of 
#0.1. This range covers most of the interesting behavior of the sigmoid function.
z = np.arange(-7, 7, 0.1)

# CALCULATION OF THE SIGMOID FUNCTION
# The sigmoid function is applied to all values ​​of z and the result is stored in sigma_z.
sigma_z = sigmoid(z)

# CHART CREATION
# * plt.plot(z, sigma_z)
# The graph of the sigmoid function is plotted with z on the X axis and sigma_z on the Y axis.
# * plt.axvline(0.0, color='k')
# A vertical line is drawn at z = 0 (in black), which helps visualize the point 
# in which the sigmoid function is 0.5.
# * plt.ylim(-0.1, 1.1)
# Y axis limits are set to make the graph look cleaner.
# * plt.xlabel('z') and plt.ylabel('$\sigma (z)$')
# Labels are added to the X and Y axes, respectively. Y axis label uses notation 
# math to show the sigmoid function.
plt.plot(z, sigma_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')

# CONFIGURATION OF GRID MARKS AND LINES
# * plt.yticks([0.0, 0.5, 1.0])
# Set the Y axis ticks to only show 0.0, 0.5, and 1.0.
# * ax.yaxis.grid(True)
# Grid lines on the Y axis are enabled to improve chart readability.
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

# LAYOUT SETTING AND DISPLAY
# * plt.tight_layout()
# Adjust the layout of the chart so that there are no overlaps.
# The line that saves the graph as a PNG file is commented out. If uncommented, 
# the graph would be saved to the specified path with a resolution of 300 dpi.
# * plt.show()
# Display the generated graph in a popup window.
plt.tight_layout()
# plt.savefig('figures/03_02.png', dpi=300)
plt.show()




# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_03.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_03.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_25.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_25.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ### Learning the weights of the logistic loss function



# DEFINITION OF LOSS FUNCTIONS
# * loss_1(z)
# Calculate the loss function when the class label is 1. Use the sigmoid function and cross 
# entropy loss formula for positive classifications.
# * loss_0(z)
# Calculate the loss function when the class label is 0. Similarly, use the cross entropy 
# loss formula for negative classifications.
def loss_1(z):
    return - np.log(sigmoid(z))
def loss_0(z):
    return - np.log(1 - sigmoid(z))

# GENERATION OF VALUES FOR Z
# An array z containing values ​​from -10 to 10 (not including 10) is created, with a step of 
#0.1. This allows us to observe how the loss functions behave in a wide range of 
# values.
# sigma_z is calculated, which represents the values ​​of the sigmoid function evaluated in the range 
# of z.
z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)

# CALCULATION OF THE LOSS FOR WHEN Y=1
# A list comprehension is used to calculate the loss using loss_1 for each value in 
#z. The result is stored in c1.
# Next, c1 is plotted as a function of sigma_z, labeling the line L(w, b) if y=1.
c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='L(w, b) if y=1')

# CALCULATION OF THE LOSS FOR WHEN Y=0
# Similar to the previous step, the loss is calculated using loss_0 for each value in z and 
# stored in c0.
# Plot c0 as a function of sigma_z, using a dashed line style (linestyle='--') 
# and labeling it L(w, b) if y=0.
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')

# SETTING CHART LIMITS AND LABELS
# * plt.ylim(0.0, 5.1)
# Set the Y axis limit between 0 and 5.1 to focus on calculated losses.
# * plt.xlim([0, 1])
# Sets the limit of the X axis between 0 and 1, which is the range of the sigmoid function.
# Labels are added to the X and Y axes, showing that the X axis represents the output of the 
# sigmoid function and the Y axis represents the loss.
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')

# ADDING LEGEND AND ADJUSTING THE LAYOUT
# * plt.legend(loc='best')
# Add a legend to the chart in the best available position.
# * plt.tight_layout()
# Automatically adjusts the chart layout so there are no overlaps.
# The line that saves the graph as a PNG file is commented out. If uncommented, 
# the graph would be saved to the specified path with a resolution of 300 dpi.
# * plt.show()
# Display the generated graph in a popup window.
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('figures/03_04.png', dpi=300)
plt.show()




# DEFINITION OF THE LOGISTICREGRESSIONGD CLASS
# This class implements a logistic regression classifier based on the algorithm 
# gradient descent. The docstring provides details about the parameters and attributes 
# of the class.
class LogisticRegressionGD:

    # INITIALIZATION PARAMETERS
    # * eta
    # Learning rate that determines the step size in each iteration of the descent 
    # of gradient.
    # * n_iter
    # Number of passes (epochs) over the training set.
    # * random_state
    # Seed for the random number generator, which is used to 
    # the initialization of weights.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # ADJUSTMENT METHOD
    # This method fits the model to the training data.
    # - X: Array of input features (n_examples, n_features).
    # - y: Vector of target labels (n_examples).
    # Inside the fit method:
    # - Initializes the weights and bias with small random values.
    # - Create a list to store the loss during training.
    # For each iteration:
    # - Calculates the net input (net_input).
    # - Apply the sigmoid function to obtain the model output (activation).
    # - Calculates the errors (difference between the expected output and the model output).
    # - Updates the weights and bias using the gradient descent update rule.
    # - Calculates the loss using the log loss function and stores it.
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

    # NET INPUT METHOD
    # Calculates the net input of the model, which is the dot product of the characteristics of 
    # input with the weights plus the bias.
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # ACTIVATION FUNCTION
    # Applies the sigmoid function to the net input, which transforms the values ​​into a range 
    # between 0 and 1. Use np.clip to avoid numeric overflows.
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    # PREDICTION METHOD
    # Returns the class label (0 or 1) for the input samples. If the activation is 
    # greater than or equal to 0.5, returns 1; otherwise it returns 0.
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




# DATA FILTERING
# * X_train_01_subset
# Create a subset of the training features (X_train_std) that includes only 
# the samples where the label (y_train) is 0 or 1. This is done to simplify the problem 
# classification to a binary case, removing the extra classes.
# * y_train_01_subset
# Similarly, create a label vector containing only the corresponding labels 
# to classes 0 and 1.
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# MODEL INSTALLATION AND TRAINING
#*lrgd
# An object of the LogisticRegressionGD class with a learning rate (eta) is instantiated 
# of 0.3 and a number of iterations (n_iter) of 1000. The random_state is set to 1 to 
# guarantee the reproducibility of the results.
# * fit
# The fit method is called to fit the model to the subset training data 
# (0 and 1).
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

# VISUALIZATION OF DECISION REGIONS
# * plot_decision_regions
# This function displays the decision regions of the trained classifier. Shows how the 
# model divides the feature space between classes 0 and 1.
plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

# TAGS, DISPLAY AND LEGEND
# Labels are added to the X and Y axes to indicate that they represent the length and width of 
# the petals, respectively. A legend is included in the upper left corner.
# * tight_layout
# Adjust the layout so that graph elements do not overlap.
# * show
# Shows the generated graph.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_05.png', dpi=300)
plt.show()


# ### Training a logistic regression model with scikit-learn



# MODEL IMPORT
# The LogisticRegression class is imported from scikit-learn, which allows creating a classifier 
# based on logistic regression.

# INSTANTATION OF THE CLASSIFIER
# * C=100.0
# This inverse regularization parameter controls the amount of regularization applied 
# to the model. A high value like 100.0 means little regularization, which can allow 
# make the model fit better to the training data.
# * solver='lbfgs'
# Specifies the algorithm to use to optimize the loss function. lbfgs is a method 
# which uses quasi-Newton approximations.
# * multi_class='ovr'
# Sets the approach for handling multi-class classification problems. The 'ovr' method 
# (One-vs-Rest) trains a classifier for each class against the others.
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')

# MODEL TRAINING
# The lr model is trained using the X_train_std data set (features 
# standardized) and y_train (class tags). The model adjusts the internal parameters 
# to minimize the loss function on the training set.
lr.fit(X_train_std, y_train)

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function is called, which displays the decision regions 
# generated by the trained lr model. This shows how the model classifies the space of 
# features based on the classes in y_combined.
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))

# TAGS, DISPLAY AND LEGEND
# Labels are added to the X and Y axes to indicate that they represent length and width 
# of standardized petals.
# A legend is included in the upper left corner.
# * tight_layout
# Adjust the layout so that graph elements do not overlap.
# * show
# Show the generated graph
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_06.png', dpi=300)
plt.show()




# DESCRIPTION
# This line calls the predict_proba method of the lr classifier, which returns the probabilities 
# of class for the samples in X_test_std.
# DETAILS
# X_test_std[:3, :] selects the first 3 samples from the test data set 
# standardized.
# The result is an array of probabilities in which each row corresponds to a sample 
# and each column to the probability of belonging to each class.

lr.predict_proba(X_test_std[:3, :])




# DESCRIPTION
# This line calculates the sum of the predicted probabilities for each of the 3 samples.
# DETAILS
# sum(axis=1) sums the probabilities across the columns (i.e., for each sample).
# The result must be an array with a value of 1 for each sample, since the probabilities 
# of membership in all classes must add up to 1.

lr.predict_proba(X_test_std[:3, :]).sum(axis=1)




# DESCRIPTION
# This line finds the index of the class with the highest probability for each of the 3 
# samples.
# DETAILS
# argmax(axis=1) returns the index of the class with the highest probability for each row 
# (sample).
# This allows us to identify which is the most probable class according to the model for each of the 
# samples.

lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)




# DESCRIPTION
# This line predicts the classes for the first 3 samples in the test data set 
# standardized.
# DETAILS
# The predict method returns the predicted classes directly, based on the probabilities 
# calculated.
# Uses a threshold of 0.5 to classify whether a sample belongs to the positive class or 
# negative, or use the index of the class with the highest probability in the case of multiple 
# classes.

lr.predict(X_test_std[:3, :])




# DESCRIPTION
# This line predicts the class of a single (first) sample of the test data set 
# standardized.
# DETAILS
# X_test_std[0, :].reshape(1, -1) reshapes the first sample to have the shape 
# appropriate that the predict method expects.
# Using reshape(1, -1) converts the sample to a one-row array with the 
# features in columns, which is necessary for the model to work correctly 
# with a single sample.

lr.predict(X_test_std[0, :].reshape(1, -1))


# ### Tackling overfitting via regularization



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_07.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_07.png', which is a relative path to the current directory.
# * width=700
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# INITIALIZATION OF LISTS
# Two empty lists are created: weights to store the model coefficients and params to 
# store the C values.
weights, params = [], []

# LOOP OVER C VALUES
# It iterates over a range of c values ​​from -5 to 5.
# For each value of c:
# - A logistic regression model is defined with the regularization parameter C = 10^C. 
# This parameter controls regularization: smaller values ​​of C increase the 
# regularization, while larger values ​​allow more complexity in the model.
# - Fit the model using standardized training data 
# (X_train_std and y_train).
# - The model coefficients are stored (lr.coef_[1] which corresponds to the second class 
# in a binary classification model) in the weights list.
# - The value of C (calculated as 10^C) is also added to the params list.
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

# CONVERSION TO NUMPY ARRAY
# Convert the list of weights to a Numpy array to facilitate data management.
weights = np.array(weights)

# DISPLAY RESULTS
# The coefficients of the characteristics (length and width of the petal) are plotted in 
# function of C values.
# plt.plot is used to plot the coefficients corresponding to the length of the 
# petal and the width of the petal.
# Labels are added to the axes and a legend to identify each line.
# Set the x-axis scale to logarithmic (plt.xscale('log')) for better visualization 
# changes in coefficients over several orders of magnitude.
# Finally, the graph is displayed.
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
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_09.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_09.png', which is a relative path to the current directory.
# * width=700
# Set the image width to 700 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Maximum margin intuition

# ## Dealing with the nonlinearly separable case using slack variables



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_10.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_10.png', which is a relative path to the current directory.
# * width=600
# Set the image width to 600 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# SVC CLASS IMPLEMENTATION
# The SVC (Support Vector Classifier) ​​class is imported from the sklearn library, which allows creating 
# classification models using Support Vector Machines.

# SVM MODEL CREATION
# An instance of the SVM classifier is created
# * kernel='linear'
# It is specified that a linear kernel will be used, which means that a boundary of 
# linear decision between classes.
# * C=1.0
# This regularization parameter controls the balance between a wider margin and the 
# classification of data points. A higher C value attempts to classify all 
# data points correctly, while a lower value allows a wider margin, 
# even if it means some points are misclassified.
# * random_state=1
# This parameter is set to ensure the reproducibility of the results,
# especially in random model initialization.
svm = SVC(kernel='linear', C=1.0, random_state=1)

# MODEL TRAINING
# The SVM model is fitted to the training data (X_train_std and y_train), where:
# * X_train_std
# Characteristics of standardized training data.
# * y_train
# Labels of the classes corresponding to the training data.
svm.fit(X_train_std, y_train)

# VISUALIZATION OF THE DECISION BORDER
# The plot_decision_regions function is called, which plots the decision boundary of the 
# SVM classifier.
# * X_combined_std
# Combined data (both training and testing) standardized for visualization.
# * y_combined
# Class labels corresponding to the combined data.
# * classifier=svm
# The trained SVM classifier is passed so that the function can use it to predict the 
# class and graph the decision boundary.
# * test_idx=range(105, 150)
# Indices of the test points to be highlighted in the display.
plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))

# LABELS AND AESTHETICS OF THE GRAPHIC
# Axis labels (xlabel and ylabel) are set to describe the characteristics 
# used in the graph.
# A legend is added to identify the classes in the graph.
# plt.tight_layout() adjusts the layout of the chart so that elements do not overlap.
# Finally, plt.show() displays the graph on the screen. The line commented out with plt.savefig 
# suggests that you could save the generated figure as a PNG file.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_11.png', dpi=300)
plt.show()


# ## Alternative implementations in scikit-learn



# SGDCLASSIFIER CLASSIFIER IMPORT
# The SGDClassifier class is imported from the sklearn library, which allows implementing different 
# Classification algorithms using the stochastic gradient descent method.

# CREATION OF A PERCEPTRON CLASSIFIER
# An instance of the classifier is created with the Perceptron loss function:
# * loss='perceptron'
# Specifies to use the perceptron learning rule, a linear classifier that is 
# train using online learning technique.
ppn = SGDClassifier(loss='perceptron')

# CREATION OF A LOGISTIC REGRESSION CLASSIFIER
# An instance of the classifier is created with the Logistic Regression loss function:
# * loss='log'
# Indicates that the logarithmic loss function will be used, suitable for classification 
# binary, meaning the model will be adjusted to maximize the probability of the 
# classes using logistic regression.
lr = SGDClassifier(loss='log')

# CREATION OF AN SVM CLASSIFIER
# The classifier is instantiated with the Hinge loss function:
# * loss='hinge'
# Uses the hinge loss function, which is commonly used in Support Machines 
# Vector (SVM) for linear classifications. This type of classifier seeks to maximize 
# the margin between classes.
svm = SGDClassifier(loss='hinge')


# # Solving non-linear problems using a kernel SVM



# LIBRARY IMPORT
# The necessary libraries are imported: matplotlib.pyplot to create plots and numpy to 
# manipulate numeric data.

# RANDOM SEED SETUP
# A seed is set for numpy's random number generator to ensure consistency. 
# reproducibility of the results, so that the same random numbers are obtained in 
# each execution.
np.random.seed(1)

# RANDOM DATA GENERATION
# 200 random data points are generated in two-dimensional space (2 features) 
# using a standard normal distribution (mean 0 and standard deviation 1).
X_xor = np.random.randn(200, 2)

# CREATION OF XOR TAGS
# The np.logical_xor function is used to assign labels (classes) to data points. 
# Class 1 is assigned to points where one of the coordinates is positive and the other is 
# negative, while class 0 is assigned where both coordinates are equal 
# (both positive or both negative).
# The result is a y_xor array containing 1s and 0s, representing the two classes.
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
print(X_xor[1])
print(y_xor[1])

# DATA VISUALIZATION
# Data points are plotted in a scatter plot.
# Class 1 points (labeled 1 in y_xor) are represented with colored squares 
# blue ('royalblue').
# Points of class 0 (labeled 0 in y_xor) are represented with colored circles 
# red ('tomato').
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

# AXIS CONFIGURATION AND LEGEND
# The limits of the x and y axes are set.
# The axes are labeled.
# A legend is added to identify the classes.
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()

# FINAL DISPLAY
# Uncomment the line (if desired) to save the graph as a PNG image.
# Finally, the graph is shown with plt.show().
# plt.savefig('figures/03_12.png', dpi=300)
plt.show()




# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_13.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_13.png', which is a relative path to the current directory.
# * width=700
# Set the image width to 700 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Using the kernel trick to find separating hyperplanes in higher dimensional space



# CREATION OF THE SVM CLASSIFIER
# An SVC (Support Vector Classification) object is created with the following parameters:
# * kernel='rbf'
# Uses a radial kernel (Radial Basis Function), which is suitable for nonlinear problems 
# like XOR. This type of kernel allows the SVM to find more complex decision boundaries.
# * random_state=1
# Set a seed for the random number generator to ensure reproducibility model.
# * gamma=0.10
# This parameter controls the influence of a single training example. Higher value 
# of gamma implies that the model will consider points closer to the support vectors, which 
# which can lead to greater complexity in the model.
# * C=10.0
# This parameter controls the penalty for classification errors. A high C value 
# means that errors are penalized more heavily, which can result in a model 
# more complex.
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)

# MODEL TRAINING
# The SVM classifier is trained using the X_xor data set (the features) 
# and y_xor (the class labels). The model adjusts its internal parameters to be able to separate 
# the two kinds of data.
svm.fit(X_xor, y_xor)

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function is called, which was probably defined in a block of 
# code above. This function generates a graph showing the decision regions of the 
# trained SVM classifier. Each region in the graph represents a different class based 
# in the model predictions. Data points are plotted along with regions 
# colored indicating the predicted classes.
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

# LEGEND SETTING AND DISPLAY
# A legend is added to the upper left corner of the graph to identify the classes.
# plt.tight_layout() adjusts the chart elements so that they do not overlap.
# Finally, the graph is shown with plt.show(). If the line plt.savefig(...) is uncommented, 
# the graph would be saved as a PNG file.
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_14.png', dpi=300)
plt.show()




# SVC CLASS IMPORT
# The SVC class is imported from sklearn.svm, which is used to create a supporting classifier 
# vector.

# CREATION OF THE SVM CLASSIFIER
# An object of class SVC is initialized with the following parameters:
# * kernel='rbf'
# Specifies that a radial basis function (RBF) kernel will be used. This type of core is 
# suitable for nonlinear problems, as it allows the model to find limits of 
# more complex decision.
# * random_state=1
# Sets a seed for the random number generator, ensuring that the 
# results are reproducible.
# * gamma=0.2
# This parameter controls the extent of influence of a single support vector. A value of 
# lower gamma implies that the model will be smoother and may be more general, while a 
# Higher value may result in a tighter model that better fits the data 
# training.
# * C=1.0
# This parameter determines the penalty for classification errors. Higher C value 
# tends to create a model that fits the training data more closely, while a value 
# may allow for more errors, favoring greater generalization.
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)

# MODEL TRAINING
# The SVM model is trained using the training data set X_train_std 
# (standardized features) and y_train (class tags). The model adjusts its 
# internal parameters based on training data.
svm.fit(X_train_std, y_train)

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function is called, which displays the decision regions of the 
# trained SVM classifier. It uses X_combined_std and y_combined, which probably 
# contain both training and test data, and test_idx is used to 
# highlight specific test examples on the chart.
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))

# LABEL AND DISPLAY SETUP
# Labels are added to the X and Y axes of the chart to indicate which features are being 
# representing (length and width of petals).
# A legend is added in the upper left corner to identify the classes.
# plt.tight_layout() adjusts chart elements to avoid overlaps.
# Finally, plt.show() shows the graph. The line that is commented out (plt.savefig(...)) 
# would be used to save the graphic as a PNG file if uncommented.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_15.png', dpi=300)
plt.show()




# CREATION OF THE SVM CLASSIFIER
# An object of the SVC (Support Vector Classifier) ​​class is initialized with the following 
# parameters:
# * kernel='rbf'
# Uses a radial basis function (RBF) kernel. This core is suitable for problems not 
# linear, allowing the classifier to find complex decision boundaries.
# * random_state=1
# Set the seed of the random number generator to ensure that the results are 
# reproducible.
# * gamma=100.0
# This parameter determines the extent of influence of a single support vector. a value 
# of gamma, such as 100.0, makes the model fit the training data more closely, 
# can result in overfitting if not handled properly.
# * C=1.0
# This parameter controls the penalty for classification errors. One more C value 
# high makes the model penalize errors more, seeking a more exact fit to the data 
# of training.
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)

# MODEL TRAINING
# The SVM classifier is trained using the training data set X_train_std 
# (which has been standardized) and y_train (the corresponding class labels). During this 
# process, the model adjusts its internal parameters according to the data provided.
svm.fit(X_train_std, y_train)

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function is called, which creates a plot that displays the regions 
# of the trained SVM classifier. It uses X_combined_std and y_combined, which 
# probably contain both training and test data, and test_idx will be 
# used to highlight specific test examples in the chart.
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))

# LABEL AND DISPLAY SETUP
# Labels are added to the X and Y axes of the chart to indicate which features are being 
# representing (length and width of petals).
# A legend is added in the upper left corner to identify the classes 
# represented in the graph.
# plt.tight_layout() automatically adjusts the plot spacing to avoid overlaps.
# Finally, plt.show() displays the graph on the screen. The commented line (plt.savefig(...)) 
# could be used to save the graph as a PNG file if uncommented.
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_16.png', dpi=300)
plt.show()


# # Decision tree learning



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_17.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_17.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# DEFINITION OF THE ENTROPY FUNCTION
# This function calculates the entropy H of a binary distribution given a probability p that 
# an event belongs to class 1. Entropy is measured in bits and is used to quantify 
# the uncertainty of a distribution:
# The formula takes into account both the probability p that an event belongs to class 1 
# as the 1-p probability that it belongs to class 0.
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# GENERATION OF PROBABILITY VALUES
# An array x containing values ​​from 0.0 to 1.0 (not including 1.0) is created, with a step 
# of 0.01. This represents different probabilities p for class 1.
x = np.arange(0.0, 1.0, 0.01)

# CALCULATION OF ENTROPY FOR EACH PROBABILITY
# A list comprehension is used to calculate the entropy for each value of p in the 
# array x.
# If p is equal to 0, None is assigned to avoid calculating logarithms of zero, since 
# log(0) is not defined.
ent = [entropy(p) if p != 0 else None 
       for p in x]

# ENTROPY DISPLAY
# Labels are added to the chart axes:
# The Y axis represents entropy.
# The X axis represents the probability of belonging to class 1 (p(i=1)).
# Plt.plot(x, ent) is used to plot entropy as a function of probabilities.
# Finally, plt.show() displays the graph on the screen. The commented line (plt.savefig(...)) 
# could be used to save the graphic as a PNG file if it is uncommented.
plt.ylabel('Entropy')
plt.xlabel('Class-membership probability p(i=1)')
plt.plot(x, ent)
# plt.savefig('figures/03_26.png', dpi=300)
plt.show()




# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_18.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_18.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Maximizing information gain - getting the most bang for the buck



# LIBRARY IMPORT
# Matplotlib.pyplot is imported to create plots and numpy is imported to perform numerical calculations.

# DEFINITION OF IMPURITY FUNCTIONS

# GINI
# Calculates the Gini impurity, which measures the probability that an element will be classified 
# incorrectly when chosen randomly.
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

# ENTROPY
# Calculates entropy, which measures the uncertainty in the classification.
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# CLASSIFICATION ERROR
# Calculate the classification error, which is 1 minus the maximum probability of membership 
# any of the classes.
def error(p):
    return 1 - np.max([p, 1 - p])

# GENERATION OF PROBABILITY VALUES
# An array x is created containing probability values ​​ranging from 0.0 to 1.0 (not including 1.0), 
# with a step of 0.01.
x = np.arange(0.0, 1.0, 0.01)

# CALCULATION OF IMPURITY INDICES
# The entropy is calculated for each value of p, scaled by half (sc_ent).
# The classification error is also calculated for each p value.
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

# VIEWING THE RESULTS
# A figure and an axis are created for the graph.
# A for loop is used to plot each impurity index as a function of p. Each line is 
# will be labeled and assigned a line style and color.
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini impurity', 'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# CHART SETUP
# A legend is added to the graph showing which line corresponds to which impurity index.
# Horizontal lines are added at y=0.5 and y=1.0 for reference.
# The Y axis limits are adjusted and the axes are labeled.
# Finally, the graph is displayed using plt.show(). The commented line (plt.savefig(...)) 
# could be used to save the graphic as a PNG file if it is uncommented.
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



# IMPORT DECISION TREE MODEL
# DecisionTreeClassifier is imported, which is the library's decision tree classifier 
# scikit-learn.

# CREATION OF THE DECISION TREE MODEL
# A DecisionTreeClassifier object called tree_model is instantiated with the following 
# parameters:
# * criterion='gini'
# Uses the Gini impurity as a criterion for the division of nodes.
# * max_depth=4
# Limits the maximum tree depth to 4, which helps prevent overfitting.
# * random_state=1
# Sets a seed for the random number generator, ensuring that the 
# results are reproducible.
tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)

# MODEL TRAINING
# The model is trained with the training data (X_train as features and y_train 
# as tags).
tree_model.fit(X_train, y_train)

# PREPARATION OF COMBINED DATA
# Training and testing data sets are combined. X_combined contains all 
# features of both sets, and y_combined contains the corresponding tags.
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function is called, which is used to display the regions.
# classifier=tree_model specifies the model to display.
# test_idx=range(105, 150) highlights the data points in the test set, which are 
# found in the index range 105 to 150.
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree_model,
                      test_idx=range(105, 150))

# LABELING AND GRAPH FORMAT
# The X and Y axes are labeled to indicate that they represent the length and width of the 
# petals, respectively.
# A legend is added to the top left.
# plt.tight_layout() makes sure that the chart elements fit tightly without overlapping.
# The commented line (plt.savefig(...)) can be used to save the graph as a 
# PNG file if uncommented.
# Finally, plt.show() displays the graph on the screen.
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('figures/03_20.png', dpi=300)
plt.show()




# TREE MODULE IMPORT
# The scikit-learn tree module is imported, which contains functions and classes to work with 
# with decision trees.

# DEFINITION OF FEATURE NAMES
# A list called feature_names is created that contains the names of the features that 
# were used to train the model. These names correspond to the measurements of the flowers 
# in the data set, probably the Iris data set.
feature_names = ['Sepal length', 'Sepal width',
                 'Petal length', 'Petal width']

# DECISION TREE DISPLAY
# The plot_tree function is called to display the decision tree represented by tree_model.
# * tree_model
# The decision tree model that was pre-trained.
# * feature_names=feature_names
# Feature names are passed to appear on the graph, making it easy 
# the interpretation of the tree.
# * filled=True
# This parameter indicates that the nodes should be filled with colors based on the predicted classes, 
# which helps visualize the decision tree more clearly.
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)

# SHOW THE GRAPH
# The commented line (plt.savefig(...)) suggests that the graph could be saved as a file 
# PDF if uncommented.
# plt.show() is used to display the graph on the screen, allowing users 
# display the decision tree.
# plt.savefig('figures/03_21_1.pdf')
plt.show()


# ## Combining weak to strong learners via random forests



# RANDOM FOREST CLASSIFIER IMPORT
# The RandomForestClassifier class is imported from the scikit-learn ensemble module. 
# This classifier uses multiple decision trees to improve accuracy and reduce 
# overfitting.

# CREATION OF THE RANDOM FOREST MODEL
# An object of the RandomForestClassifier class called forest is instantiated.
# * n_estimators=25
# This parameter indicates that 25 decision trees will be created in the forest. More trees 
# They generally improve model accuracy, but increase training time.
# * random_state=1
# A random seed is set to ensure the reproducibility of the results; 
# that is, the model will produce the same results every time it is run with this seed.
# * n_jobs=2
# This parameter allows using 2 processing cores to train the model in 
# parallel, which can speed up the process, especially with a large data set.
forest = RandomForestClassifier(n_estimators=25, 
                                random_state=1,
                                n_jobs=2)

# MODEL TRAINING
# The forest model is trained using the training data X_train (features) and 
# y_train (tags or classes).
forest.fit(X_train, y_train)

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function is called to display the decision regions of the 
# classifier in feature space.
# * X_combined
# Contains both training and test data.
# * y_combined
# Contains the corresponding labels for the combined data.
# * classifier=forest
# The trained random forest model is passed so that its decision regions are plotted.
# * test_idx=range(105, 150)
# A range of indices is specified to highlight test instances in the graph, 
# which allows us to observe how the model classifies these points.
plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

# LABELING AND PRESENTATION OF THE GRAPH
# The X and Y axis labels are set.
# A legend is added to the top left of the chart.
# plt.tight_layout() automatically adjusts the chart parameters so that it looks good on 
# the figure.
# The commented line plt.savefig(...) suggests that the graph could be saved as a file 
# if uncommented.
# plt.show() is used to display the graph on the screen.
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('figures/03_2.png', dpi=300)
plt.show()


# # K-nearest neighbors - a lazy learning algorithm



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/03_23.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/03_23.png', which is a relative path to the current directory.
# * width=400
# Set the image width to 400 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# KNN CLASSIFIER IMPORTATION
# The KNeighborsClassifier class is imported from the neighbors module of scikit-learn, which is used 
# to create a classification model based on the KNN algorithm.

# KNN MODEL CREATION
# An object of the KNeighborsClassifier class called knn is instantiated with the following 
# parameters:
# * n_neighbors=5
# This parameter indicates that the model will consider the 5 nearest neighbors when performing the 
# classification.
# * p=2
# This parameter specifies the distance to use. With p=2, the distance is used 
# Euclidean (which is a form of the Minkowski metric).
# * metric='minkowski'
# It is defined that the metric used to calculate the distance between points will be the metric 
# of Minkowski.
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')

# MODEL TRAINING
# The knn model is trained using the training data X_train_std (features 
# standardized) and y_train (tags or classes). In the context of KNN, no 
# explicit training, but training data information is stored for 
# future classification.
knn.fit(X_train_std, y_train)

# DISPLAY OF DECISION RULES
# The plot_decision_regions function is called to display the decision regions of the 
# KNN classifier in feature space.
# * X_combined_std
# Contains both training and test data (already standardized).
# * y_combined
# Contains the corresponding labels for the combined data.
# * classifier=knn
# The trained KNN model is passed so that its decision regions are plotted.
# * test_idx=range(105, 150)
# A range of indices is specified to highlight test instances in the graph, 
# allowing you to observe how the model classifies these points.
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

# LABELING AND PRESENTATION OF THE GRAPH
# The X and Y axis labels are set to indicate the characteristics that are being 
# viewing.
# A legend is added to the top left of the graph to identify the classes.
# plt.tight_layout() automatically adjusts the chart parameters to make it look nice 
# in the figure.
# The commented line plt.savefig(...) suggests that the graph could be saved as a file 
Image # if uncommented.
# plt.show() is used to display the graph on the screen.
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
# * --input ch03.ipynb
# This is an option or argument that tells the script what the input file is, in this 
# case, the notebook ch03.ipynb.
# * --output ch03.py
# This option tells the script to save the output (the converted file) with the name 
#ch03.py, which is a Python script.


