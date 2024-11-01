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
# This module, by name, is designed to verify that the Python environment 
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
# Import from the display submodule of the IPython package. This module is designed to display 
# and render different types of data within interactive environments, such as Jupyter Notebooks.
# * import Image
# Import the Image class from the display module. The Image class is used to display 
# images in the interactive environment (for example, in a Jupyter Notebook cell).



# # Artificial neurons - a brief glimpse into the early history of machine learning



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_01.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_01.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## The formal definition of an artificial neuron



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_02.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_02.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## The perceptron learning rule



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_03.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_03.png', which is a relative path to the current directory.
# * width=600
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_04.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_04.png', which is a relative path to the current directory.
# * width=600
# Set the image width to 600 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API




# PERCEPTRON CLASS
# It is a supervised learning model. The perceptron adjusts the weights of a linear model 
# using a training data set and a learning algorithm.
class Perceptron:
    
    # INITIALIZATION PARAMETERS
    # - eta: Learning rate (a value between 0 and 1). Controls the size of the settings in the 
    # weights during training.
    # - n_iter: Number of iterations (or epochs) on the training data.
    # - random_state: Seed to generate random numbers, ensuring reproducibility when 
    # initialize the weights randomly.
    # ATTRIBUTES
    # - w_: Vector of weights that will be adjusted during training.
    # - b_: Bias, which is also adjusted during training.
    # - errors_: List that stores the number of classification errors (updates) 
    # in each period of training.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # FIT METHOD
    # - X: Matrix with the training data (characteristics of the examples).
    # - y: Vector with the target labels or classes for each example.
    # The method adjusts the weights (w_) and bias (b_) over several iterations over the 
    # training data.
    # In each iteration, the predictions for each sample are calculated, the error is evaluated, and 
    # update the weights and bias based on that error.
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

    # NET_INPUT METHOD
    # Calculates the "net input", which is the linear combination of the input characteristics 
    # weighted by the weights plus the bias (this is what the perceptron evaluates to do 
    # predictions).
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # PREDICT METHOD
    # Returns the class prediction for an input. If the net input is greater than or equal to 0, 
    # class 1 is predicted, and if not, class 0 is predicted. This is equivalent to applying a function 
    # (step unit).
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)




# VECTOR DEFINITION
# * v1 = np.array([1, 2, 3])
# A three-component vector v1 [1, 2, 3] is created using numpy.
# * v2 = 0.5 * v1
# A new vector v2 is created, which is the result of multiplying each component of v1 by 0.5, 
# getting the vector [0.5, 1.0, 1.5].
v1 = np.array([1, 2, 3])
v2 = 0.5 * v1

# CALCULATION OF THE ANGLE BETWEEN V1 AND V2
# * v1.dot(v2)
# Calculates the dot product (or scalar product) between the vectors v1 and v2.
# * np.linalg.norm(v1) and np.linalg.norm(v2)
# They calculate the norms (magnitudes) of the vectors v1 and v2, respectively.
# * v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
# Calculate the cosine of the angle between the vectors using the dot product formula 
# normalized.
# * np.arccos(...)
# Apply the arccosine function (arccos) to obtain the angle in radians between v1 and v2.
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ## Training a perceptron model on the Iris dataset

# ### Reading-in the Iris data



# LIBRARY IMPORT
# * imports
# Module to interact with the operating system (although it is not used in this code).
# * import pandas as pd
# Import the pandas library, which is used for data manipulation in structures 
# as DataFrame.

# TRY-EXCEPT BLOCK
# An attempt is made to load the data file from a URL.
# Inside the try block:
# * URL of the dataset
# The s variable stores the URL of the Iris dataset, provided by UCI Machine Learning 
#Repository.
# * pd.read_csv
# Used to read the CSV file from the URL. The read_csv function downloads and loads the 
# file as a pandas DataFrame.
# * header=None
# Indicates that the file has no headers.
# * encoding='utf-8'
# Specifies the character encoding.
# * except HTTPError
# If an error occurs (for example, the URL cannot be accessed), the code will attempt to load 
# the file from a local path (iris.data). This allows the program to continue running 
# even though the URL cannot be accessed.
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

# SHOW LAST ROWS
# * df.tail()
# Once the file is uploaded (either from the URL or the local file), this line shows the 
# last 5 rows of the data set.
df.tail()


# ### Plotting the Iris data



# MATPLOTLIB CONFIGURATION
# * %matplotlib inline
# This line is a specific command for environments like Jupyter Notebook, which allows you to display 
# the graphs directly in the workbook.
# * import matplotlib.pyplot as plt
# Import the matplotlib.pyplot library as plt to create plots.
# * import numpy as np
# Imports numpy, which is useful for manipulation of numerical arrays.

# SELECTION OF FLOWER CLASSES
# * y = df.iloc[0:100, 4].values
# Extracts the class labels (Iris-setosa or Iris-versicolor) from the first 100 rows of the 
# DataFrame df (which contains the Iris dataset) and the fifth column (which has the names of the 
# species).
# * y = np.where(y == 'Iris-setosa', 0, 1)
# Converts labels to numeric values: 0 for Iris-setosa and 1 for Iris-versicolor.
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# FEATURE EXTRACTION
# * X = df.iloc[0:100, [0, 2]].values
# Extract two numerical features (sepal length and petal length) from the 
# first 100 rows, selecting the first column (sepal length) and the third 
# column (petal length). This creates a matrix X of 100 rows and 2 columns.
X = df.iloc[0:100, [0, 2]].values

# CREATION OF THE SCATTER PLOT
# * plt.scatter(...)
# Create two scatter plots (one for each flower class).
# * First scatter
# Plot the Iris-setosa data (first 50 rows of X), using red dots 
# (color='red', marker='o').
# * Second scatter
# Plot the Iris-versicolor data (rows 50 to 100 of X), using blue squares 
# (color='blue', marker='s').
# * plt.xlabel('Sepal length [cm]')
# Label the X axis as "Sepal length [cm]".
# * plt.ylabel('Petal length [cm]')
# Label the Y axis as "Petal length [cm]".
# * plt.legend(loc='upper left')
# Add a legend to identify the classes, placed in the upper left corner.
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
# plt.savefig('images/02_06.png', dpi=300)

# SHOW THE GRAPH
# * plt.show()
# Shows the graph on the screen.
plt.show()


# ### Training the perceptron model



# CREATION OF THE PERCEPTRON
# * ppn = Perceptron(eta=0.1, n_iter=10)
# An instance of the Perceptron model is created with a learning rate (eta) of 0.1 and 
# a number of iterations (epochs) equal to 10. This perceptron will use the algorithm 
# learning to adjust weights during training.
ppn = Perceptron(eta=0.1, n_iter=10)

# MODEL TRAINING
# * ppn.fit(X, y)
# The perceptron is trained with the input data X (features) and the labels y 
# (target classes). During training, the perceptron adjusts its weights based on 
# the classification errors in each epoch.
ppn.fit(X, y)

# DISPLAY OF THE NUMBER OF ERRORS
# * plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# A graph is generated showing the number of classification errors (updates) in 
# each era.
# * range(1, len(ppn.errors_) + 1)
# Represents the epochs, that is, the number of iterations from 1 to the last.
# * ppn.errors_
# It is a list that contains the number of errors in each training epoch.
# * marker='o'
# Use circles to mark points on the graph.
# * plt.xlabel('Epochs')
# Label the X axis "Epochs".
# * plt.ylabel('Number of updates')
# Label the Y axis as "Number of updates" 
# (number of updates or classification errors).
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
# plt.savefig('images/02_07.png', dpi=300)

# SHOW THE GRAPH
# * plt.show()
# Shows the graph on the screen.
plt.show()


# ### A function for plotting decision regions



# IMPORT OF MODULES
# Imports ListedColormap from matplotlib, which allows you to create custom colormaps for 
# the graphics.

# DEFINITION OF THE FUNCTION
# * def plot_decision_regions(X, y, classifier, resolution=0.02)
# Defines the function that takes as arguments:
# * X
# Feature matrix (two dimensions).
# * y
# Vector class labels.
# * classifier
# A trained classifier model (for example, a Perceptron).
# * resolution
# Graph resolution (the pitch between points on the mesh).
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # MARKER AND COLOR SETTINGS
    # * markers and colors
    # Lists of symbols (markers) and colors are defined to represent different classes 
    # on the chart.
    # * cmap
    # A color map is created from the list of colors, limiting it to the number of 
    # unique classes in and.
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # CALCULATION OF THE DECISION AREA
    # The range of values ​​for the characteristics in X (minimums and maximums) is determined, 
    # adjusting them to add a margin.
    # * np.meshgrid(...)
    # Creates a grid of points (mesh) over the space defined by the boundaries of the 
    # features, using a specified resolution.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # CLASS PREDICTION ON THE MESH
    # * classifier.predict(...)
    # Use the classifier to predict the classes for all points in the mesh.
    # * lab = lab.reshape(xx1.shape)
    # Reshapes the predictions to match the shape of the mesh.
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    # VISUALIZATION OF DECISION REGIONS
    # * plt.contourf(...)
    # Draws a filled contour graph representing the decision regions 
    # in space, using the defined color map.
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # VIEWING CLASS EXAMPLES
    # A for loop loops through the unique classes in y and uses plt.scatter(...) 
    # to draw the actual data points on the graph, assigning to each class 
    # its respective color and marker.
    # * label=f'Class {cl}'
    # Tag each class in the legend.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')




# FUNCTION CALL
# * plot_decision_regions(X, y, classifier=ppn)
# This line calls the function that was previously defined to show how the ppn classifier 
# (Perceptron) separates the different classes in the space defined by the characteristics X and 
# the tags and.
# * X
# Contains the characteristics (e.g. sepal length and petal length).
# * y 
# Contains the class tags (for example, flower classes).
plot_decision_regions(X, y, classifier=ppn)

# AXLE LABELS
# * plt.xlabel('Sepal length [cm]')
# Set the X axis label to "Sepal length [cm]", indicating that it is being plotted 
# the length of the sepal.
# * plt.ylabel('Petal length [cm]')
# Sets the Y axis label to "Petal length [cm]", indicating that it is being plotted 
# the length of the petal.
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')

# LEGEND
# * plt.legend(loc='upper left')
# Adds a legend to the chart in the upper left corner, which helps identify 
# the classes represented in the graph.
plt.legend(loc='upper left')
#plt.savefig('images/02_08.png', dpi=300)

# SHOW THE GRAPH
# * plt.show()
# Display the generated graph in the visualization interface 
# (for example, a Jupyter notebook or a graphics window).
plt.show()


# # Adaptive linear neurons and the convergence of learning

# ## Minimizing cost functions with gradient descent



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_09.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_09.png', which is a relative path to the current directory.
# * width=600
# Set the image width to 600 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_10.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_10.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Implementing an adaptive linear neuron in Python



# CLASS AND DOCUMENTATION
# * AdalineGD
# Defines the class for the Adaline model.
# The documentation (docstring) describes the parameters, attributes and methods of the class.
class AdalineGD:

    # INITIALIZATION PARAMETERS
    # * eta
    # Learning rate, controls the size of the adjustments to the weights (value between 0.0 and 1.0).
    # * n_iter
    # Number of iterations (epochs) on the training data set.
    # * random_state
    # Seed to initialize the random weights, allowing reproducibility.
    # ATTRIBUTES
    # * w_
    # One-dimensional array that stores the adjusted weights after training.
    # * b_
    # Scalar representing the adjusted bias.
    #*losses_
    # List that stores the values ​​of the loss function (mean squared error) in each 
    # epoch.

    # __INIT__ METHOD
    # Initializes the model parameters and sets the weights and bias.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # FIT METHOD
    # - Training: This method trains the model using the data X (features) and 
    # and (tags).
    # - Generates random initial weights and a bias.
    # In each era:
    # - Calculate net input using net_input.
    # - Calculates the output using the activation function (in this case, linear).
    # - Calculates the error as the difference between the true labels and the predictions.
    # - Updates weights and bias based on error.
    # - Calculates the loss (mean square error) and stores it.
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

    # NET_INPUT METHOD
    # Calculates the net entry as the dot product of X and the weights, plus the bias.
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # ACTIVATION METHOD
    # Defines the activation function. In this case, it simply returns the input without 
    # changes (identity function).
    def activation(self, X):
        return X

    # PREDICT METHOD
    # Returns the class label for new samples. Use a threshold of 0.5 to 
    # classify outputs as 0 or 1.
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




# CREATION OF THE FIGURE AND THE AXES
# * fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# Creates a figure (fig) with two subgraphs (ax), organized in one row and two columns. 
# The size of the figure is 10x4 inches.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# TRAINING WITH THE FIRST LEARNING RATE
# * ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
# An instance of the Adaline model is created with 15 epochs and a learning rate (eta) of 0.1. 
# The model is then trained with the X and y data.
# * ax[0].plot(...)
# The evolution of the loss function (mean squared error) is plotted as a function of the 
# epochs. np.log10(ada1.losses_) is used to display the base 10 logarithm of the loss, 
# which helps to better visualize changes over wide ranges.
# * ax[0].set_xlabel(...)
# Set the X axis label to "Epochs".
# * ax[0].set_ylabel(...)
# Set the Y axis label to "log(Mean squared error)".
# * ax[0].set_title(...)
# Set the title of the first graph to "Adaline - Learning rate 0.1".
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

# TRAINING WITH THE SECOND LEARNING RATE
# * ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
# Another instance of the Adaline model is created with 15 epochs and a learning rate of 0.0001, 
# and is trained with the same data.
# * ax[1].plot(...)
# The loss function is plotted directly (without logarithm) as a function of epochs.
# * ax[1].set_xlabel(...)
# Set the X axis label to "Epochs".
# * ax[1].set_ylabel(...)
# Set the Y axis label to "Mean squared error".
# * ax[1].set_title(...)
# Set the title of the second graph to "Adaline - Learning rate 0.0001".
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
# plt.savefig('images/02_11.png', dpi=300)

# SHOW THE GRAPH
# * plt.show()
# Shows the figure with both graphs on the screen.
plt.show()




# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_12.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_12.png', which is a relative path to the current directory.
# * width=700
# Set the image width to 700 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Improving gradient descent through feature scaling



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/02_13.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/02_13.png', which is a relative path to the current directory.
# * width=700
# Set the image width to 700 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).





# COPY OF ARRAY
# * X_std = np.copy(X)
# Create a copy of array X, called X_std, so that transformations do not modify the 
# original data set.
X_std = np.copy(X)

# NORMALIZATION OF THE FIRST CHARACTERISTICS
# * X[:, 0].mean()
# Calculate the mean of the first characteristic.
# * X[:, 0].std()
# Calculate the standard deviation of the first characteristic.
# The mean is subtracted from each value and divided by the standard deviation, 
# which transforms the first column of X to a new scale where it will have mean 0 
# and standard deviation 1.
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

# NORMALIZATION OF THE SECOND CHARACTERISTICS
# X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
# Perform the same normalization process for the second feature 
# (second column of X).
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()




# MODEL CREATION AND TRAINING
# * ada_gd = AdalineGD(n_iter=20, eta=0.5)
# An instance of the Adaline model is created with 20 epochs and a learning rate (eta) of 0.5.
# * ada_gd.fit(X_std, y)
# The model is trained using the standardized data set X_std and the y labels.
ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

# VISUALIZATION OF DECISION REGIONS
# * plot_decision_regions(X_std, y, classifier=ada_gd)
# The plot_decision_regions function is called to display the decision regions of the 
# Adaline classifier in standardized feature space.
# * plt.title('Adaline - Gradient descent')
# The title of the chart is set.
# * plt.xlabel('Sepal length [standardized]')
# The X axis is labeled, indicating that the standardized sepal length is being plotted.
# * plt.ylabel('Petal length [standardized]')
# The Y axis is labeled for the standardized petal length.
# * plt.legend(loc='upper left')
# A legend is added to the chart in the upper left corner.
# * plt.tight_layout()
# Adjust the layout of the chart so that elements do not overlap.
# * plt.show()
# Shows the graph with the decision regions.
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

# LOSS FUNCTION DISPLAY
# * plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
# The evolution of the mean square error over the epochs is graphed.
# * range(1, len(ada_gd.losses_) + 1)
# Generate the range of epochs from 1 to the number of trained epochs.
# * ada_gd.losses_
# Contains the loss function values ​​for each epoch.
# * plt.xlabel('Epochs')
# The X axis is labeled "Epochs".
# * plt.ylabel('Mean squared error')
# Label the Y axis as "Mean squared error".
# * plt.tight_layout()
# Adjust the layout of the graph.
# * plt.show()
# Shows the loss function graph.
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
#plt.savefig('images/02_14_2.png', dpi=300)
plt.show()


# ## Large scale machine learning and stochastic gradient descent



# ADALINESGD CLASS
# Implements an adaptive linear neuron classifier (Adaline) using the method of 
# stochastic gradient descent (SGD) to train the model.
class AdalineSGD:

    # PARAMETERS
    # * eta
    # The learning rate, which controls the size of the weight adjustment steps in 
    # each iteration.
    # * n_iter
    # Number of epochs or iterations on the training data set.
    # * shuffle
    # If True, shuffles the data at each epoch to avoid repetitive cycles that can 
    # affect learning.
    # * random_state
    # Seed for initialization of weights and random mixing, which guarantees 
    # reproducible results.
    # ATTRIBUTES
    # * w_
    # The weights of the model, which are adjusted during training.
    # * b_
    # The bias of the model, which is also adjusted during training.
    # * losses_
    # Stores the mean square errors for each epoch, which indicate how well it is doing 
    # adjusting the model.
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    # FIT METHOD
    # Fit the model weights and bias to the training data X (features) and 
    # and (tags). For each epoch, update the weights using gradient descent 
    # stochastic.
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

    # PARTIAL_FIT METHOD
    # Similar to fit, but does not reset the weights, allowing you to do training 
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

    # _SHUFFLE METHOD
    # Randomly reorganize the data in each iteration to avoid repetitive patterns 
    # that may interfere with learning.
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    # INITIALIZE_WEIGHTS METHOD
    # Initialize the weights and bias with small random numbers.
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
    
    # _UPDATE_WEIGHTS METHOD
    # Updates the weights based on a single example xi and its target target. 
    # Calculate the error and adjust the weights and bias using Adaline's learning rule.
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
    
    # NET_INPUT METHOD
    # Calculates the weighted sum of the inputs plus the bias.
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    # ACTIVATION METHOD
    # Returns the activation value, which in this case is linear (Adaline uses activation 
    # linear).
    def activation(self, X):
        return X

    # PREDICT METHOD
    # Predict the class (0 or 1) for a new input based on the output of the activation.
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




# MODEL INITIALIZATION
# Adaline classifier is instantiated using stochastic gradient descent 
# (AdalineSGD) with 15 iterations (n_iter=15), a learning rate of 0.01 (eta=0.01), 
# and a random seed (random_state=1) to ensure reproducible results.
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)

# MODEL TRAINING
# The model is trained using the X_std data (standardized features) and 
# and (class tags). Here, X_std probably contains standardized features 
# of sepal length and flower petal length, since these attributes are 
# mentioned later in the axes.
ada_sgd.fit(X_std, y)

# VISUALIZATION OF DECISION REGIONS
# The plot_decision_regions function plots the decision regions generated by the model 
# Adaline in the X_std and y data. The axis labels indicate that the characteristics 
# correspond to the standardized sepal length and petal length.
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')

# ADJUSTING FIGURE LAYOUT AND SAVED
# plt.tight_layout() automatically adjusts the plot margins to avoid 
# overlaps, then save the figure to the path 'figures/02_15_1.png' with a resolution 
# of 300 DPI. Finally, the graph is shown.
plt.tight_layout()
plt.savefig('figures/02_15_1.png', dpi=300)
plt.show()

# LOSS FUNCTION PLOT
# A graph is generated that shows the evolution of the average loss (mean squared error) 
# across the 15 training epochs, which helps visualize how the model improves 
# during learning.
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')

# SAVE AND VIEW THE LOSS GRAPH
# This graph is saved as 'figures/02_15_2.png' with a resolution of 300 DPI and then 
# sample.
plt.savefig('figures/02_15_2.png', dpi=300)
plt.show()




# * ada_sgd
# It is a pre-trained or initialized instance of the Adaline classifier that uses 
# stochastic gradient descent (SGD).
# * X_std[0, :]
# Select the first example of the standardized characteristics (X_std). Here, X_std[0, :] 
# takes the first row of X_std, which corresponds to the characteristics of a single example 
# (a flower in this case).
# * y[0]
# Take the target value or label associated with that first example.
# The partial_fit method fits the model using only this example (X_std[0, :] and y[0]), 
# which means it updates the model weights without reinitializing them. This type of 
# incremental adjustment is useful in situations where data arrives in sequence or when it does not 
# you want to retrain the model from scratch.

ada_sgd.partial_fit(X_std[0, :], y[0])


# # Summary

# --- 
# 
# Readers may ignore the following cell



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
# * --input ch02.ipynb
# This is an option or argument that tells the script what the input file is, in this 
# case, the notebook ch02.ipynb.
# * --output ch02.py
# This option tells the script to save the output (the converted file) with the name 
#ch02.py, which is a Python script.


