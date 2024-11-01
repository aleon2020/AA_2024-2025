# coding: utf-8


import sys
# * from python_environment_check import check_packages
from python_environment_check import check_packages

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
# import is used. When you try to import a module, Python searches the paths specified in this
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
    'scipy': '1.7.0',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}
check_packages(d)


# # Chapter 1 - Giving Computers the Ability to Learn from Data

# ### Overview

# - [Building intelligent machines to transform data into knowledge](#Building-intelligent-machines-to-transform-data-into-knowledge)
# - [The three different types of machine learning](#The-three-different-types-of-machine-learning)
#     - [Making predictions about the future with supervised learning](#Making-predictions-about-the-future-with-supervised-learning)
#         - [Classification for predicting class labels](#Classification-for-predicting-class-labels)
#         - [Regression for predicting continuous outcomes](#Regression-for-predicting-continuous-outcomes)
#     - [Solving interactive problems with reinforcement learning](#Solving-interactive-problems-with-reinforcement-learning)
#     - [Discovering hidden structures with unsupervised learning](#Discovering-hidden-structures-with-unsupervised-learning)
#         - [Finding subgroups with clustering](#Finding-subgroups-with-clustering)
#         - [Dimensionality reduction for data compression](#Dimensionality-reduction-for-data-compression)
#         - [An introduction to the basic terminology and notations](#An-introduction-to-the-basic-terminology-and-notations)
# - [A roadmap for building machine learning systems](#A-roadmap-for-building-machine-learning-systems)
#     - [Preprocessing - getting data into shape](#Preprocessing--getting-data-into-shape)
#     - [Training and selecting a predictive model](#Training-and-selecting-a-predictive-model)
#     - [Evaluating models and predicting unseen data instances](#Evaluating-models-and-predicting-unseen-data-instances)
# - [Using Python for machine learning](#Using-Python-for-machine-learning)
# - [Installing Python packages](#Installing-Python-packages)
# - [Summary](#Summary)



# * from IPython.display
# Import from the display submodule of the IPython package. This module is designed to display 
# and render different types of data within interactive environments, such as Jupyter Notebooks.
# * import Image
# Import the Image class from the display module. The Image class is used to display 
# images in the interactive environment (for example, in a Jupyter Notebook cell).



# # Building intelligent machines to transform data into knowledge

# # The three different types of machine learning



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_01.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_01.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Making predictions about the future with supervised learning



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_02.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_02.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ### Classification for predicting class labels



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_03.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_03.png', which is a relative path to the current directory.
# * width=300
# Set the image width to 300 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ### Regression for predicting continuous outcomes



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_04.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_04.png', which is a relative path to the current directory.
# * width=300
# Set the image width to 300 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Solving interactive problems with reinforcement learning



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_05.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_05.png', which is a relative path to the current directory.
# * width=300
# Set the image width to 300 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ## Discovering hidden structures with unsupervised learning

# ### Finding subgroups with clustering



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_06.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_06.png', which is a relative path to the current directory.
# * width=300
# Set the image width to 300 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ### Dimensionality reduction for data compression



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_07.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_07.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# ### An introduction to the basic terminology and notations



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_08.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_08.png', which is a relative path to the current directory.
# * width=500
# Set the image width to 500 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



# # A roadmap for building machine learning systems



# * Image(...)
# Use the Image class (probably imported from IPython.display, as in the previous example) 
# to display an image in an interactive environment such as Jupyter Notebook.
# * filename='./figures/01_09.png'
# Specifies the path of the image to display. In this case, the image is located in the
# file './figures/01_09.png', which is a relative path to the current directory.
# * width=700
# Set the image width to 700 pixels. This resizes the image so that it occupies that 
# space width, while its height is adjusted proportionally (if you do not specify a 
# height).



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
# * --input ch01.ipynb
# This is an option or argument that tells the script what the input file is, in this 
# case, the notebook ch01.ipynb.
# * --output ch01.py
# This option tells the script to save the output (the converted file) with the name 
#ch01.py, which is a Python script.


