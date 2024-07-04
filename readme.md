# Decision Tree Implementation in Python

This project implements a Decision Tree classifier from scratch in Python. The classifier supports both categorical and numerical attributes and provides functionality to build the tree, print it, and make predictions.

## Overview

A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It splits the dataset into subsets based on the attribute value and builds a tree-like model of decisions. This project implements the Decision Tree classifier to handle datasets with both categorical and numerical attributes.

## Features

- Build a Decision Tree classifier from scratch.
- Handle both categorical and numerical attributes.
- Print the decision tree in a readable format.
- Predict classes for new instances.
- Customize maximum tree depth.

## Usage

### Building the Decision Tree

To build a decision tree, initialize the `DecisionTree` class and use the `fit` method with your dataset. Or, you can check the test.ipynb file.

```python
import pandas as pd
from decision_tree import DecisionTree

# Load your dataset
data = pd.read_csv('path/to/your/dataset.csv')

# Initialize the Decision Tree
tree = DecisionTree(max_depth=5)

# Fit the tree with the dataset
tree.fit(data, class_name='Class')
```
