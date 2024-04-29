import pandas as pd
import numpy as np
from collections import Counter
import math

class DecisionTree:
    def __init__(self):
        """
        Initialize the DecisionTree object.
        """
        self.tree = None

    def entropy(self, class_labels):
        """
        Calculate the entropy of a list of class labels.

        Parameters:
        - class_labels (list): List of class labels

        Returns:
        - entropy (float): Entropy value
        """
        counts = Counter(class_labels)
        entropy = 0
        total_samples = len(class_labels)
        
        for label in counts:
            probability = counts[label] / total_samples
            entropy -= probability * math.log2(probability)
        
        return entropy

    def information_gain(self, data, attribute_name, class_name):
        """
        Calculate the information gain for splitting on a particular attribute.

        Parameters:
        - data (DataFrame): Input dataset
        - attribute_name (str): Name of the attribute to calculate information gain for
        - class_name (str): Name of the class variable column

        Returns:
        - gain (float): Information gain value
        """
        total_entropy = self.entropy(data[class_name])
        values = data[attribute_name].unique()

        weighted_entropy = 0
        total_samples = len(data)

        for value in values:
            subset = data[data[attribute_name] == value]
            subset_entropy = self.entropy(subset[class_name])
            subset_weight = len(subset) / total_samples
            weighted_entropy += subset_weight * subset_entropy

        return total_entropy - weighted_entropy

    def choose_best_attribute(self, data, attribute_names, class_name):
        """
        Choose the best attribute to split on based on maximum information gain.

        Parameters:
        - data (DataFrame): Input dataset
        - attribute_names (list): List of attribute names
        - class_name (str): Name of the class variable column

        Returns:
        - best_attribute (str): Name of the best attribute
        """
        best_attribute = None
        best_gain = -1

        for attribute in attribute_names:
            gain = self.information_gain(data, attribute, class_name)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute

        return best_attribute

    def build_tree(self, data, attributes, class_name):
        """
        Recursively build the decision tree.

        Parameters:
        - data (DataFrame): Input dataset
        - attributes (list): List of attribute names
        - class_name (str): Name of the class variable column

        Returns:
        - tree (dict): Constructed decision tree
        """
        class_labels = data[class_name]

        # Base cases: all samples have the same class or no attributes left to split
        if len(set(class_labels)) == 1:
            return class_labels.iloc[0]
        if len(attributes) == 0:
            return class_labels.mode()[0]

        # Choose the best attribute to split on
        best_attribute = self.choose_best_attribute(data, attributes, class_name)
        tree = {best_attribute: {}}

        # Remove the chosen attribute from the list of attributes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Recursively build the subtree for each attribute value
        for value in data[best_attribute].unique():
            subset = data[data[best_attribute] == value]
            subtree = self.build_tree(subset, remaining_attributes, class_name)
            tree[best_attribute][value] = subtree

        return tree

    def fit(self, data, class_name):
        """
        Fit the decision tree model to the training data.

        Parameters:
        - data (DataFrame): Training dataset
        - class_name (str): Name of the class variable column
        """
        attributes = list(data.columns[:-1])  # Get all attribute names except the class column
        self.tree = self.build_tree(data, attributes, class_name)

    def predict_instance(self, instance):
        """
        Predict the class label for a single instance using the decision tree.

        Parameters:
        - instance (Series): Single instance (row) from the dataset

        Returns:
        - prediction: Predicted class label
        """
        tree = self.tree
        while isinstance(tree, dict):
            attribute = list(tree.keys())[0]
            value = instance[attribute]
            tree = tree[attribute][value]
        return tree

    def predict(self, test_data):
        """
        Predict the class labels for a dataset using the decision tree.

        Parameters:
        - test_data (DataFrame): Dataset containing instances to classify

        Returns:
        - predictions (list): List of predicted class labels
        """
        predictions = []
        for idx, instance in test_data.iterrows():
            prediction = self.predict_instance(instance)
            predictions.append(prediction)
        return predictions

# Example usage:
# Assuming 'data' is a pandas DataFrame representing your dataset with attribute columns and a class column
# Specify the name of the class column ('class_name')
# Instantiate and fit the DecisionTree model
tree_model = DecisionTree()
tree_model.fit(data, class_name='class_variable')

# Assuming 'test_data' is a pandas DataFrame representing new examples to classify
# Use the trained model to predict class labels for the test data
predictions = tree_model.predict(test_data)

# Print or use predictions as needed
print(predictions)
