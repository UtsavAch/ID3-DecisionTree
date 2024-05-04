import pandas as pd
import numpy as np
from collections import Counter
import math

class DecisionTree:
    def __init__(self):
        """Initialize the DecisionTree object."""
        self.tree = None

    def entropy(self, class_labels):
        """
        Calculate the entropy of a set of class labels.

        Parameters:
        - class_labels (list or pandas.Series): List of class labels.

        Returns:
        - float: Entropy value.
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
        Calculate the information gain for a specific attribute.

        Parameters:
        - data (pandas.DataFrame): DataFrame containing the dataset.
        - attribute_name (str): Name of the attribute for which information gain is calculated.
        - class_name (str): Name of the class variable.

        Returns:
        - float: Information gain value.
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
        Choose the best attribute to split the dataset based on information gain.

        Parameters:
        - data (pandas.DataFrame): DataFrame containing the dataset.
        - attribute_names (list): List of attribute names.
        - class_name (str): Name of the class variable.

        Returns:
        - str: Name of the best attribute.
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
        - data (pandas.DataFrame): DataFrame containing the dataset.
        - attributes (list): List of attribute names.
        - class_name (str): Name of the class variable.

        Returns:
        - dict: Decision tree represented as a nested dictionary.
        """
        class_labels = data[class_name]

        if len(set(class_labels)) == 1:
            return class_labels.iloc[0]
        if len(attributes) == 0:
            return class_labels.mode()[0]

        best_attribute = self.choose_best_attribute(data, attributes, class_name)
        tree = {best_attribute: {}}
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        for value in data[best_attribute].unique():
            subset = data[data[best_attribute] == value]
            subtree = self.build_tree(subset, remaining_attributes, class_name)
            tree[best_attribute][value] = subtree

        return tree

    def fit(self, data, class_name):
        """
        Fit the decision tree model to the training data.

        Parameters:
        - data (pandas.DataFrame): DataFrame containing the training dataset.
        - class_name (str): Name of the class variable.
        """
        attributes = list(data.columns[:-1])  # Exclude the last column (class_name)
        self.tree = self.build_tree(data, attributes, class_name)

    def predict_instance(self, instance):
        """
        Predict the class label for a single instance using the trained decision tree.

        Parameters:
        - instance (pandas.Series): Single instance (row) of the dataset.

        Returns:
        - str: Predicted class label.
        """
        tree = self.tree
        while isinstance(tree, dict):
            attribute = list(tree.keys())[0]
            value = instance[attribute]
            tree = tree[attribute].get(value)  # Use .get() to handle unknown attribute values
            if tree is None:
                break
        return tree

    def predict(self, test_data):
        """
        Predict class labels for a dataset using the trained decision tree.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test dataset.

        Returns:
        - list of tuples: List of (prediction, formatted_tree) for each instance in the test dataset.
          prediction (str): Predicted class label.
          formatted_tree (str): Formatted string representation of the decision tree path for the instance.
        """
        predictions = []
        for idx, instance in test_data.iterrows():
            prediction = self.predict_instance(instance)
            formatted_tree = self.format_tree(self.tree)  # Format the tree for display
            predictions.append((prediction, formatted_tree))  # Store prediction and formatted tree
        return predictions

    def format_tree(self, tree, depth=0):
        """
        Recursively format the decision tree into a string representation.

        Parameters:
        - tree (dict): Decision tree represented as a nested dictionary.
        - depth (int): Current depth in the tree (used for indentation).

        Returns:
        - str: Formatted string representation of the decision tree.
        """
        if not isinstance(tree, dict):
            return tree

        formatted_tree = ""
        for attribute, subtree in tree.items():
            formatted_tree += f"<{attribute}>\n"
            for value, subsubtree in subtree.items():
                formatted_tree += f"    {'    ' * depth}{value}: {self.format_tree(subsubtree, depth + 1)}\n"

        return formatted_tree

# Load the restaurant dataset
file_path = 'restaurant.csv'
data = pd.read_csv(file_path)

# Display the dataset (optional)
print("Restaurant Dataset:")
print(data.head())

# Remove 'ID' column from the dataset
data = data.drop('ID', axis=1)

# Instantiate and fit the DecisionTree model
tree_model = DecisionTree()
tree_model.fit(data, class_name='Class')  # Assuming 'Class' is the class variable

# Example test data (new instances to classify)
test_data_restaurant = pd.DataFrame({
    'Alt': ['No', 'Yes', 'No'],       
    'Bar': ['Yes', 'Yes', 'Yes'],      
    'Fri': ['Yes', 'Yes', 'No'],       
    'Hun': ['No', 'Yes', 'No'],       
    'Pat': ['Full', 'Some', 'Full'],  
    'Price': ['$', '$$', '$'],       
    'Rain': ['No', 'Yes', 'No'],     
    'Res': ['No', 'Yes', 'No'],        
    'Type': ['Thai', 'French', 'Thai'],
    'Est': ['30-60', '0-10', '10-30']
})

# Use the trained model to predict class labels for the test data
predictions = tree_model.predict(test_data_restaurant)

# Display the formatted predictions and final classifications
print("\nFormatted Predictions:")
for i, (pred, formatted_tree) in enumerate(predictions):
    print(f"Scenario {i+1}: Predicted Outcome = {pred}")
    print("Formatted Decision Tree:")
    print(formatted_tree)
    print()  # Add a blank line for clarity
