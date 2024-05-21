import numpy as np
import pandas as pd
import math

class DecisionTree:
    def __init__(self, max_depth=None):
        """Initialize the decision tree with an optional max depth"""
        self.tree = None
        self.max_depth = max_depth

    def build_decision_tree(self, dataset, attributes, class_name, current_depth=0):
        """
        Recursively build the decision tree.
        Parameters:
        - dataset: DataFrame containing the dataset.
        - attributes: List of attribute names.
        - class_name: Name of the class variable.
        - current_depth: Current depth of the tree.
        Returns:
        - dict: Decision tree represented as a nested dictionary.
        """
        # Replace None values with "None" string in categorical attributes
        dataset = dataset.fillna("None")

        # Base case: if all instances belong to the same class
        class_counts = self.attribute_value_counts_simple(dataset, class_name)[0]
        if len(class_counts) == 1:
            class_label = next(iter(class_counts))
            count = class_counts[class_label]
            return (class_label, count)  # Return the class label with count

        # Base case: if no attributes left to split on or max depth reached
        if len(attributes) == 0 or (self.max_depth is not None and current_depth >= self.max_depth):
            most_frequent_class = dataset[class_name].mode()[0]
            count = class_counts[most_frequent_class]
            return (most_frequent_class, count)  # Return the most frequent class with count

        # Select the best attribute to split on
        best_attribute = self.get_best_attribute(dataset, attributes, class_name)
        tree = {best_attribute: {}}

        # Remove the best attribute from the list of attributes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Determine attribute type (numerical or categorical)
        attr_type = self.attribute_type(dataset, best_attribute)

        if attr_type == 'categorical':
            # For categorical attribute, create sub-tree for each attribute value
            attribute_values = dataset[best_attribute].unique()
            for value in attribute_values:
                filtered_subset = self.split_dataset(dataset, best_attribute, value)[0]
                # Recursively build the subtree for the current attribute value
                subtree = self.build_decision_tree(filtered_subset, remaining_attributes, class_name, current_depth + 1)
                # Add the subtree to the current tree for the specific attribute value
                tree[best_attribute][value] = subtree

        elif attr_type == 'numerical':
            # For numerical attribute, find the best threshold and split the dataset
            threshold = self.threshold_of_numerical_attribute(dataset, best_attribute, class_name)
            left_subset = self.split_dataset(dataset, best_attribute, threshold)[0]
            right_subset = self.split_dataset(dataset, best_attribute, threshold)[1]
            # Recursively build subtrees for the splits
            subtree_left = self.build_decision_tree(left_subset, remaining_attributes, class_name, current_depth + 1)
            subtree_right = self.build_decision_tree(right_subset, remaining_attributes, class_name, current_depth + 1)
            # Construct the numerical attribute sub-tree with threshold conditions
            tree[best_attribute]["<=" + str(threshold)] = subtree_left
            tree[best_attribute][">" + str(threshold)] = subtree_right

        return tree
    
    def fit(self, dataset, class_name):
        """
        Parameters: dataset, class_name
        """
        attributes = self.get_attributes(dataset)
        self.tree = self.build_decision_tree(dataset, attributes, class_name)

    #######################################################################################
    # TO PRINT THE TREE IN THE DESIRED FORMAT
    #######################################################################################

    def print_tree(self, tree, indent=""):
        """
        Prints the decision tree in a structured format with attributes wrapped in <>.
        Parameters:
        - tree (dict): The decision tree to be printed.
        """
        if not isinstance(tree, dict):
            # If the tree is not a dictionary, it's a leaf node (class, count)
            print(f"{tree[0]} ({tree[1]})")
            return

        for attribute, subtree in tree.items():
            print(f"{indent}<{attribute}>")
            for value, subsubtree in subtree.items():
                if isinstance(subsubtree, tuple):
                    # If it's a leaf node
                    print(f"{indent}    {value}: {subsubtree[0]} ({subsubtree[1]})")
                else:
                    # If it's an internal node, print the value and recursively call print_tree
                    print(f"{indent}    {value}:")
                    self.print_tree(subsubtree, indent + "        ")

    #######################################################################################
    # HANDLING PREDICTIONS FOR TEST DATA
    #######################################################################################
    def predict_instance(self, instance, tree):
        """Predict the class for a single instance using the decision tree."""
        # Traverse the tree until reaching a leaf node (class label)
        while isinstance(tree, dict):
            attribute, subtree_dict = next(iter(tree.items()))
            attribute_value = instance[attribute]
            
            if self.attribute_type(pd.DataFrame([instance]), attribute) == 'numerical':
                threshold = float(list(subtree_dict.keys())[0].replace("<=", ""))
                if attribute_value <= threshold:
                    tree = subtree_dict["<=" + str(threshold)]
                else:
                    tree = subtree_dict[">" + str(threshold)]
            else:
                if attribute_value in subtree_dict:
                    tree = subtree_dict[attribute_value]
                else:
                    return None
        return tree 

    def predict(self, test_data):
        """Predict the classes for multiple instances using the decision tree."""
        predictions = []
        for _, instance in test_data.iterrows():
            prediction = self.predict_instance(instance, self.tree)
            if prediction:
                predictions.append(prediction[0])
            else:
                predictions.append(None)
        return predictions

    #######################################################################################

    def entropy(self, dataset, class_name):
        """
        Parameters: dataset and class name whose entropy is to be found
        Returns: An entropy of the class/target
        """
        class_counts = self.attribute_value_counts_simple(dataset,class_name)
        entropy = 0.0
        for class_count in class_counts[0].values():
            if(class_count > 0):
                probability = class_count / class_counts[1]
                entropy -= probability * math.log2(probability)
        return entropy

    def get_best_attribute(self, dataset, attributes, class_name):
        """
        Parameters: dataset, list of attributes 
        Returns: attribute with highest information gain 
        """
        best_attribute = attributes[0]
        curr_information_gain = self.information_gain_of_attribute(dataset, attributes[0], class_name)
        for i in range(0,len(attributes)):
            if(self.information_gain_of_attribute(dataset, attributes[i],class_name) > curr_information_gain):
                best_attribute = attributes[i]
                curr_information_gain = self.information_gain_of_attribute(dataset, attributes[i],class_name)
        return best_attribute
    
    def information_gain_of_attribute(self, dataset, attribute, class_name):
        """
        Parameter: dataset and attribute
        Returns: information gain of the provided attribute
        """
        if(self.attribute_type(dataset, attribute) == "categorical"):
            return self.information_gain_of_categorical_attribute(dataset, attribute, class_name)
        elif(self.attribute_type(dataset, attribute) == "numerical"):
            return self.information_gain_of_numerical_attribute(dataset, attribute, class_name)
        
    #######################################################################################
    # HANDLING INFORMATION GAIN FOR NUMERICAL ATTRIBUTE AND GETTING THRESHOLD
    #######################################################################################
    def numerical_information_gain_and_threshold(self, dataset, numerical_attribute, class_name):
        parent_entropy = self.entropy(dataset, class_name)
        sorted_dataset = dataset.sort_values(numerical_attribute)       
        best_info_gain = 0
        best_threshold = None       
        for i in range(len(sorted_dataset) - 1):
            # Calculate midpoint as potential threshold
            threshold = (sorted_dataset[numerical_attribute].iloc[i] + sorted_dataset[numerical_attribute].iloc[i + 1]) / 2           
            # Split the dataset based on the threshold
            left_subset = sorted_dataset[sorted_dataset[numerical_attribute] <= threshold]
            right_subset = sorted_dataset[sorted_dataset[numerical_attribute] > threshold]           
            left_entropy = self.entropy(left_subset, class_name)
            right_entropy = self.entropy(right_subset, class_name)           
            # Calculate information gain
            info_gain = parent_entropy - (len(left_subset) / len(sorted_dataset)) * left_entropy \
                                        - (len(right_subset) / len(sorted_dataset)) * right_entropy          
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold   
        return best_info_gain, best_threshold
 
    def information_gain_of_numerical_attribute(self, dataset, numerical_attribute, class_name):
        return self.numerical_information_gain_and_threshold(dataset, numerical_attribute, class_name)[0]
    
    def threshold_of_numerical_attribute(self, dataset, numerical_attribute, class_name):
        return self.numerical_information_gain_and_threshold(dataset, numerical_attribute, class_name)[1]
    
    #######################################################################################
    # HANDLING INFORMATION GAIN FOR CATEGORICAL ATTRIBUTE
    #######################################################################################
    def information_gain_of_categorical_attribute(self, dataset, categorical_attribute, class_name):
        """
        Parameters: dataset, attribute whose information gain is to be determined, class of the cataset
        Returns: Information gain of the provided attribute
        """
        information_gain = self.entropy(dataset, class_name)
        attribute_info = self.attribute_value_counts(dataset, categorical_attribute, class_name)
        attribute_info_dict = attribute_info[0]
        attribute_info_count = attribute_info[1]
        for attribute_value in attribute_info_dict.values():
            attribute_val_entropy = 0
            for class_count in attribute_value[0].values():
                if(class_count > 0):
                    probability = class_count / attribute_value[1]
                    attribute_val_entropy -= probability * math.log2(probability)
            information_gain -= (attribute_value[1] / attribute_info_count) * attribute_val_entropy
        return information_gain
    
    #######################################################################################
    # HELPER METHODS
    #######################################################################################   
    def get_attributes(self, dataset):
        """
        Parameters: dataset
        Returns: list: List of all attributes.
        """
        return list(dataset.columns[:-1])
    
    def attribute_type(self, dataset, attribute):
        """
        Parameters: dataset and attribute
        Returns: numerical if an attribute is numerical, categorical if an attribute is categorical
        """
        dtype = dataset[attribute].dtype
        if np.issubdtype(dtype, np.number):
            return 'numerical'
        else:
            return 'categorical'
        
    def attribute_value_counts_simple(self, dataset, attribute):
        """
        Parameters: dataset and attribute whose values are to be checked
        Returns: a tuple like this for example:- ({'yes': 9, 'no': 5}, 14)
        """
        dataset = dataset.fillna("None")
        attribute_series = dataset[attribute]
        value_counts = attribute_series.value_counts().to_dict()
        total_values = len(attribute_series)
        return value_counts, total_values
    
    def attribute_value_counts(self, dataset, attribute, class_name):
        """
        Parameters: dataset, attribute and class
        Returns: a tuple like this for example:- ({'overcast': 
        ({'no': 0, 'yes': 4}, 4), 'rainy': ({'no': 2, 'yes': 3}, 5), 'sunny': ({'no': 3, 'yes': 2}, 5)}, 14)
        """
        dataset = dataset.fillna("None")
        grouped = dataset.groupby(attribute)[class_name].value_counts().unstack(fill_value=0)
        attribute_value_counts = {}
        total_attribute_values = len(dataset[attribute])
        for attr_value, class_counts in grouped.iterrows():
            class_counts_dict = class_counts.to_dict()
            total_count = sum(class_counts_dict.values())
            attribute_value_counts[attr_value] = (class_counts_dict, total_count)
        return attribute_value_counts, total_attribute_values
 
    def split_dataset(self, dataset, attribute, attribute_value):
        """
        Parameters: dataset, attribute of a dataset and attribute_value
        Returns: Splitted dataset based on attribute and attribute value, and the remaining dataset
        """
        filtered_dataset = dataset.copy()
        remaining_dataset = dataset.copy()
        if self.attribute_type(dataset, attribute) == "categorical":
            filtered_dataset = filtered_dataset[filtered_dataset[attribute] == attribute_value]
            filtered_dataset = filtered_dataset.drop(columns=[attribute])
            remaining_dataset = remaining_dataset[remaining_dataset[attribute] != attribute_value]
        elif self.attribute_type(dataset, attribute) == "numerical":
            filtered_dataset = filtered_dataset[filtered_dataset[attribute] <= attribute_value]
            filtered_dataset = filtered_dataset.drop(columns=[attribute])
            remaining_dataset = remaining_dataset[remaining_dataset[attribute] > attribute_value]
            remaining_dataset = remaining_dataset.drop(columns=[attribute])
        return filtered_dataset, remaining_dataset
