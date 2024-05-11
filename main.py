import pandas as pd
from id3 import DecisionTree;

# Load the dataset
# file_path = 'weather.csv'
file_path = 'iris.csv'
# file_path = 'restaurant.csv'
data = pd.read_csv(file_path)

# Remove 'ID' column from the dataset
data = data.drop('ID', axis=1)

# Instantiate and fit the DecisionTree model
tree_model = DecisionTree()
tree_model.fit(data, class_name='class')

print(tree_model.tree)