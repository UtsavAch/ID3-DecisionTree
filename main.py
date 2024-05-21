import pandas as pd
from id3 import DecisionTree
from pre_process import train_test_split, get_target_name

# Load the dataset
#file_path = 'weather.csv'
file_path = 'iris.csv'
#file_path = 'restaurant.csv'
# file_path = 'connect4.csv'
data = pd.read_csv(file_path)

#Remove 'ID' column from the dataset
#For connect4 dont remove anything
data = data.drop('ID', axis=1)

# Instantiate and fit the DecisionTree model
train, test = train_test_split(data, 0.8)
tree_model = DecisionTree()
# tree_model = DecisionTree(max_depth = 3)
tree_model.fit(data, get_target_name(data))
tree_model.print_tree(tree_model.tree)

