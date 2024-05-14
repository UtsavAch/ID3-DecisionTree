import pandas as pd
from id3 import DecisionTree;

# Load the dataset
# file_path = 'weather.csv'
# file_path = 'iris.csv'
file_path = 'restaurant.csv'
# file_path = 'connect4.csv'
data = pd.read_csv(file_path)

# Remove 'ID' column from the dataset
#For connect4 dont remove anything
data = data.drop('ID', axis=1)

# Instantiate and fit the DecisionTree model
tree_model = DecisionTree()
tree_model.fit(data, class_name='Class')

print(tree_model.tree)
print(tree_model.attribute_value_counts(data, "Pat", "Class"))

# test_data = pd.DataFrame({
#     'Weather': ['sunny', 'overcast', 'rainy', 'sunny'],
#     'Temp': [85, 75, 72, 75],
#     'Humidity': [85, 70, 90, 70],
#     'Windy': [False, True, True, False]
# })

# test_data_restaurant = pd.DataFrame({
#     'Alt': ['Yes','No','Yes'],
#     'Bar':['Yes','No','Yes'],
#     'Fri':['Yes','No','Yes'],
#     'Hun':['Yes','No','Yes'],
#     'Pat':['Full', 'None', 'Full'],
#     'Price':['$$$', '$', '$'],
#     'Rain':['No', 'No','No'],
#     'Res':['Yes', 'No', 'No'],
#     'Type':['Italian', 'Thai', 'Burger'],
#     'Est':['10-30','0-10', '30-60']
# })

test_data_iris = pd.DataFrame({
    'sepallength': [5.1, 4.9, 4.7, 4.6],
    'sepalwidth': [3.5, 3.0, 3.2, 3.1],
    'petallength': [1.4, 2, 1.3, 1.5],
    'petalwidth': [0.2, 0.2, 0.2, 0.2],
})

# print(tree_model.predict(test_data_iris))

#To Do:- Need to add depth to the tree