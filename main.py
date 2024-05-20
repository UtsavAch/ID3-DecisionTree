import pandas as pd
from id3 import DecisionTree
from pre_process import train_test_split, get_target_name
from metrics import confusion_matrix, evaluate_metrics

# Load the dataset
file_path = 'weather.csv'
#file_path = 'iris.csv'
#file_path = 'restaurant.csv'
#file_path = 'connect4.csv'
data = pd.read_csv(file_path)

# Remove 'ID' column from the dataset
#For connect4 dont remove anything
data = data.drop('ID', axis=1)

# Instantiate and fit the DecisionTree model
tree_model = DecisionTree()
train, test = train_test_split(data, 0.8)
tree_model.fit(train, get_target_name(data))
print(tree_model.tree)
#print(tree_model.attribute_value_counts_simple(data, "Pat"))

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

# test_data_iris = pd.DataFrame({
#     'sepallength': [5.1, 4.9, 4.7, 4.6],
#     'sepalwidth': [3.5, 3.0, 3.2, 3.1],
#     'petallength': [1.4, 2, 1.3, 1.5],
#     'petalwidth': [0.2, 0.2, 0.2, 0.2],
# })
print(test)
predicted_labels=tree_model.predict(test)
true_labels = test[test.columns[-1]].tolist()
evaluate_metrics(true_labels, predicted_labels)
#print(true_labels)
#print(confusion_matrix(true_labels, predicted_labels))
#print(tree_model.predict(test_data_iris))
#To Do:- Format the tree as in the assignment