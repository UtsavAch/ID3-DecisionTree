import pandas as pd
import time
import random


def get_class_imbalance(dataset):
    class_col=dataset.columns[-1]
    names_class= dataset[class_col].unique()
    counts_class = {}
    for value in dataset[class_col]:
        counts_class[value] = counts_class.get(value, 0) + 1

    percentages_class={}
    for name in names_class:
        percentages_class[name]=round(counts_class.get(name)/dataset.shape[0], 2)

    return percentages_class,counts_class




def create_train_test(dataset, counts_class_train, counts_class_test):
    dataset_cop = dataset.sample(frac=1, random_state=int(time.time())).reset_index(drop=True)
    class_col = dataset_cop.columns[-1]
    train = pd.DataFrame(columns=dataset_cop.columns)
    test = pd.DataFrame(columns=dataset_cop.columns)
    
    for index, row in dataset_cop.iterrows():
        if counts_class_train[row[class_col]] > 0:
            train = pd.concat([train, row.to_frame().T], ignore_index=True)
            counts_class_train[row[class_col]] -= 1
        elif counts_class_test[row[class_col]] > 0:
            test = pd.concat([test, row.to_frame().T], ignore_index=True)
            counts_class_test[row[class_col]] -= 1

    return train,test

 
def get_target_name(dataset):
    return dataset.columns[-1]



def train_test_split(dataset,train_size):
    percentages_class,counts_class=get_class_imbalance(dataset)
    #print(f"Class imbanlance do dataset original: {percentages_class, counts_class}")
    train_size=round(train_size*dataset.shape[0])
    test_size=dataset.shape[0]-train_size

    counts_class_train={}
    counts_class_test={}
    for name in counts_class:
        counts_class_train[name]=round(train_size*percentages_class[name])
    
    for name in counts_class:
        counts_class_test[name]=round(test_size*percentages_class[name])

    train,test= create_train_test(dataset=dataset, counts_class_train=counts_class_train, counts_class_test=counts_class_test)
    
    return train, test

