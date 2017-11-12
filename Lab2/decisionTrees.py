import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# loading file with data 
def load_data(filename):
	return pd.read_csv(filename, header=None).values
	
# splitting data on training and test sets
def split_dataset(test_size):
	dataset = load_data('data.csv')
	occ_class = dataset[:, 0]
	occ_attr = dataset[:, 1:]	
	occ_class = occ_class.astype(np.float)
	occ_attr = occ_attr.astype(np.float)
	data_train, data_test, class_train, class_test = train_test_split(occ_attr, occ_class, test_size=test_size, random_state=55)
	return data_train, class_train, data_test, class_test

def main():
    for size in np.arange(0.1, 0.4, 0.1):
        data_train, class_train, data_test, class_test = split_dataset(size)
        print('Size: ', round(size,1))
        decisionForest = DecisionTreeClassifier() 
        decisionForest = decisionForest.fit(data_train, class_train)
        decisionAcc = decisionForest.score(data_test, class_test)
        print('Decision Tree accuracy: ', round(decisionAcc,10))
        randonForest = RandomForestClassifier()
        randonForest = randonForest.fit(data_train, class_train)
        randomAcc = randonForest.score(data_test, class_test)
        print('Random Tree accuracy: ', round(randomAcc,10))
        print('______________________________')

main()