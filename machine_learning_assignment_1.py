# machine learning assignment 1
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC

# this is the data for training purposes
train_data =np.loadtxt('wine_train.txt', delimiter =',')
# test_data =np.loadtxt('wine_test.txt', delimiter =',')

# remove labels from test data
# test_data =np.delete(test_data, 0, axis =1)
# print(test_data)

# getting all the features from wine data 
features =train_data[:, 1:]
print(features, '\n')

# getting all the target(y) value
target =np.int32(train_data[:, 0])
print(target)

# training/testing data split
X_train, X_test, y_train, y_test =train_test_split(
	features, target, test_size =0.15, random_state =42)

# choosing svm for classification
clf =SVC(kernel ='linear', C =1).fit(features, target)

# prediction from testing data 
# svm_predictions =clf.predict(test_data)

# model accuracy for X_test
# accuracy =clf.score()



