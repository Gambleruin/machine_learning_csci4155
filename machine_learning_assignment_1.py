# machine learning assignment 1
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# this is the data for training purposes
train_data =np.loadtxt('wine_train.txt', delimiter =',')
# test_data =np.loadtxt('wine_test.txt', delimiter =',')

# remove labels from test data
# test_data =np.delete(test_data, 0, axis =1)
# print(test_data)

features =train_data[:, 1:]
target =np.int32(train_data[:, 0])
X_train, X_test, y_train, y_test =train_test_split(
	features, target, test_size =0.2, random_state =42)

# choosing svm for classification
# model =SVC(kernel ='linear', C =1).fit(X_train, y_train)
# model =SVC(kernel='rbf', C=1.2).fit(X_train, y_train)
model =RandomForestClassifier(n_estimators =10).fit(X_train, y_train)
svm_predictions =model.predict(X_test)

# model accuracy for X_test
accuracy =model.score(X_test, y_test)
print(accuracy)



