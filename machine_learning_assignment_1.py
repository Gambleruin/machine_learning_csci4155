# machine learning assignment 1
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# this is the data for training purposes
train_data =np.loadtxt('wine_train.txt', delimiter =',')
test_data =np.loadtxt('wine_test.txt', delimiter =',')

# remove labels from test data
test_data_no_lable =np.delete(test_data, 0, axis =1)
# print(test_data)

features =train_data[:, 1:]
target =np.int32(train_data[:, 0])
X_train, X_test, y_train, y_test =train_test_split(
	features, target, test_size =0.2, random_state =42)

# choosing svm for classification
# model =SVC(kernel ='linear', C =1).fit(X_train, y_train)
# model =SVC(kernel='rbf', C=1.2).fit(X_train, y_train)
k_fold =KFold(n_splits =5)
model_rf =RandomForestClassifier(n_estimators =10).fit(X_train, y_train)
print(cross_val_score(model_rf, X_train, y_train, cv =k_fold, n_jobs =-1), '\n')

model =SVC(kernel='rbf', C=1).fit(X_train, y_train)
print(cross_val_score(model_rf, X_train, y_train, cv =k_fold, n_jobs =-1), '\n')

final_prediction =model.predict(test_data_no_lable)
print(final_prediction)



