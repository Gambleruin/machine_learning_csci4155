import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn import svm, metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

iris_data =np.loadtxt('iris_data.txt', delimiter =',')
train_x =iris_data[0: -1:2,0:4]
train_y =np.int32(iris_data[0:-1:2,4])
# test_x =iris_data[1:-1:2,0:4]
# test_y =np.int32(iris_data[1:-1:2,4])
# the k-fold cross validation method (also called just cross validation) is a resampling
# method that provides a more accurate estimate of algorithm performance. 
vali_ready =iris_data[0: -1:2]
# self_customised cross validation function
def cross_validation_split(train_data, K, randomise =False):
	for k in range(K):
		training =[x for i, x in enumerate(train_data) if i%K !=k]
		validation =[x for i, x in enumerate(train_data) if i%K ==k]
		yield training, validation 

if __name__ == '__main__':
	k_fold =KFold(n_splits =5)
	for training, validation in cross_validation_split(vali_ready, 5):
		training =np.asarray(training)
		validation =np.asarray(validation)
		iter_train_x =training[:,0:4]
		iter_train_y =np.int32(training[:,-1])
		iter_test_x =validation[:,0:4]
		iter_test_y =np.int32(validation[:,-1])

		model =svm.SVC(kernel ='linear')
		model.fit(iter_train_x, iter_train_y)
		predicted_y =model.predict(iter_test_x)
		print('the customised cv score per each fold: \n', np.mean(iter_test_y ==predicted_y), '\n')

	
	k_fold =KFold(n_splits =5)
	# comparing cv result with native function
	k_fold_cross_validation_score =[model.fit(train_x[train], train_y[train]).score(train_x[test], train_y[test])
		for train ,test in k_fold.split(train_x)]

	print('the native cv score is:\n', k_fold_cross_validation_score)


	model_rf =RandomForestClassifier(n_estimators =10).fit(train_x, train_y)
	# comparing the result between svm and random_forest
	print('the comparison of cv score between svm and rf is:\n') 
	print(cross_val_score(model, train_x, train_y, cv =k_fold, n_jobs =-1), '\n')
	print(cross_val_score(model_rf, train_x, train_y, cv =k_fold, n_jobs =-1))

		
	
