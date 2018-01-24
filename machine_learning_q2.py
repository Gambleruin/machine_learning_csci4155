import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn import svm, metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

iris_data =np.loadtxt('iris_data.txt', delimiter =',')
train_x =iris_data[0: -1:2,0:4]
train_y =np.int32(iris_data[0:-1:2,4])
test_x =iris_data[1:-1:2,0:4]
test_y =np.int32(iris_data[1:-1:2,4])

# model
model =svm.SVC(kernel ='linear')
# train
model.fit(train_x, train_y)
# prediction
predicted_y =model.predict(test_x)
# evaluation
# print('percentage_correct_(accuracy) of svm is: \n)', np.mean(test_y ==predicted_y))

# the k-fold cross validation method (also called just cross validation) is a resampling
# method that provides a more accurate estimate of algorithm performance. 
vali_ready =iris_data[0: -1:2]



# self_customised cross validation
def cross_validation_split(train_data, K, randomise =False):
	for k in range(K):
		training =[x for i, x in enumerate(train_data) if i%K !=k]
		validation =[x for i, x in enumerate(train_data) if i%K ==k]
		yield training, validation 

if __name__ == '__main__':
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
		# print('percentage of accuracy to the model is: \n', np.mean(iter_test_y ==predicted_y), '\n')
		# print(validation[:,0:4], '\n\n\n')
		
	# print(training.shape, '\n', validation.shape )

	# print(iris_data.shape,'\n', train_y)
	# print(vali_ready.shape)
	# print(train_y)
	# validated_train_x =train
	k_fold =KFold(n_splits =5)
	# for train_indices, test_indices in k_fold.split(vali_ready):
		# print('Train: %s | test: %s'%(train_indices, test_indices), '\n')

	# using model to fit k folds of validation sets with default function
	k_fold_cross_validation_score =[model.fit(train_x[train], train_y[train]).score(train_x[test], train_y[test])
		for train ,test in k_fold.split(train_x)]

	# alternatively
	print(cross_val_score(model, train_x, train_y, cv =k_fold, n_jobs =-1))

		
	
