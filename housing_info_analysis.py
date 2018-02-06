import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from scipy import stats

def gradient_descent(X, y, beta_coef, learning_rate):
	m = X.shape[0]

	x_transpose =X.transpose()
	hypothesis =np.dot(X, beta_coef)
	loss =hypothesis -y
	J =np.sum(loss**2)/(2*m)
	gradient =np.dot(x_transpose, loss)/m
	beta_coef =beta_coef -learning_rate*gradient
	# print(beta_coef) 
	return beta_coef

def gradient_descent_ridge(X, y, beta_coef, learning_rate, ridge_alpha):
	m = X.shape[0]

	x_transpose =X.transpose()
	hypothesis =np.dot(X, beta_coef)
	loss =hypothesis -y
	J =np.sum(loss**2)/(2*m)
	gradient =np.dot(x_transpose, loss)/m
	beta_coef =beta_coef*(1-2*learning_rate*ridge_alpha) -learning_rate*gradient
	print(beta_coef) 
	return beta_coef

def gradient_descent_ridge_added_on_momentum(X, y, beta_coef, learning_rate, ridge_alpha):
	m = X.shape[0]
	mu =0.9
	v =np.zeros(3)

	x_transpose =X.transpose()
	hypothesis =np.dot(X, beta_coef)
	loss =hypothesis -y
	J =np.sum(loss**2)/(2*m)
	gradient =np.dot(x_transpose, loss)/m
	v =mu*v +learning_rate*gradient

	beta_coef =beta_coef*(1-2*learning_rate*ridge_alpha) -v
	print(beta_coef) 
	return beta_coef

# modified data including interaction with quatratic term(with base function intact)


if __name__ == '__main__':

	num_row =21613
	df =pd.read_csv('houses.csv')
	X_2 =df.loc[:, 'sqft_living'].as_matrix()
	X_3 =df.ix[:, df.columns != 'price (grands)']
	y_ =df.iloc[:,:1].as_matrix()
	y2 =y_.flatten()

	minmax = (X_3 - X_3.min()) / (X_3.max() - X_3.min())
	scaled_y = (y2 - y2.min()) / (y2.max() - y2.min())
	n=np.shape(minmax)

	# minmax =np.c_[np.ones(n), minmax]

	# for multiple_regression
	add_c =np.ones(num_row)
	add_c_2d =np.reshape(add_c, (-1, 1))
	print(n, '\n', np.shape(add_c_2d))
	# print(add_c_2d, '\n')
	mul_X =np.append(add_c_2d, minmax, axis = 1)
	X_3 =np.append(add_c_2d, X_3, axis = 1)
	print(np.shape(mul_X))

	beta_coef = np.ones(16)
	print(beta_coef)

	numIterations =1000
	learning_rate =0.1
	# set up ridge regression parameter
	alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

	'''
	# set up quotratic term
	Q =df.loc[:, 'sqft_living']**2
	X =df.loc[:, 'sqft_living']
	Q_term =pd.concat([X, Q], axis =1)
	poly_X =Q_term.as_matrix()

	# scaling on polynomial input
	poly_minmax = (poly_X - poly_X.min()) / (poly_X.max() - poly_X.min())
	add_c =np.ones(n)
	add_c_2d =np.reshape(add_c, (-1, 1))
	# print(add_c_2d, '\n')
	poly_X =np.append(add_c_2d, poly_minmax, axis = 1)
	print(poly_X)
	# poly_minmax =np.c_[np.ones(n0), poly_minmax]
	# print(np.shape(add_c), np.shape(poly_minmax))
	'''
	


	
	for iter in range(0, numIterations):
		beta_coef =gradient_descent(mul_X, scaled_y, beta_coef, learning_rate)

	print(beta_coef)
	
	y_predict = np.dot(X_3, beta_coef)
	# print( y_predict, '\n', y2)

	










'''
	plt.xlim(min(X_2), max(X_2))
	plt.ylim(min(y_predict), max(y_predict))
	# plt.plot(X_2,y2,'o')
	plt.plot(X_2,y_predict,'r')
	plt.show()
'''

	
	
	
	
	
	






	















