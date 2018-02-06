import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pylab
from scipy import stats


'''
def linear_regression_ridge_momentum(X, y, m_current, b_current, epochs =1000, learning_rate =0.0001, ridge_alpha, activation ="gradient_descent"):
	N =float(len(y))
	mu =0.9
	v_m =0
	v_b =0
	for i in range(epochs):
		y_current =(m_current*X) +b_current
		cost =sum([i**2 for i in (y -y_current)])/N
		m_gradient = -(2/N) * sum(X * (y - y_current))
		b_gradient = -(2/N) * sum(y - y_current)
		if activation == "gradient_descent":
			m_current = m_current*(1 -2*ridge_alpha*learning_rate) - (learning_rate * m_gradient)
			b_current = b_current*(1 -2*ridge_alpha*learning_rate) - (learning_rate * b_gradient)
		elif activation == "ridge"
			m_current = m_current - (learning_rate * m_gradient)
			b_current = b_current - (learning_rate * b_gradient)
		elif activation == "momentum"
			v_m =mu*v_m +(learning_rate * m_gradient)
			v_b =mu*v_b +(learning_rate * b_gradient)
			m_current = m_current*(1 -2*ridge_alpha*learning_rate) - v_m
			b_current = b_current*(1 -2*ridge_alpha*learning_rate) - v_b


	return m_current, b_current, cost
'''

def gradient_descent(X, y, beta_coef, learning_rate):
	m = X.shape[0]

	x_transpose =X.transpose()
	hypothesis =np.dot(X, beta_coef)
	loss =hypothesis -y
	J =np.sum(loss**2)/(2*m)
	gradient =np.dot(x_transpose, loss)/m
	beta_coef =beta_coef -learning_rate*gradient
	print(beta_coef) 
	return beta_coef

if __name__ == '__main__':

	df =pd.read_csv('houses.csv')
	X =df.loc[:, df.columns != 'price (grands)']
	X_2 =df.loc[:, 'sqft_living'].as_matrix()
	y_ =df.iloc[:,:1].as_matrix()
	y2 =y_.flatten()
	# m =float(len(y2))
	n=np.shape(X_2)

	minmax = (X_2 - X_2.min()) / (X_2.max() - X_2.min())
	scaled_y = (y2 - y2.min()) / (y2.max() - y2.min())
	n=np.shape(minmax)
	minmax =np.c_[np.ones(n), minmax]


	beta_coef = np.ones(2)
	print(beta_coef)

	numIterations =10000
	learning_rate =0.001
	# set up ridge regression parameter
	alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
	
	for iter in range(0, numIterations):
		beta_coef =gradient_descent(minmax, scaled_y, beta_coef, learning_rate)
	
	y_predict = beta_coef[0] + beta_coef[1]*X_2
	pylab.plot(X_2,y2,'o')
	pylab.plot(X_2,y_predict,'r')
	pylab.show()
	
	# using library
	'''
	from sklearn.linear_model import LinearRegression
	lm =LinearRegression()
	X_2 =np.c_[np.ones(n), X_2]
	
	lm.fit(minmax, y2)
	print(lm.coef_)
	'''






	















