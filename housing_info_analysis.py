import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def gradient_descent(X, y, beta_coef, learning_rate, m):
	x_transpose =X.transpose()
	hypothesis =np.dot(X, beta_coef)
	loss =hypothesis -y
	J =np.sum(loss**2)/(2*m)
	gradient =np.dot(x_transpose, loss)/m

	beta_coef =beta_coef -learning_rate*gradient 
	return beta_coef

if __name__ == '__main__':

	df =pd.read_csv('houses.csv')
	X =df.loc[:, df.columns != 'price (grands)']
	X_2 =df.loc[:, 'sqft_living'].as_matrix()
	y_ =df.iloc[:,:1].as_matrix()
	y2 =y_.flatten()
	# m =float(len(y2))

	print(np.shape(X_2), '\n')
	m=np.shape(X_2)


	mean =np.mean(X_2, axis =0)
	std =np.std(X_2, axis =0)
	std_data =(X_2 -mean)/std

	std_data =np.c_[np.ones(m), std_data]


	beta_coef = np.ones(2)
	numIterations =100000
	learning_rate =0.00000000001
	# set up ridge regression parameter
	alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
	for iter in range(0, numIterations):
		beta_coef =gradient_descent(std_data, y2, beta_coef, learning_rate, m)

	print(beta_coef)




	















