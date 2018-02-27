'''
Due to the weakness of the k-means model, we are motivated to 
use GMM, A Guassian mixture model is very similar to k-means: 
it uses an expectation-maximization approach which qualitatively 
does the following:

	1. Choose starting guesses for the location and shape

	2. Repeat until converged:

		1. E-step: for each point, find weights encoding the probability
		   of membership in each cluster
		2. M-step: for each cluster, update its (where the contour is at)location, normalization, 
		   and shape based on all data points, making use of the weights

This algorithm can sometimes miss the globally optimal solution, and thus in practice multi-
ple random initializations are used. 
'''
import math
import numpy as np 
import random as rd
import matplotlib.pyplot as plt 
from sklearn import mixture
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import pairwise_distances_argmin

'''
def initialize_clusters(data, n_clusters, rseed =2):
	rng =np.random.RandomState(rseed)
	i =rng.permutation(data.shape[0])[:n_clusters]
	contour =data[i]
	return contour, labels
'''


def initialize_data(K, data):
	# initialize the mu_k randomly
	cols =(data.shape)[1]
	# multi-variant gaussian distribution
	mu =np.zeros((K, cols))
	for row in range(K):
		idx =int(np.floor(rd.random()*len(data)))
		for col in range(cols):
			mu[row][col] +=data[idx][col]

	sigma =[]
	for k in range(K):
		sigma.append(np.cov(data.T))

	# initialize the pi_k randomly, the implicit label is as order: 0, 1, 2
	sum_pi =1.0
	pi =np.zeros(K)
	pi +=sum_pi/K
	return mu, sigma, pi

def prior_prob(K, mu, sigma, pi, data):
	pb =0.0
	for k in range(K):
		pb +=pi[k]*gdf(data, mu[k], sigma[k])
	return pb 

def gdf(x, mu, sigma):
	score =0.0
	x_mu =np.matrix(x -mu)
	inv_sigma =np.linalg.inv(sigma)
	det_sqrt =np.linalg.det(sigma)**0.5

	norm_const =1.0/((2*np.pi)**(len(x)/2)*det_sqrt)
	exp_value =math.pow(math.e, -0.5*(x_mu*inv_sigma*x_mu.T))
	result =norm_const*exp_value

	return result

def likelihood(K, mu, sigma, pi, data):
	'''
	Calculate the log likelihood
	'''
	log_score =0.0
	for n in range(len(data)):
		log_score +=np.log(prior_prob(K, mu, sigma, pi, data[n]))

	return log_score 

def gmm(K, data):
	mu, sigma, pi =initialize_data(K, data)
	log_score =likelihood(K, mu, sigma, pi, data)
	log_score_0 =log_score 
	threshold =0.001
	i =0
	max_iter =100
	while i <max_iter:
		r_nk =e_step(K, mu, sigma, pi, data)
		mu, sigma, pi =m_step(K, r_nk, data)
		new_log_score =likelihood(K, mu, sigma, pi, data)
		if abs(new_log_score -log_score) <threshold:
			break

		print ("|", i+1, "|", log_score, "|", new_log_score)
		log_score =new_log_score

		i +=1

	# print ("converged\n\n") 
	
	return r_nk

def e_step(K, mu, sigma, pi, data):
	'''
	Evaluate the responsibility using the current parameter 
	value
	'''
	r_nk =np.zeros((len(data), K))
	for i in range(len(data)):
		for k in range(K):
			r_nk[i][k] =(pi[k]*gdf(data[i], mu[k], sigma[k]))/prior_prob(K, mu, sigma, pi, data[i])

	return r_nk

def m_step(K, r_nk, data):
	'''
	re-estimate the parameters using the current
	responsibility
	'''
	# print(K)
	N_k =np.zeros(K)
	cols =(data.shape)[1]
	new_mu_k =np.zeros((K, cols))
	for k in range(K):
		for n in range(len(data)):
			N_k[k] += r_nk[n][k]
			new_mu_k[k] +=(r_nk[n][k]*data[n])

		new_mu_k[k] /=N_k[k]

	new_sigma_k =np.zeros((K, cols, cols))
	for k in range(K):
		for n in range(len(data)):
			xn =np.zeros((1, 4))
			mun =np.zeros((1, 4))
			xn +=data[n]
			mun +=new_mu_k[k]
			x_mu =xn -mun
			new_sigma_k[k] +=(r_nk[n][k]*x_mu*x_mu.T)
		new_sigma_k[k] /=N_k[k]

	new_pi_k =np.zeros(3)
	for k in range(K):
		new_pi_k[k] +=(N_k[k]/len(data))

	return new_mu_k, new_sigma_k, new_pi_k

if __name__ == "__main__":
	# data for unpervised learning using EM algorithm
	iris_data =np.loadtxt('iris_data.txt', delimiter =',')
	K =3
	# this is the data 
	observable_var =iris_data[:,0:4]
	'''
	g =mixture.GaussianMixture(n_components =3)
	g.fit(observable_var)
	GMM_predict =g.predict(observable_var)
	print (GMM_predict)
	'''
	plot_data =[]
	# this is the RGB color 
	color_list= gmm(K, observable_var)
	col_list =color_list.tolist()
	# predicted_label =[np.argmax(elem) for elem in final_posterior_estimate]
	# pre_l =np.transpose(predicted_label)
	feature_list =iris_data[:, 0:2]
	fea_list =feature_list.tolist()
	for feature, color in zip(fea_list, col_list):
		plot_data.append(feature +color)

	plot_data_arr =np.asarray(plot_data)
	pda =np.reshape(plot_data_arr, (150, 5))
	for t in pda:
		plt.scatter(t[0], t[1], c =(t[2], t[3], t[4]))

	plt.legend(loc ='upper right')
	plt.show()
	
	
	
	

	




	


