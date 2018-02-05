
import random
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# any difference if: from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split

df =pd.read_csv('houses.csv')
# could have been chosen for multiple regression
X =df.loc[:, df.columns != 'price (grands)']
# select one column to be the predictor
X_2 =df.loc[:, 'sqft_living'].as_matrix()
y_ =df.iloc[:,:1].as_matrix()
y2 =y_.flatten()

print(np.shape(X_2), '\n', np.shape(y2))

# normalize the input
normalized_data =(X_2 -np.mean(X_2,axis =0))/(np.max(X_2,axis =0)-np.min(X_2,axis =0))

# standize the data
mean =np.mean(X_2, axis =0)
std =np.std(X_2, axis =0)
std_data =(X_2 -mean)/std

X_1=np.array([937, 1150, 1170, 1290, 1275, 1410, 1550, 1730, 1910])
y1=np.array([187,  222,  330,  310,  290,  440,  600,  550,  600])
print(type(X_1), type(X_2),'\n')
print(type(y1), type(y2))
'''
TBF6D:Desktop daniel$ python3 housing_info_analysis.py 
(9,) (21613,) 
(9,) (21613,)
'''
'''
# initial theta
a=np.array([-1]); b=np.array([-1]); L=np.array([])
for iter in range(10-1):
    y_hat=a[-1]*X_2+b[-1]
    a=np.append(a,a[-1]-lr*sum((y_hat-y2)*X_2))
    b=np.append(b,b[-1]-lr*sum(y_hat-y2))
    L=np.append(L,sum((y_hat-y2)**2))
 '''

# set up ridge regression parameter
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

def gradient_descent(alpha, x, y, ep, max_iter=100000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    '''
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])
    '''

    t0 =0
    t1 =0

    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])
	# Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
        
        '''
        temp0 =t0*(1-2*5*alpha) -alpha*grad0
        temp1 =t1*(1-2*5*alpha) -alpha*grad1
        '''
    
        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) 

        if abs(J-e) <= ep:
            converged = True
    
        J = e   # update error 
        iter += 1  # update iter
    
        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0,t1

lr=0.000001
ep =0.01
theta0, theta1 =gradient_descent(lr, std_data, y2, ep, max_iter =100)
print (theta0, theta1)

y_hat =theta1*X_2 +theta0 

'''
plt.xlim([min(X_2),max(X_2)])
plt.ylim([min(y2),max(y2)])
'''
print(y2, '\n', y_hat)
plt.plot(X_2, y2, '*')
plt.plot(X_2, y_hat)

plt.show()












