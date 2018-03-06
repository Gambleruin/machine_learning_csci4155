# multi_layer_perceptron
import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
'''
with open('pattern1.csv') as f:
	for line in f:
		print(type(line.rstrip()))
'''
count =0
# the column header will never be needed
num_instance =26
df =pd.read_csv('pattern1.csv', names =[' '])
# print (df.iloc[0: 12])
for i in range(num_instance):
	training_instance =df.iloc[count: count+12].values
	print ('this is ', i, 'th traning instance\n', training_instance, '\n')
	count =count +12


# design Multiple Layer perceptron to read in ndarray as training instance
X=np.array([[0,0,1,1],
	[0,1,0,1],
	[1,1,1,1]])
Y =np.array([[1,0,0,1]])

# model specifications
Ni =3; Nh =4; No =1;

# parameter and array initialization
Ntrials =10000
wh =np.random.randn(Nh, Ni); dwh =np.zeros(wh.shape)
wo =np.random.randn(No, Nh); dwo =np.zeros(wo.shape)
error =np.array([])

for trial in range(Ntrials):
	h =1/(1+np.exp(-wh@X)) #hidden activation for all pattern
	y =1/(1+np.exp(-wo@h)) #output for all pattern
	#print(y)
	#break

	do =y*(1-y)*(Y-y) #delta output
	dh =h*(1-h)*(wo.transpose()@do) #delta backpropagated 

	# update weights with mometum
	dwo =0.9*dwo +do@h.T 
	wo =wo +0.1*dwo
	dwh =0.9*dwh +dh@X.T 
	wh=wh+0.1*dwh

	error =np.append(error, np.sum(abs(Y-y)))


plt.plot(error)
plt.show()
