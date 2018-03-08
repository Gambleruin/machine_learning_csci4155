import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def str_int(training_instance):
    t =[]
    for i in range(12):
        temp =training_instance[i][0].replace(' ', '')
        m =map(int, temp)
        training_instance[i][0] = np.fromiter(m, dtype =np.int)
        t.append(training_instance[i][0].tolist())

    training_instance =np.asarray(t)
        
    # np.vstack(training_instance)
    # print(training_instance)
    return training_instance

class NeuralNetwork:
    def __init__(self, layer, activation ):
        self.activation = sigmoid
        self.layer =layer
        # print(type(sigmoid))

    def init_param(self):
        n_in =self.layer[0]
        n_hidden =self.layer[1]
        n_out =self.layer[2]

        wh =np.random.randn(n_hidden, n_in)
        # b_0 =np.zeros(shape =(n_hidden, 1))
        wo =np.random.randn(n_out, n_hidden)
        # b_1 =np.zeros(shape =(n_out, 1))

        parameters ={'wh': wh,
                    'wo': wo}
        return parameters

    def delta_rule(self, X, Y, learning_rate):
        params =self.init_param()
        wh =params['wh']
        wo =params['wo']

        dwh =np.zeros(wh.shape)
        dwo =np.zeros(wo.shape)
        N_trial =10000
        error =np.array([])
        # forward and backward path
        for trial in range(N_trial):
            h =self.activation(np.dot(wh, X))
            y =self.activation(np.dot(wo, h))
            '''
            if (trial == N_trial):
                return y, wh, wo
            '''
            # this shows error is descending towards minimum
            err =np.sum(abs(Y-y))
            ##################
            print(err)
            ##################
            
            delta =y*(1-y)*(Y-y)
            # print(delta.shape,'\n\n')
            update_hidden =h*(1-h)*np.dot(wo.T,delta)
        
            # update weights with mometum
            dwo =learning_rate*dwo +np.dot(delta, h.T)
            wo =wo +0.1*dwo
            dwh =learning_rate*dwh +np.dot(update_hidden, X.T)
            wh =wh +0.1*dwh
            error=np.append(error,np.sum(abs(Y-y)))
            # print(error.astype(int))
        return y, wh, wo, error

if __name__ == '__main__':
    nn = NeuralNetwork([156,26,8], sigmoid)
    count =0
    asc =65
    num_instance =26
    df =pd.read_csv('pattern1.csv', names =[' '])
    # print (df.iloc[0: 12])
    Y =np.zeros((26, 8))
    for i in range(26):
        str_letter =''.join(str(1 & int(asc) >> i) for i in range(8)[::-1])
        for j in range(8):
            Y[i][j] =np.fromstring(str_letter[j],dtype =int, sep=',')
        asc =asc+1

    Y =Y.astype(np.int)
    Y =Y.T

    X =np.zeros((26, 156))
    for i in range(num_instance):
        data_sample =str_int(df.iloc[count: count+12].values)
        # print ('this is', i+1,'th traning instance\n', training_instance, '\n') 
        training_instance = np.reshape(data_sample.flatten(), (-1))
        t =np.reshape(training_instance, (1, -1)) 
        # print(t)
        X[i] =t
        count =count +12

    X =X.astype(np.int)
    X =X.T
    # print(X.shape)
    result =nn.delta_rule(X, Y, 0.9)
    l =result[0].T
    print(l)
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
    '''
[[  4.54849253e-08   1.00000000e+00   6.57715209e-15   3.80751662e-03
    5.00240083e-03   6.17306925e-07   7.23097894e-04   9.98789182e-01]
 [  5.81414945e-07   1.00000000e+00   5.46096993e-14   8.43652491e-04
    3.22275835e-03   3.13786452e-03   9.98816071e-01   2.35020900e-03]
 [  2.78519756e-07   1.00000000e+00   2.58164680e-14   1.85998761e-04
    3.35701792e-03   6.96215988e-03   9.98686339e-01   9.94396764e-01]
 [  4.71223990e-07   1.00000000e+00   4.29895986e-13   4.26873368e-03
    4.95367465e-03   9.99932352e-01   2.94871715e-03   2.66409109e-03]
 [  7.97038733e-09   1.00000000e+00   3.80483164e-17   8.20645823e-03
    6.34376823e-05   9.98162086e-01   6.14989602e-03   9.90541517e-01]
 [  1.15120052e-07   1.00000000e+00   3.01404655e-15   4.08687438e-03
    1.59099237e-04   9.92941386e-01   9.93028969e-01   7.42668047e-03]
 [  3.03201296e-07   1.00000000e+00   2.04673411e-14   3.74834405e-03
    2.10588329e-03   9.96009949e-01   9.93308581e-01   9.99947602e-01]
 [  2.07771076e-07   1.00000000e+00   8.93403196e-15   8.82610710e-06
    9.99442561e-01   3.02992057e-03   3.90348014e-03   4.52779105e-03]
 [  4.27183692e-07   1.00000000e+00   6.39588343e-15   4.92283818e-03
    9.98388388e-01   9.13621682e-06   3.03819117e-04   9.99654277e-01]
 [  2.23610173e-06   1.00000000e+00   6.04142242e-13   3.85039426e-03
    9.95633529e-01   3.14114925e-03   9.96633698e-01   1.21306781e-03]
 [  1.56017417e-06   1.00000000e+00   5.31472829e-15   1.16403087e-04
    9.97228332e-01   5.93381265e-05   9.95723539e-01   9.96061779e-01]
 [  1.31680847e-08   1.00000000e+00   1.83221656e-16   5.48509255e-04
    9.93264489e-01   9.94158919e-01   4.15961966e-03   7.43289443e-03]
 [  2.90651250e-08   1.00000000e+00   1.39463343e-15   2.77311427e-04
    9.99518554e-01   9.98141514e-01   1.29341986e-03   9.98748935e-01]
 [  1.97756594e-07   1.00000000e+00   4.82099492e-13   8.14216649e-06
    9.96970005e-01   9.98015878e-01   9.98129197e-01   1.99256675e-04]
 [  2.64243141e-07   1.00000000e+00   4.89244846e-15   8.41228244e-06
    9.94985981e-01   9.96725430e-01   9.96542114e-01   9.99751260e-01]
 [  1.06651742e-07   1.00000000e+00   2.31496843e-14   9.95404266e-01
    5.05138372e-03   4.49794738e-03   3.15350200e-03   2.58935376e-04]
 [  4.05605440e-07   1.00000000e+00   3.90745627e-15   9.94823599e-01
    7.22826343e-07   3.98260012e-03   4.99544657e-03   9.97611649e-01]
 [  6.60551855e-07   1.00000000e+00   2.82227215e-14   9.99686049e-01
    1.12944313e-03   3.68960676e-07   9.95266297e-01   4.63449085e-04]
 [  7.06701283e-07   1.00000000e+00   1.50890670e-15   9.92939052e-01
    2.45702075e-04   8.45879829e-03   9.98772243e-01   9.99554448e-01]
 [  2.46129729e-07   1.00000000e+00   5.74203112e-14   9.99122821e-01
    1.33538017e-03   9.98809431e-01   3.28978516e-06   1.99714015e-03]
 [  6.81786264e-07   1.00000000e+00   5.45049489e-15   9.93106681e-01
    3.53196783e-03   9.93766543e-01   9.17242942e-03   9.99965117e-01]
 [  1.50958275e-05   1.00000000e+00   6.07373131e-11   9.99999765e-01
    1.05396462e-06   9.93912884e-01   9.97383845e-01   6.22671317e-03]
 [  4.56309547e-06   1.00000000e+00   3.96929460e-12   9.99974383e-01
    6.06422449e-03   9.99346705e-01   9.90952071e-01   9.98183206e-01]
 [  1.30193702e-06   1.00000000e+00   1.26621167e-12   9.95858173e-01
    9.95419146e-01   2.18059835e-05   6.33876078e-03   2.78442055e-03]
 [  3.19303183e-06   1.00000000e+00   5.97459013e-14   9.99963423e-01
    9.96891992e-01   3.33298259e-03   2.04117427e-03   9.94630961e-01]
 [  2.23771033e-06   1.00000000e+00   5.72568044e-12   9.95316591e-01
    9.97027879e-01   2.31570018e-03   9.96578842e-01   4.05443930e-04]]
    Interpretion:

    As for the result of 26* 8 array, for each of 8 bits precisely represent what is 
    in the ASC 2 table for capital letter from A to Z
    '''

    wh =result[1]
    wo =result[2]
    error =result[3]
    plt.xlabel("Iteration")
    plt.ylabel("Absolute difference")
    plt.plot(error)
    plt.show()
    #######################################
    #          test data part             #
    #######################################
    '''
    the original data works pretty well

[[  2.44762989e-07   9.99967541e-01   1.10010877e-04   1.12399340e-02
    9.96601858e-01   4.36285831e-04   9.88386026e-01   9.95834753e-01]]
    '''
    noise_test =np.array([[1,1,1,0,0,0,0,0,0,1,1,1,0,
            1,1,1,0,0,0,0,0,1,1,1,0,0,
            1,1,1,0,0,0,0,1,1,1,0,0,0,
            1,1,1,0,0,0,1,1,1,0,0,0,0,
            1,1,1,0,1,1,1,1,0,0,0,0,0,
            1,1,1,1,1,1,1,0,0,0,0,0,0,
            1,1,1,1,1,1,1,0,0,0,0,0,0,
            1,1,1,0,1,1,1,1,0,0,0,0,0,
            1,1,1,0,0,0,1,1,1,0,0,0,0,
            1,1,1,0,0,0,0,1,1,1,0,0,0,
            1,1,1,0,0,0,0,0,1,1,1,0,0,
            1,1,1,0,0,0,0,0,1,1,1,0,0]])
    '''
    by randomly creating noise (choose one random column to reverse value to be the noise)
    [[  1.23667432e-16   9.99999998e-01   1.17390447e-15   4.01238834e-03
    9.09736675e-01   5.26003988e-01   6.60764297e-01   9.99976427e-01]]

    the results still works pretty well
    '''

    noise_test0 =np.array([[1,1,1,0,0,0,0,0,0,1,1,1,1,
            1,1,1,0,0,0,0,0,1,1,1,0,1,
            1,1,1,0,0,0,0,1,1,1,0,0,1,
            1,1,1,0,0,0,1,1,1,0,0,0,1,
            1,1,1,0,1,1,1,1,0,0,0,0,1,
            1,1,1,1,1,1,1,0,0,0,0,0,1,
            1,1,1,1,1,1,1,0,0,0,0,0,1,
            1,1,1,0,1,1,1,1,0,0,0,0,1,
            1,1,1,0,0,0,1,1,1,0,0,0,1,
            1,1,1,0,0,0,0,1,1,1,0,0,1,
            1,1,1,0,0,0,0,0,1,1,1,0,1,
            1,1,1,0,0,0,0,0,1,1,1,0,1]])
    '''
    do the same but also add on another row by random
    [[  1.27339606e-11   1.00000000e+00   2.79903416e-10   9.94030771e-01
    1.76461163e-02   2.84419300e-01   1.02897300e-02   9.99813040e-01]]
    we clearly that the result is horribly off, it has most misclassfied value
    the letter can not be recognised by machine even if it is still visible to human.
    '''
    noise_test1 =np.array([[1,1,1,0,0,0,0,0,0,1,1,1,1,
            1,1,1,0,0,0,0,0,1,1,1,0,1,
            1,1,1,0,0,0,0,1,1,1,0,0,1,
            1,1,1,0,0,0,1,1,1,0,0,0,1,
            1,1,1,0,1,1,1,1,0,0,0,0,1,
            1,1,1,1,1,1,1,0,0,0,0,0,1,
            1,1,1,1,1,1,1,0,0,0,0,0,1,
            1,1,1,0,1,1,1,1,0,0,0,0,1,
            0,0,0,1,1,1,0,0,0,1,1,1,0,
            1,1,1,0,0,0,0,1,1,1,0,0,1,
            1,1,1,0,0,0,0,0,1,1,1,0,1,
            1,1,1,0,0,0,0,0,1,1,1,0,1]])

    h =sigmoid(np.dot(wh, noise_test1.T))
    y =sigmoid(np.dot(wo, h))
    print(y.T)

    X0 =np.array([[0,0,1,0,0,1,0,0,0,1,1,1,1,
        1,1,1,0,0,1,0,0,0,1,1,1,0, 
        0,0,1,0,0,1,0,1,1,1,0,0,0,
        1,1,0,0,0,0,1,0,1,0,1,0,0, 
        1,1,1,0,0,1,1,1,0,1,1,0,1, 
        0,1,0,0,1,0,0,0,0,0,0,1,0,
        1,1,1,1,1,1,1,0,1,0,0,0,0,
        1,0,1,0,1,1,1,1,1,1,1,0,1,
        1,1,1,1,1,0,1,1,1,0,1,0,0, 
        1,1,1,0,0,0,0,1,0,1,0,0,0,
        1,0,1,1,0,1,1,0,0,1,0,1,0,
        1,1,0,0,0,0,0,1,0,1,0,1,1]])

    # X0 =X0.astype(np.int)
    X0 =X0.T

    h =sigmoid(np.dot(wh, X0))
    y =sigmoid(np.dot(wo, h))
    print(y.T)
    '''
    the output:
    the letter is: Y given info that,

[[  9.93473993e-15   1.00000000e+00   4.00027289e-13   9.62171011e-01
    8.64559716e-01   2.64450026e-03   3.77442885e-03   1.20424192e-01]]
    '''






        
        





