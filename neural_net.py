import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class NeuralNetwork:

    def __init__(self, layer, activation ='sig'):
        self.activation = sigmoid
        self.layer =layer
        print(type(sigmoid))

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

    def back_propogation(self, X, Y, learning_rate=0.9):
        params =self.init_param()
        wh =params['wh']
        wo =params['wo']

        dwh =np.zeros(wh.shape)
        dwo =np.zeros(wo.shape)
        N_trial =10000
        # forward and backward path
        for trial in range(N_trial):
            h =self.activation(np.dot(wh, X))
            y =self.activation(np.dot(wo, h))

            error =Y -y
            delta =y*(1-y)*error
            update_hidden =h*(1-h)*np.dot(wo.T,delta)
        
            # update weights with mometum
            dwo =learning_rate*dwo +np.dot(delta, h.T)
            wo =wo +0.1*dwo
            dwh =learning_rate*dwh +np.dot(update_hidden, X.T)
            wh =wh +0.1*dwh
        return wh, wo 

if __name__ == '__main__':
    nn = NeuralNetwork([3,50,1])
    # training data
    X = np.array([[0,0,1,1],
                  [0,1,0,1],
                  [1,1,1,1]])
    # testing data
    X_test = np.array([[0,0,1,1],
                  [0,1,0,1],
                  [0,0,0,0]])
    Y = np.array([[1, 0, 0, 1]])
    result =nn.back_propogation(X, Y, 0.9)


    #testing
    h =sigmoid(np.dot(result[0], X_test))
    y =sigmoid(np.dot(result[1], h))
    print(y)



