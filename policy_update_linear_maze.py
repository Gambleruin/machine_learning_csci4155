import random
import matplotlib.pyplot as plt 
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

# maze representation as net input
def initMaze():
    maze =np.zeros((1,5,3))
    #place mouse
    maze[0, 2]=np.array([0,0,1])
    return maze

# update the mouse's movement
def makeMove(state, action):
    maze =np.zeros((1,5,3))
    new_state =tau(state, action)
    print('\n\n\n\n\n\nthis is new state',new_state,'\n\n\n\n\n\n\n')
    maze[0, int(new_state)] =np.array([0,0,1])
    return maze


# transition function
def tau(s,a):
    if s==0 or s==4:  return(s)
    else:      return(s+a)
# reward function
def rho(s,a):
    return(s==1 and a==-1)+2*(s==3 and a==1)   
def calc_policy(Q):
    policy=np.zeros(5)
    for s in range(0,5):
        action_idx=np.argmax(Q[s,:])
        policy[s]=2*action_idx-1
        policy[0]=policy[4]=0
    return policy.astype(int)

def idx(a):
    return(int((a+1)/2))

#initialize the network
model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(15,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

# train and predict
epochs =1000
gamma =0.5
epsilon =0.1

alpha=0.2
policy=np.zeros(5)
Q=np.zeros([5,2])
for trial in range(epochs):
    maze_ =initMaze()
    # policy=calc_policy(Q)
    cur_state=2
    for t in range(0,5):
        
        print('\n\n\n\n\n\n\n\n\nwhy is it not an integer??\n\n\n\n\n\n\n\n\n', cur_state)
        action=policy[cur_state]
        print('this is previous action', action)
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(maze_.reshape(1,15), batch_size=1)
        
    
        if np.random.rand()<0.1: 
            action=-action #epsilon greedy
       
        print(qval)
        print('this is cur_state', cur_state, 'and this is action', action)
        new_state =tau(cur_state, action)
        # print(new_state,'\n\n\n\n\n\n')
        maze_new =makeMove(cur_state, action)
        reward =rho(cur_state, action)
        newQ = model.predict(maze_new.reshape(1,15), batch_size=1)
        maxQ =np.max(newQ)
        update =alpha*(reward +(gamma*maxQ))
        # print(update,'\n\n\n\n\n\n\n\n\n')
        print('\n\n\n\n\n\n\nthis is idx(action)',action)
        Q[cur_state][idx(action)] =update
        new_qval =Q.reshape(1,10)
        print('this is Q',Q)
        # update policy
        policy=calc_policy(qval.reshape(5,2))
        print('This is policy',policy)
        # update the network
        model.fit(maze_.reshape(1, 15), new_qval, batch_size =1, nb_epoch =1, verbose =1)
        # update the state
        cur_state =int(new_state)

print('Q values: \n',np.transpose(Q))
print('policy: \n',np.transpose(policy))


