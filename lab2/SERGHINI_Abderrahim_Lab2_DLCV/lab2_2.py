import numpy as np
#In this first part, we just prepare our data (mnist)
#for training and testing

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import backend as K 
import time


(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


# #Display one image and corresponding label
#import matplotlib
#import matplotlib.pyplot as plt
#i = 3
#print('y[{}]={}'.format(i, y_train[:,i]))
#plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
#plt.axis("off")
#plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron.

#sigmoid function
def sigf(z):
    s = 1./ (1.+ np.exp(-z))
    return s

#The loss function
def computeLoss(y,y_hat):
    m = y.shape[1]
    L = - 1./m * ( np.sum(np.multiply(y,np.log(y_hat))) +  np.sum(np.multiply(1-y,np.log(1-y_hat))) )
    return L

# def conf_matrix(predicted, expected, n_classes):

#     m = [[0] * n_classes for i in range(n_classes)]
#     for pred, exp in zip(predicted, expected):
#         m[pred][exp] += 1
#     return m

def accuracy(y_true, y_pred): # binary accuracy
    return np.mean(np.equal(y_true, np.round(y_pred)))

#Build and train the Network
X = X_train
Y = y_train

learning_rate = 1.

n_x = X.shape[0]
m = X.shape[1]

W1 = np.random.randn(1,n_x) * 0.01
W2 = np.random.randn(1,64) * 0.1
b1 = np.zeros((64,1))
b2 = np.zeros((1,1))

epoch = 500
slice = epoch/20

t = time.time()

for k in range(epoch):

    #Forward propagation
    Z1 = np.matmul(W1,X) + b1
    A1 = sigf(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = sigf(Z2)
    
    #print('A2', type(A2))
    cost = computeLoss(Y,A2)
    #cm = conf_matrix(Y,A2,1)
    acc = accuracy(Y,A2)

    #Backward propagation
    dW2 =  1./m * np.matmul((A2-Y), A1.T)
    db2 =  1./m * np.sum((A2 - Y), axis = 1, keepdims = True)

    P = np.matmul(W2.T,(A2-Y))
    P2 = np.multiply(A1,1-A1)
    prod = np.multiply(P,P2)
    dW1 =  1./m * np.matmul(prod, X.T)
    db1 =  1./m * np.sum(prod, axis = 1, keepdims= True)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    if (k % slice == 0 ):
        print("Epoch", k, "cost :", cost , "Accuracy :", acc)

elapsed = time.time() - t

print("Final cost : ", cost, "Accuracy : ", acc, "Elapsed : ", elapsed)