import numpy as np

#In this first part, we just prepare our data (mnist)
#for training and testing

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


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

#the loss function
def computeLoss(y,y_hat):
    m = y.shape[1]
    L = - 1/m * ( np.sum(np.multiply(y,np.log(y_hat))) +  np.sum(np.multiply(1-y,np.log(1-y_hat))) )
    return L

#define the backward propagation, build and train our network
X = X_train
Y = y_train

learning_rate = 1.8

n_x = X.shape[0]
m = X.shape[1]

W = np.random.randn(1,n_x) * 0.01
b = np.zeros((1,1))


for k in range(500):
    Z = np.matmul(W,X) + b
    Y_hat = sigf(Z)

    cost = computeLoss(Y,Y_hat)

    wL =  1./m * np.matmul((Y_hat-Y), X.T)
    db =  1./m * np.sum((Y_hat - Y), axis = 1, keepdims = True)
    W = W - learning_rate * wL
    b = b - learning_rate * db

    if (k % 20 == 0 ):
        print("Epoch", k, "cost :", cost)

print("Final cost : ", cost)
