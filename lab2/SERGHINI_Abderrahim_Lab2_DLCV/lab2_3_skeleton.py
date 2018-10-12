import numpy as np

#In this first part, we just prepare our data (mnist)
#for training and testing
import time
import keras
from keras import backend as K
from keras.datasets import mnist
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


# one-hot encode labels
digits = 10

def one_hot_encode(y, digits):
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('uint32')]  #shape (1, 70000, 10)
    Y_new = Y_new.T.reshape(digits, examples)
    return Y_new

y_train=one_hot_encode(y_train, digits)
y_test=one_hot_encode(y_test, digits)

m = X_train.shape[1]

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

# #Display one image and corresponding label
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network

#sigmoid function
def sigf(z):
    s = 1./ (1.+ np.exp(-z))
    return s

#the loss function
def computeLoss(y,y_hat):
    m = y.shape[1] * 10
    L = - 1/m * ( np.sum(np.multiply(y,np.log(y_hat))) +  np.sum(np.multiply(1-y,np.log(1-y_hat))) )
    return L

#Accuracy
def accuracy(y_true, y_pred):
    K = np.argmax(y_true, axis = -1)
    K2 = np.argmax(y_pred, axis = -1)
    return np.mean(np.equal(K, K2))

#Build and train the Network
X = X_train
Y = y_train

learning_rate = 1.

n_x = X.shape[0]
m = X.shape[1]

W1 = np.random.randn(1,n_x) * 0.01
W2 = np.random.randn(10,64) * 0.01
b1 = np.zeros((64,1))
b2 = np.zeros((10,1))

epoch = 500
slice = epoch/25

t = time.time()

for k in range(epoch):

    #Forward propagation
    Z1 = np.matmul(W1,X) + b1
    A1 = sigf(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = sigf(Z2)

    cost = computeLoss(Y,A2)
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
        print("Epoch", k, "cost :", cost)

elapsed = time.time() - t

print("Final cost : ", cost, "Elapsed : ", elapsed)