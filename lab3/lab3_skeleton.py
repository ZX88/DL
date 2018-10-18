from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
import numpy as np

#print('tensorflow:', tf.__version__)
#print('keras:', keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8


#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


num_classes = 1


#Let start our work: creating a neural network
#First, we just use a single neuron. 

m = x_train.shape[1]

def one_neuron():
    model = Sequential()
    model.add(Dense(num_classes, input_dim = num_pixels ,\
     kernel_initializer = 'normal', activation = 'sigmoid'))
    
    #Compiling the neuron 
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc', 'mae'])
    return model    

def mult_neuron():
    model = Sequential()
    
    model.add(Dense(16, input_dim = num_pixels, kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dense(num_classes, kernel_initializer = 'normal', activation = 'sigmoid'))
    
    #Compiling the neuron 
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model    



#Basing the model

model = mult_neuron()
	
# model1 = mult_neuron()
#model1.summary()

X = x_train
Y = y_train
model.fit(X,Y, validation_data=(x_test, y_test), epochs = 100, batch_size = 40)
score = model.evaluate(x_test, y_test)
print("NN accuracy : %.2f%%" %(score[1]*100))




