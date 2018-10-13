from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)


#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

seed = 7
np.random.seed(seed)

num_classes = y_train.shape[1]


#Our first simple CNN
def n_network():
    model = Sequential()
    
    #Conv layer
    model.add(Conv2D(64, (5,5), input_shape=(3,32,32), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128 ,kernel_initializer = 'random_uniform', activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    #Compiling the neuron 
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc', 'mae'])
    
    return model    


model = n_network()

X = x_train
Y = y_train
model.fit(X,Y, validation_data=(x_test, y_test), epochs = 100, batch_size = 1000)
score = model.evaluate(x_test, y_test)
print("CNN accuracy : %.2f%%" %(score[1]*100))