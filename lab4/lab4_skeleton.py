from __future__ import print_function
import numpy as np


import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
K.set_image_dim_ordering('th')


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
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

seed = 7
np.random.seed(seed)

num_classes = y_train.shape[1]


def n_network():
    model = Sequential()
    
    #Conv layer
    model.add(Conv2D(32, (5,5), input_shape = (1,28,28), activation =params['relu']))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128 ,kernel_initializer = 'random_uniform', activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    #Compiling the neuron 
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc', 'mae'])
    
    return model    

#Ex
model = n_network()

#model.summary()

X = x_train
Y = y_train
model.fit(X,Y, validation_data=(x_test, y_test), epochs = 100, batch_size = 1000)
score = model.evaluate(x_test, y_test)
print("CNN accuracy : %.2f%%" %(score[1]*100))