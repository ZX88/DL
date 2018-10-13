from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np

import time

NAME = "lab4_2CNN-{}".format(int(time.time()))


print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)
#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
#x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)

num_classes = 10
batch_size = 200
epochs = 40

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


#Our first simple CNN
def n_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
        
    return model    


model = n_network()
tensorboard = TensorBoard(log_dir="logs/{}".format(NAMe))
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[tensorboard])

score = model.evaluate(x_test, y_test)

model.save('cnn_models/{}.model'.format(NAME))
print("CNN accuracy : %.2f%%" %(score[1]*100))