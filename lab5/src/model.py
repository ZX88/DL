from __future__ import print_function
import keras
print('Keras version : ', keras.__version__)
from keras.models import Sequential
from keras.layers import Dense, Activation, ZeroPadding3D, Dropout, Conv3D, MaxPooling3D, Flatten
from keras.models import Model

#from keras import backend as K
#K.set_image_dim_ordering('tf')

#############################################
############## Make the model ###############
#############################################


def make_one_branch_model(temporal_dim, width, height, channels, nb_class):
    model = Sequential()

    #model.add(ZeroPadding3D(2, input_shape=( channels, temporal_dim, height, width)))
    model.add(Conv3D(30, (3,3,3),input_shape=(temporal_dim, height, width, channels),\
            padding='same' ))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    #model.add(Dropout(0.2))

    model.add(Conv3D(60, (3,3,3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    #model.add(Dropout(0.25))

    model.add(Conv3D(80, (3,3,3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    #model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))


    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['acc']);
    return model


def make_model(temporal_dim, width, height, nb_class):
    #TODO

    #input1 = Input()
    #model = Model(inputs=, outputs=)
    #Build the siamese model and compile it.
    #Use the following optimizer
    #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    return model
