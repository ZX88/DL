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

