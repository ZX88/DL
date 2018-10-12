import numpy as np

import tensorflow as tf
import keras 
import cv2

#For visualizing/ i guess
import matplotlib.pyplot as plt


model1 = tf.keras.models.load_model('cnn_models/cnn_32x3.model')

#Get picture
test1 = cv2.imread('Tests/1.png')
test = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
#test = cv2.resize(test, (28,28))


plt.imshow(test,cmap=plt.cm.binary)
plt.show()

test = np.reshape(test, (1,1,28,28))

#make predictions
predictions = model1.predict(test)
print(np.argmax(predictions[0]))