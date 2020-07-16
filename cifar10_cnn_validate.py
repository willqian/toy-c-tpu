import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = tf.saved_model.load("./cifar10_cnn_keras_saved_graph")

## conv0
x = tf.constant([train_images[1],], dtype=tf.float32)
k0 = model.trainable_variables[0]
b0 = model.trainable_variables[1]
y0 = tf.nn.conv2d(x, k0, strides=[1, 1, 1, 1], padding='VALID')
y0 = tf.math.add(y0, b0)
y0 = tf.nn.relu(y0)
#print(y0)
yp0 = tf.nn.max_pool(y0, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
#print(yp0)

## conv1
k1 = model.trainable_variables[2]
b1 = model.trainable_variables[3]
y1 = tf.nn.conv2d(yp0, k1, strides=[1, 1, 1, 1], padding='VALID')
y1 = tf.math.add(y1, b1)
y1 = tf.nn.relu(y1)
#print(y1)
yp1 = tf.nn.max_pool(y1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
#print(yp1)

## conv2
k2 = model.trainable_variables[4]
b2 = model.trainable_variables[5]
y2 = tf.nn.conv2d(yp1, k2, strides=[1, 1, 1, 1], padding='VALID')
y2 = tf.math.add(y2, b2)
y2 = tf.nn.relu(y2)
#print(y2)

## flatten
yf = tf.keras.backend.flatten(y2)
#print(yf)

## d0
dw0 = model.trainable_variables[6]
db0 = model.trainable_variables[7]
yd0 = tf.matmul(tf.constant(yf, shape=[1, 1024]), dw0) + db0
yd0 = tf.nn.relu(yd0)
#print(yd0)

## d1
dw1 = model.trainable_variables[8]
db1 = model.trainable_variables[9]
yd1 = tf.matmul(tf.constant(yd0, shape=[1, 64]), dw1) + db1
print(yd1)
