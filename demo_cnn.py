import tensorflow as tf
import numpy as np

x_in = np.array([[
    [[1], [2], [3], [4], [5]],
    [[5], [4], [3], [2], [1]],
    [[1], [0], [1], [0], [1]],
    [[2], [3], [4], [1], [1]],
    [[3], [1], [4], [1], [5]], ]])

kernel_in = np.array([
    [ [[0]], [[1]], [[1]] ],
    [ [[1]], [[1]], [[1]] ],
    [ [[1]], [[1]], [[0]] ], ])

x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
# print(x)
# print(kernel)

y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
print(y)

y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
print(y)

y = tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='VALID')
print(y)

y = tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')
print(y)