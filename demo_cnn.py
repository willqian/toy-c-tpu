import tensorflow as tf
import numpy as np

x_in = np.array([[
    [[1], [2], [3], [4], [5]],
    [[5], [4], [3], [2], [1]],
    [[1], [0], [1], [0], [1]],
    [[2], [3], [4], [1], [1]],
    [[3], [1], [4], [1], [5]], ]])

x2_in = np.array([[
    [[1, 3], [2, 2], [3, 1], [4, 0], [5, 4]],
    [[5, 2], [4, 1], [3, 3], [2, 1], [1, 0]],
    [[1, 1], [0, 2], [1, 0], [0, 1], [1, 2]],
    [[2, 3], [3, 2], [4, 4], [1, 3], [1, 1]],
    [[3, 1], [1, 1], [4, 0], [1, 0], [5, 1]], ]])

x7_in = np.array([[
    [[1], [2], [3], [4], [5], [6], [7]],
    [[5], [4], [3], [2], [1], [0], [1]],
    [[1], [0], [1], [0], [1], [0], [1]],
    [[2], [3], [4], [1], [1], [2], [3]],
    [[3], [1], [4], [1], [5], [1], [6]],
    [[7], [5], [2], [9], [1], [8], [5]],
    [[6], [3], [1], [2], [3], [7], [9]], ]])

kernel_in = np.array([
    [ [[0]], [[1]], [[1]] ],
    [ [[1]], [[1]], [[1]] ],
    [ [[1]], [[1]], [[0]] ], ])

kernel2_in = np.array([
    [ [[0, 1]], [[1, 0]], [[1, 0]] ],
    [ [[1, 0]], [[1, 1]], [[1, 0]] ],
    [ [[1, 0]], [[1, 0]], [[0, 1]] ], ])

kernel22_in = np.array([
    [ [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [0, 1]] ],
    [ [[1, 0], [0, 1]], [[1, 1], [1, 1]], [[1, 0], [0, 1]] ],
    [ [[1, 0], [0, 1]], [[1, 0], [1, 0]], [[0, 1], [1, 0]] ], ])

kernel5_in = np.array([
    [ [[0]], [[1]], [[1]], [[0]], [[1]] ],
    [ [[1]], [[1]], [[1]], [[1]], [[0]] ],
    [ [[1]], [[1]], [[0]], [[1]], [[0]] ],
    [ [[0]], [[0]], [[1]], [[0]], [[0]] ],
    [ [[1]], [[1]], [[1]], [[1]], [[1]] ], ])

x = tf.constant(x_in, dtype=tf.float32)
x2 = tf.constant(x2_in, dtype=tf.float32)
x7 = tf.constant(x7_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
kernel2 = tf.constant(kernel2_in, dtype=tf.float32)
kernel22 = tf.constant(kernel22_in, dtype=tf.float32)
kernel5 = tf.constant(kernel5_in, dtype=tf.float32)

y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
print(y)

y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
print(y)

y = tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='VALID')
print(y)

y = tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')
print(y)

y = tf.nn.conv2d(x, kernel2, strides=[1, 2, 2, 1], padding='SAME')
print(y)

y = tf.nn.conv2d(x2, kernel22, strides=[1, 2, 2, 1], padding='SAME')
print(y)

y = tf.nn.conv2d(x7, kernel5, strides=[1, 2, 2, 1], padding='SAME')
print(y)