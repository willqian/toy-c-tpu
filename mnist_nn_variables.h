#ifndef MNIST_NN_VARIABLES_H
#define MNIST_NN_VARIABLES_H

// tf.Variable 'my_model/dense/kernel:0' shape=(784, 128) dtype=float32
extern float dw1[784][128];
// tf.Variable 'my_model/dense/kernel:0' shape=(128,) dtype=float32
extern float db1[128];
// tf.Variable 'my_model/dense_1/kernel:0' shape=(128, 10) dtype=float32
extern float dw2[128][10];
// tf.Variable 'my_model/dense_1/bias:0' shape=(10,) dtype=float32
extern float db2[10];

#endif /* MNIST_NN_H */