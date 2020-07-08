# tensorflow 2
import tensorflow as tf
import numpy as np

# 输入为两个维度，输出为一个维度
# 如下训练数据中，两个维度越大，则输出的label-Y为1，两个维度越小，则输出的label-Y为0
X = [[0.7, 0.9],
    [0.1, 0.4],
    [0.5, 0.8],
    [0.6, 0.9],
    [0.2, 0.4],
    [0.6, 0.8]]
Y = [[1], [0], [1], [1], [0], [1]]

date_size =len(X)
batch_size =3

# 三层网络结构，分别为2X3，3X1
random_normal = tf.random_normal_initializer(stddev=1)
w1 = tf.Variable(random_normal(shape=[2, 3]), name="w1")
w2 = tf.Variable(random_normal(shape=[3, 1]), name="w2")
b1 = tf.Variable(random_normal([3]), name="b1")
b2 = tf.Variable(random_normal([1]), name="b2")

def hidden_layer(input):
    return tf.matmul(input, w1) + b1

def output_layer(input):
    return tf.matmul(input, w2) + b2

class Model(object):
    def  __init__(self):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
    def  __call__(self, inputs):
        return self.output_layer(self.hidden_layer(inputs))

# 使用交叉熵作为损失函数，对training的结果进行评判
def cross_entropy(output, labels):
    return -tf.reduce_mean(labels * tf.math.log(tf.math.sigmoid(output)) 
            + (tf.constant(1.0) - labels) * tf.math.log(tf.constant(1.0)  - tf.math.sigmoid(output)))

def make_loss(model, inputs, labels):
    return cross_entropy(model(inputs), labels)

# Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train(model, x, y):
    optimizer.minimize(lambda:make_loss(model, x, y), var_list=[w1, b1, w2, b2])

# 初始化model
model = Model()

# 打印初始状态
print("begin: ")
print(w1)
print(w2)
print(b1)
print(b2)
print(model([[0.7, 0.9]]))
print(model([[0.6, 0.8]]))
print(model([[0.1, 0.4]]))
print(model([[0.2, 0.3]]))

# 训练阶段，训练1000次，loss基本可以减小到接近于0
for i in range(1000):
    for k in range(0, date_size, batch_size):
        mini_batch = X[k:k+batch_size]
        train_y = Y[k:k+batch_size]
        train(model, mini_batch, train_y)
    print("iter %d:" % (i))
    print(make_loss(model, mini_batch, train_y))

# 预测阶段，预测值会被分为两个区间，其中前四个预测值较大，后四个预测值较小
print("predict: ")
print(model([[0.7, 0.9]]))
print(model([[0.6, 0.8]]))
print(model([[0.6, 1.0]]))
print(model([[0.5, 0.9]]))
print(model([[0.1, 0.9]]))
print(model([[0.2, 0.8]]))
print(model([[0.1, 0.4]]))
print(model([[0.2, 0.3]]))

# 输出最终weight和bias
print(w1)
print(w2)
print(b1)
print(b2)