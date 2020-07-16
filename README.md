# toy-c-tpu

## TPU demo

该demo集成了TPU的最基本的指令功能，可以进行基本的矩阵运算和ReLU激活函数处理

TPU矩阵运算的优化，核心在于，乘法器矩阵中左(input)上(weight)到右下的对角线加载和计算，能够完整利用所有的乘法器计算资源
1. 可以降低TPU load input数据的次数
2. 每一个pipeline clock，内部有256个tick，每个tick都可以输出一次结果

有一点需要注意，累加操作应该是累加到累加器RAM中，而不是乘法器之间做累加，乘法器是8位的，累加会有溢出

```
gcc -o demo demo.c vm.c
```

## TPU 32位 float cifar CNN 图像识别

已支持max pooling
```
gcc -o cifar10 cifar10_cnn_32.c vm32.c cifar10_cnn_variables.c cifar10_cnn_data.c
```

推理结果
```
predict: 9, truck, test_y: 9, truck
detail:
1.353900 -0.863579 2.427378 -4.060456 -3.588637 -1.933179 -10.055259 2.915245 -4.191559 4.489056

predict: 8, ship, test_y: 8, ship
detail:
2.967486 -1.649458 0.286217 -0.655495 0.704763 -2.225053 -1.592144 -2.496777 5.010174 -3.057529

predict: 6, frog, test_y: 6, frog
detail:
-10.382006 -7.122347 -0.053340 1.615066 -1.460678 1.234555 6.963420 -6.225483 -5.144471 -10.842333

predict: 6, frog, test_y: 2, bird
detail:
-1.958389 -3.877736 0.378107 -0.071973 -0.029878 -1.665769 0.732384 -1.969363 -5.271281 -2.894828

predict: 5, dog, test_y: 5, dog
detail:
-6.067231 -9.018535 -0.971456 1.954368 -0.616314 4.614746 -0.701623 -0.363836 -8.493745 -5.801422
```

tensorflow训练
```
python cifar10_cnn.py
```

训练模型
```
cifar10_cnn_keras_saved_graph
```

tensorflow帮助验证TPU正确性
```
python cifar10_cnn_validate.py
```

## TPU CNN

已支持卷积操作，包括channel，padding，stride，kernel size的处理

TPU卷积运算的优化，核心在于，针对卷积核中的每一个元素都先对整个输入矩阵进行一次卷积遍历，然后把每个卷积核元素的结果累加起来
1. 可以降低TPU load input数据的次数
2. 可以最大化利用TPU的乘法器资源

当然，TPU论文中并没有描述具体的卷积运算实现方式，但根据算法分析以及2015年的一篇论文：Optimizing FPGA-based accelerator design for deep convolutional neural networks，可以基本确认这种实现方式是高性能的

### TPU INT8 CNN demo

```
gcc -o demo_cnn demo_cnn.c vm.c
```

### TPU 32位 float CNN demo

```
gcc -o demo_cnn_32 demo_cnn_32.c vm32.c
```

### 参考输出
```
CNN normal:
18.000000 17.000000 16.000000
14.000000 13.000000 10.000000
14.000000 14.000000 12.000000

CNN with padding 1:
8.000000 15.000000 16.000000 17.000000 12.000000
13.000000 18.000000 17.000000 16.000000 9.000000
12.000000 14.000000 13.000000 10.000000 4.000000
9.000000 14.000000 14.000000 12.000000 9.000000
9.000000 15.000000 11.000000 12.000000 7.000000

CNN with stride 2:
18.000000 16.000000
14.000000 12.000000

CNN with padding 1 and stride 2:
8.000000 16.000000 12.000000
12.000000 13.000000 4.000000
9.000000 11.000000 7.000000

CNN with padding 1 and stride 2 and kernel 2:
8.000000 16.000000 12.000000
12.000000 13.000000 4.000000
9.000000 11.000000 7.000000

5.000000 5.000000 5.000000
4.000000 6.000000 3.000000
3.000000 7.000000 6.000000

CNN with padding 1 and stride 2 and kernel 2 and channel 2:
14.000000 21.000000 16.000000
18.000000 21.000000 8.000000
10.000000 13.000000 11.000000

10.000000 9.000000 10.000000
10.000000 15.000000 9.000000
10.000000 15.000000 8.000000

CNN with input 7x7 kernel 5x5, padding 2, and stride 2:
9.000000 13.000000 17.000000 14.000000
23.000000 43.000000 41.000000 31.000000
25.000000 34.000000 39.000000 37.000000
22.000000 44.000000 42.000000 31.000000
```

### 使用tensorflow卷积用来做验证
```
python demo_cnn.py
```

## TPU 32位 float 推理mnist全连接神经网络

基于tensorflow训练的结果进行推理
```
gcc -o mnist_nn_32 mnist_nn_32.c vm32.c mnist_nn_variables.c mnist_data.c
```

参考输出结果
```
predict 2, label 2
predict 6, label 6
predict 3, label 3
predict 4, label 4
predict 3, label 3
```

## TPU INT8 推理mnist全连接神经网络

基于tensorflow训练的结果进行推理，关键点有三，在于量化、矩阵运算数据溢出处理、max激励函数的选择
1. 输入图像根据黑白程度量化到区间`[0, 10]`，使用其他的区间也是可以的
2. weight和bias的量化，可以参考`quantize_data`最后一个参数`region`，目前的做法是观察绝对最大值的范围，然后根据绝对最大值设置`region`进行量化，在这里使用的`region`是16，也就是量化到`[0, 16]`，如果`region`太大容易溢出，太小则容易损失过多精度
3. 矩阵运算溢出处理，现在处理的方式是如果超过了`(-128, +128)`的范围，则取对应的`-127`或`+127`替换，打开`vm.c`的`VM_WARN`可以查看溢出调试信息
4. 与TPU float不同，INT8并不适合用softmax，这里直接选择max函数，也就是最大值设为概率1，其他的设为概率0
```
gcc -o mnist_nn mnist_nn.c vm.c mnist_nn_variables.c mnist_data.c
```

参考输出结果
```
max abs w1 0.940379
max abs b1 0.208807
max abs w2 1.095445
max abs b2 0.115191
predict 2, label 2
predict 6, label 6
predict 3, label 3
predict 4, label 4
predict 3, label 3
```

## 训练mnist全连接神经网络

使用tensorflow训练该网络，参考`mnist_nn.py`
```
python mnist_nn.py
```

## TPU 32位 float 推理3层神经网络

32位TPU虚拟机只是用来做对比验证，最终芯片实现的还是INT8
```
gcc -o nn3_32 nn3_32.c vm32.c
```

参考输出结果
```
x1 [0.7, 0.9]
x2 [0.6, 0.8]
x3 [0.9, 0.7]
x4 [0.1, 0.4]
x5 [0.4, 0.1]
x6 [0.2, 0.2]

test_y1 6.246612
test_y2 4.335637
test_y3 4.117450
test_y4 -3.731470
test_y5 -6.925214
test_y6 -6.283831
```

## 推理3层神经网络

该demo基于tensorflow训练3层神经网络的结果，进行量化后，再使用TPU进行推理
```
gcc -o nn3 nn3.c vm.c
```

参考结果如下，符合训练3层神经网络时的结果，输入x的两个维度越大，输出结果越大，否则越小
```
x1 [5, 7]
x2 [4, 6]
x3 [7, 5]
x4 [0, 3]
x5 [3, 0]
x6 [1, 1]
test_q_y1 121
test_q_y2 101
test_q_y3 89
test_q_y4 39
test_q_y5 -9
test_q_y6 5

```

## 训练3层神经网络

使用tensorflow训练该网络，得到的weight和bias可以用来给TPU进行推理

```
python nn3.py
```

参考输出weight和bias：
```
w1:
[[ 0.12360667, -0.70676994, -1.3703936 ],
 [-2.8751068 , -1.1498548 , -1.6827904 ]]
w2:
[[-2.9926474],
 [-2.2082868],
 [-2.2191722]]
b1:
[ 1.5301738 ,  1.7288847 , -0.36803755]
b2:
[-2.5253797]
```
