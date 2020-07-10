# toy-c-tpu

## TPU demo

该demo集成了TPU的最基本的指令功能，可以进行基本的矩阵运算和ReLU激活函数处理
```
gcc -o demo demo.c vm.c
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
1. 输入图像根据黑白程度量化到区间`[0, 5]`，使用其他的区间也是可以的
2. weight和bias的量化则是根据网络层数，以及该层绝对平均值、绝对最大值作为参考人工调节，可以参考`quantize_data`最后一个参数`region`，目前的做法是先根据绝对最大值进行数据量化，然后根据绝对平均值设置`region`进一步量化数据区间
3. 矩阵运算溢出处理，现在处理的方式是如果超过了`(-128, +128)`的范围，则取对应的`-127`或`+127`替换，打开`vm.c`的`VM_WARN`可以查看溢出调试信息
4. 与TPU float不同，INT8并不适合用softmax，这里直接选择max函数，也就是最大值设为概率1，其他的设为概率0
```
gcc -o mnist_nn mnist_nn.c vm.c mnist_nn_variables.c mnist_data.c
```

参考输出结果
```
max w1 0.940379, average 0.088694
max b1 0.208807, average 0.062003
max w2 1.095445, average 0.211713
max b2 0.115191, average 0.057727
predict 2, label 2
predict 6, label 6
predict 3, label 3
predict 4, label 4
predict 3, label 3
```

## 训练mnist全连接神经网络

使用tensorflow训练该网络，参考`mnist_nn.py`

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
