#include "mnist_data.h"
#include "mnist_nn_variables.h"
#include "vm32.h"

#include <stdio.h>

#include <math.h>

static float test_input_plus[1] = {1};

static int result(float *y)
{
    float max = 0;
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (max < y[i]) {
            max = y[i];
            max_index = i;
        }
    }
    return max_index;
}

int calculate(float *test_x)
{
    float test_y[10] = {0};
    //  784X128 left part
    vm32_read_host_memory(0, (float *)test_x, 28 * 28);
    vm32_read_host_memory(28 * 28, test_input_plus, 1);
    for (int i = 0; i < 28 * 28; i++) {
        vm32_read_weights((float *)dw1 + i * 128, 64);
    }
    vm32_read_weights((float *)db1, 64);
    vm32_maxtrix_multiply(0, 0, 1, 28 * 28 + 1, 28 * 28 + 1, 64);

    // 784X128 right part
    for (int i = 0; i < 28 * 28; i++) {
        vm32_read_weights((float *)dw1 + 64 + i * 128, 64);
    }
    vm32_read_weights((float *)db1 + 64, 64);
    vm32_maxtrix_multiply(0, 64, 1, 28 * 28 + 1, 28 * 28 + 1, 64);

    vm32_activate(ACT32_TYPE_RELU, 0, 0, 128);

    // 128X10
    vm32_read_host_memory(128, test_input_plus, 1);
    vm32_read_weights((float *)dw2, 128 * 10);
    vm32_read_weights((float *)db2, 10);
    vm32_maxtrix_multiply(0, 0, 1, 128 + 1, 128 + 1, 10);
    vm32_activate(ACT32_TYPE_SOFTMAX, 0, 0, 10);

    vm32_write_host_memory(test_y, 0, 10);

    return result(test_y);
}

int main(int argc, char *argv[])
{
    vm32_init();
    printf("predict %d, label %d\n", calculate((float *)test_data1), test_y1);
    printf("predict %d, label %d\n", calculate((float *)test_data2), test_y2);
    printf("predict %d, label %d\n", calculate((float *)test_data3), test_y3);
    printf("predict %d, label %d\n", calculate((float *)test_data4), test_y4);
    printf("predict %d, label %d\n", calculate((float *)test_data5), test_y5);
}