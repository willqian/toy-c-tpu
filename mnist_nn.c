#include "mnist_data.h"
#include "mnist_nn_variables.h"
#include "vm.h"

#include <stdio.h>
#include <math.h>

static int8_t test_q_data1[28][28];
static int8_t test_q_data2[28][28];
static int8_t test_q_data3[28][28];
static int8_t test_q_data4[28][28];
static int8_t test_q_data5[28][28];

static int8_t q_dw1[784][128];
static int8_t q_db1[128];
static int8_t q_dw2[128][10];
static int8_t q_db2[10];

static int8_t test_input_plus[1] = {1};

static int result(int8_t *y)
{
    int8_t max = 0;
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (max < y[i]) {
            max = y[i];
            max_index = i;
        }
    }
    return max_index;
}

int calculate(int8_t *test_x)
{
    int8_t test_y[10] = {0};
    //  784X128 left part
    vm_read_host_memory(0, (int8_t *)test_x, 28 * 28);
    vm_read_host_memory(28 * 28, test_input_plus, 1);
    for (int i = 0; i < 28 * 28; i++) {
        vm_read_weights((int8_t *)q_dw1 + i * 128, 64);
    }
    vm_read_weights((int8_t *)q_db1, 64);
    vm_maxtrix_multiply(0, 0, 1, 28 * 28 + 1, 28 * 28 + 1, 64);

    // 784X128 right part
    for (int i = 0; i < 28 * 28; i++) {
        vm_read_weights((int8_t *)q_dw1 + 64 + i * 128, 64);
    }
    vm_read_weights((int8_t *)q_db1 + 64, 64);
    vm_maxtrix_multiply(0, 64, 1, 28 * 28 + 1, 28 * 28 + 1, 64);

    vm_activate(ACT_TYPE_RELU, 0, 0, 128);

    // 128X10
    vm_read_host_memory(128, test_input_plus, 1);
    vm_read_weights((int8_t *)q_dw2, 128 * 10);
    vm_read_weights((int8_t *)q_db2, 10);
    vm_maxtrix_multiply(0, 0, 1, 128 + 1, 128 + 1, 10);
    vm_activate(ACT_TYPE_MAX, 0, 0, 10);

    vm_write_host_memory(test_y, 0, 10);

    return result(test_y);
}

static float get_abs_max(float *data, int len)
{
    float max = 0;
    for (int i = 0; i < len; i ++) {
        if (fabs(data[i]) > max) {
            max = fabs(data[i]);
        }
    }
    return max;
}

static int quantize_data(float *data, int8_t *q_data, int len, float max, int region)
{
    for (int i = 0; i < len; i ++) {
        q_data[i] = data[i] / max * region;
    }
    return 0;
}

static int quantize_wb()
{
    float abs_max = 0;
    float max_w1 = get_abs_max((float *)dw1, 784 * 128);
    abs_max = abs_max < max_w1 ? max_w1 : abs_max;
    float max_b1 = get_abs_max((float *)db1, 128);
    abs_max = abs_max < max_b1 ? max_b1 : abs_max;
    float max_w2 = get_abs_max((float *)dw2, 128 * 10);
    abs_max = abs_max < max_w2 ? max_w2 : abs_max;
    float max_b2 = get_abs_max((float *)db2, 10);
    abs_max = abs_max < max_b2 ? max_b2 : abs_max;

    int region = 16;
    quantize_data((float *)dw1, (int8_t *)q_dw1, 784 * 128, abs_max, region);
    quantize_data((float *)db1, (int8_t *)q_db1, 128, abs_max, region);
    quantize_data((float *)dw2, (int8_t *)q_dw2, 128 * 10, abs_max, region);
    quantize_data((float *)db2, (int8_t *)q_db2, 10, abs_max, region);
    return 0;
}

static int quantize_x_data(float data[28][28], int8_t q_data[28][28])
{
    //printf("\n");
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            q_data[i][j] = data[i][j] * 10;
            //printf("%d ", q_data[i][j]);
        }
        //printf("\n");
    }
}

static int quantize_x()
{
    quantize_x_data(test_data1, test_q_data1);
    quantize_x_data(test_data2, test_q_data2);
    quantize_x_data(test_data3, test_q_data3);
    quantize_x_data(test_data4, test_q_data4);
    quantize_x_data(test_data5, test_q_data5);
}

int main(int argc, char *argv[])
{
    quantize_x();
    quantize_wb();
    vm_init();
    printf("predict %d, label %d\n", calculate((int8_t *)test_q_data1), test_y1);
    printf("predict %d, label %d\n", calculate((int8_t *)test_q_data2), test_y2);
    printf("predict %d, label %d\n", calculate((int8_t *)test_q_data3), test_y3);
    printf("predict %d, label %d\n", calculate((int8_t *)test_q_data4), test_y4);
    printf("predict %d, label %d\n", calculate((int8_t *)test_q_data5), test_y5);
}