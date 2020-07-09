#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static float w1[2][3] = {
    {0.12360667, -0.70676994, -1.3703936},
    {-2.8751068 , -1.1498548 , -1.6827904}
};

static float w2[3][1] = {
    {-2.9926474},
    {-2.2082868},
    {-2.2191722}
};

static float b1[3] = {1.5301738 ,  1.7288847 , -0.36803755};
static float b2[1] = {-2.5253797};

static float test_x1[2] = {0.7, 0.9};
static float test_x2[2] = {0.6, 0.8};
static float test_x3[2] = {0.9, 0.7};
static float test_x4[2] = {0.1, 0.4};
static float test_x5[2] = {0.4, 0.1};
static float test_x6[2] = {0.2, 0.2};

static int8_t q_w1[2][3] = {0};
static int8_t q_w2[3][1] = {0};
static int8_t q_b1[3] = {0};
static int8_t q_b2[1] = {0};

static int8_t test_q_x1[2] = {0};
static int8_t test_q_x2[2] = {0};
static int8_t test_q_x3[2] = {0};
static int8_t test_q_x4[2] = {0};
static int8_t test_q_x5[2] = {0};
static int8_t test_q_x6[2] = {0};

static int8_t test_input_plus[1] = {1}; // 处理bias的矩阵运算

static int8_t test_q_y1[1]= {0};
static int8_t test_q_y2[1]= {0};
static int8_t test_q_y3[1]= {0};
static int8_t test_q_y4[1]= {0};
static int8_t test_q_y5[1]= {0};
static int8_t test_q_y6[1]= {0};

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

static int quantize_data(float *data, int8_t *q_data, int len, float max)
{
    for (int i = 0; i < len; i ++) {
        q_data[i] = data[i] / max * 128;
    }
    return 0;
}

static int quantize_wb()
{
    float abs_max = 0;
    float max_w1 = get_abs_max((float *)w1, 6);
    abs_max = abs_max < max_w1 ? max_w1 : abs_max;
    float max_w2 = get_abs_max((float *)w2, 3);
    abs_max = abs_max < max_w2 ? max_w2 : abs_max;
    float max_b1 = get_abs_max((float *)b1, 3);
    abs_max = abs_max < max_b1 ? max_b1 : abs_max;
    float max_b2 = get_abs_max((float *)b2, 1);
    abs_max = abs_max < max_b2 ? max_b2 : abs_max;
    abs_max *= 32;
    //printf("abs_max %f\n", abs_max);
    quantize_data((float *)w1, (int8_t *)q_w1, 6, abs_max);
    quantize_data((float *)w2, (int8_t *)q_w2, 3, abs_max);
    quantize_data((float *)b1, (int8_t *)q_b1, 3, abs_max);
    quantize_data((float *)b2, (int8_t *)q_b2, 1, abs_max);
    printf("w1 [[%d %d %d], [%d %d %d]]\n", q_w1[0][0], q_w1[0][1], q_w1[0][2], q_w1[1][0], q_w1[1][1], q_w1[1][2]);
    printf("w2 [[%d], [%d], [%d]]\n", q_w2[0][0], q_w2[1][0], q_w2[2][0]);
    printf("b1 [%d, %d, %d]\n", q_b1[0], q_b1[1], q_b1[2]);
    printf("b2 [%d]\n", q_b2[0]);
    return 0;
}

static int quantize_x()
{
    int quant = 16;
    quantize_data((float *)test_x1, (int8_t *)test_q_x1, 2, quant);
    quantize_data((float *)test_x2, (int8_t *)test_q_x2, 2, quant);
    quantize_data((float *)test_x3, (int8_t *)test_q_x3, 2, quant);
    quantize_data((float *)test_x4, (int8_t *)test_q_x4, 2, quant);
    quantize_data((float *)test_x5, (int8_t *)test_q_x5, 2, quant);
    quantize_data((float *)test_x6, (int8_t *)test_q_x6, 2, quant);
    printf("x1 [%d, %d]\n", test_q_x1[0], test_q_x1[1]);
    printf("x2 [%d, %d]\n", test_q_x2[0], test_q_x2[1]);
    printf("x3 [%d, %d]\n", test_q_x3[0], test_q_x3[1]);
    printf("x4 [%d, %d]\n", test_q_x4[0], test_q_x4[1]);
    printf("x5 [%d, %d]\n", test_q_x5[0], test_q_x5[1]);
    printf("x6 [%d, %d]\n", test_q_x6[0], test_q_x6[1]);
}

static int calculate(int8_t *x, int xlen, int8_t *y, int ylen)
{
    vm_read_host_memory(0, x, xlen);
    vm_read_host_memory(2, (int8_t *)test_input_plus, 1);
    vm_read_weights((int8_t *)q_w1, 6);
    vm_read_weights((int8_t *)q_b1, 3);
    vm_maxtrix_multiply(0, 0, 1, 3, 3, 3);
    vm_activate(ACT_TYPE_NONE, 0, 3);
    vm_read_host_memory(3, (int8_t *)test_input_plus, 1);
    vm_read_weights((int8_t *)q_w2, 3);
    vm_read_weights((int8_t *)q_b2, 1);
    vm_maxtrix_multiply(0, 0, 1, 4, 4, 1);
    vm_activate(ACT_TYPE_NONE, 0, 1);
    vm_write_host_memory((int8_t *)y, 0, ylen);
    return 0;
}

int main(int argc ,char *argv[])
{
    quantize_wb();
    quantize_x();
    vm_init();

    calculate(test_q_x1, 2, test_q_y1, 1);
    printf("test_q_y1 %d\n", test_q_y1[0]);
    calculate(test_q_x2, 2, test_q_y2, 1);
    printf("test_q_y2 %d\n", test_q_y2[0]);
    calculate(test_q_x3, 2, test_q_y3, 1);
    printf("test_q_y3 %d\n", test_q_y3[0]);
    calculate(test_q_x4, 2, test_q_y4, 1);
    printf("test_q_y4 %d\n", test_q_y4[0]);
    calculate(test_q_x5, 2, test_q_y5, 1);
    printf("test_q_y5 %d\n", test_q_y5[0]);
    calculate(test_q_x6, 2, test_q_y6, 1);
    printf("test_q_y6 %d\n", test_q_y6[0]);
}