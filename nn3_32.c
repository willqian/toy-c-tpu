#include "vm32.h"

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

static float test_input_plus[1] = {1}; // 处理bias的矩阵运算

static float test_y1[1]= {0};
static float test_y2[1]= {0};
static float test_y3[1]= {0};
static float test_y4[1]= {0};
static float test_y5[1]= {0};
static float test_y6[1]= {0};

static int calculate(float *x, int xn, float *y, int yn)
{
    vm32_read_host_memory(0, x, xn);
    vm32_read_host_memory(2, test_input_plus, 1);
    vm32_read_weights((float *)w1, 6);
    vm32_read_weights((float *)b1, 3);
    vm32_maxtrix_multiply(0, 0, 1, 3, 3, 3);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 3);
    vm32_read_host_memory(3, test_input_plus, 1);
    vm32_read_weights((float *)w2, 3);
    vm32_read_weights((float *)b2, 1);
    vm32_maxtrix_multiply(0, 0, 1, 4, 4, 1);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 1);
    vm32_write_host_memory(y, 0, yn);
    return 0;
}

int main(int argc ,char *argv[])
{
    vm32_init();

    calculate(test_x1, 2, test_y1, 1);
    printf("test_y1 %f\n", test_y1[0]);
    calculate(test_x2, 2, test_y2, 1);
    printf("test_y2 %f\n", test_y2[0]);
    calculate(test_x3, 2, test_y3, 1);
    printf("test_y3 %f\n", test_y3[0]);
    calculate(test_x4, 2, test_y4, 1);
    printf("test_y4 %f\n", test_y4[0]);
    calculate(test_x5, 2, test_y5, 1);
    printf("test_y5 %f\n", test_y5[0]);
    calculate(test_x6, 2, test_y6, 1);
    printf("test_y6 %f\n", test_y6[0]);
}