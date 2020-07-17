#include "vm.h"
#include "cifar10_cnn_variables.h"
#include "cifar10_cnn_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static float c_x1[3][32][32];
static float c_x2[3][32][32];
static float c_x3[3][32][32];
static float c_x4[3][32][32];
static float c_x5[3][32][32];

static float c_conv0_k[32][3][3][3];
static float c_conv1_k[64][32][3][3];
static float c_conv2_k[64][64][3][3];

static int8_t i_x1[3][32][32];
static int8_t i_x2[3][32][32];
static int8_t i_x3[3][32][32];
static int8_t i_x4[3][32][32];
static int8_t i_x5[3][32][32];

static int8_t i_conv0_k[32][3][3][3];
static int8_t i_conv1_k[64][32][3][3];
static int8_t i_conv2_k[64][64][3][3];
static int8_t i_conv0_b[32];
static int8_t i_conv1_b[64];
static int8_t i_conv2_b[64];

static int8_t i_d0_k[1024][64];
static int8_t i_d0_b[64];
static int8_t i_d1_k[64][10];
static int8_t i_d1_b[10];

static int8_t debug_data[32][30][30];

static void convert_x(float *in, float *out, int channel, int row, int col)
{
    for (int ci = 0; ci < channel; ci++) {
        for (int ir = 0; ir < row; ir++) {
            for (int ic = 0; ic < col; ic++) {
                // row-col-channel
                int in_offset = ir * col * channel  + ic * channel + ci;
                // channel-row-col
                int out_offset = ci * row * col + ir * col + ic;
                out[out_offset] = in[in_offset];
            }
        }
    }
}

static void convert_conv_k(float *in, float *out, int kernel_size, int channel, int row, int col)
{
    for (int ki = 0; ki < kernel_size; ki++) {
        for (int ci = 0; ci < channel; ci++) {
            for (int ir = 0; ir < row; ir++) {
                for (int ic = 0; ic < col; ic++) {
                    // row-col-channel-kernel
                    int in_offset = ir * col * channel * kernel_size + ic * channel * kernel_size + ci * kernel_size + ki;
                    // kernel-channel-row-col
                    int out_offset = ki * channel * row * col + ci * row * col + ir * col + ic;
                    out[out_offset] = in[in_offset];
                }
            }
        }
    }
}

static void convert_flatten(int8_t *in, int8_t *out, int channel, int row, int col)
{
    for (int ci = 0; ci < channel; ci++) {
        for (int ir = 0; ir < row; ir++) {
            for (int ic = 0; ic < col; ic++) {
                // channel-row-col
                int in_offset = ci * row * col + ir * col + ic;
                // row-col-channel
                int out_offset = ir * col * channel  + ic * channel + ci;
                out[out_offset] = in[in_offset];
            }
        }
    }
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

static int quantize_data(float *data, int8_t *q_data, int len, float max, int range)
{
    for (int i = 0; i < len; i ++) {
        q_data[i] = data[i] / max * range;
    }
    return 0;
}

static int quantize_wb()
{
    float abs_max = 0;
    float max_conv0_k = get_abs_max((float *)c_conv0_k, 32 * 3 * 3 * 3);
    abs_max = abs_max < max_conv0_k ? max_conv0_k : abs_max;
    printf("max_conv0_k %f\n", max_conv0_k);
    float max_conv1_k = get_abs_max((float *)c_conv1_k, 64 * 32 * 3 * 3);
    abs_max = abs_max < max_conv1_k ? max_conv1_k : abs_max;
    printf("max_conv1_k %f\n", max_conv1_k);
    float max_conv2_k = get_abs_max((float *)c_conv2_k, 64 * 64 * 3 * 3);
    abs_max = abs_max < max_conv2_k ? max_conv2_k : abs_max;
    printf("max_conv2_k %f\n", max_conv2_k);
    float max_conv0_b = get_abs_max((float *)conv0_b, 32);
    abs_max = abs_max < max_conv0_b ? max_conv0_b : abs_max;
    printf("max_conv0_b %f\n", max_conv0_b);
    float max_conv1_b = get_abs_max((float *)conv1_b, 64);
    abs_max = abs_max < max_conv1_b ? max_conv1_b : abs_max;
    printf("max_conv1_b %f\n", max_conv1_b);
    float max_conv2_b = get_abs_max((float *)conv2_b, 64);
    abs_max = abs_max < max_conv2_b ? max_conv2_b : abs_max;
    printf("max_conv2_b %f\n", max_conv2_b);
    float max_d0_k = get_abs_max((float *)d0_k, 1024 * 64);
    abs_max = abs_max < max_d0_k ? max_d0_k : abs_max;
    printf("max_d0_k %f\n", max_d0_k);
    float max_d1_k = get_abs_max((float *)d1_k, 64 * 10);
    abs_max = abs_max < max_d1_k ? max_d1_k : abs_max;
    printf("max_d1_k %f\n", max_d1_k);
    float max_d0_b = get_abs_max((float *)d0_b, 64);
    abs_max = abs_max < max_d0_b ? max_d0_b : abs_max;
    printf("max_d0_b %f\n", max_d0_b);
    float max_d1_b = get_abs_max((float *)d1_b, 10);
    abs_max = abs_max < max_d1_b ? max_d1_b : abs_max;
    printf("max_d1_b %f\n", max_d1_b);
    printf("abs max %f\n", abs_max);

    int range = 127; 
    // quantize_data((float *)c_conv0_k, (int8_t *)i_conv0_k, 32 * 3 * 3 * 3, max_conv0_k, range);
    // quantize_data((float *)c_conv1_k, (int8_t *)i_conv1_k, 64 * 32 * 3 * 3, max_conv1_k, range);
    // quantize_data((float *)c_conv2_k, (int8_t *)i_conv2_k, 64 * 64 * 3 * 3, max_conv2_k, range);
    // quantize_data((float *)conv0_b, (int8_t *)i_conv0_b, 32, max_conv0_k, range);
    // quantize_data((float *)conv1_b, (int8_t *)i_conv1_b, 64, max_conv1_k, range);
    // quantize_data((float *)conv2_b, (int8_t *)i_conv2_b, 64, max_conv2_k, range);
    // quantize_data((float *)d0_k, (int8_t *)i_d0_k, 1024 * 64, max_d0_k, range);
    // quantize_data((float *)d1_k, (int8_t *)i_d1_k, 64 * 10, max_d1_k, range);
    // quantize_data((float *)d0_b, (int8_t *)i_d0_b, 64, max_d0_k, range);
    // quantize_data((float *)d1_b, (int8_t *)i_d1_b, 10, max_d1_k, range);
    quantize_data((float *)c_conv0_k, (int8_t *)i_conv0_k, 32 * 3 * 3 * 3, abs_max, range);
    quantize_data((float *)c_conv1_k, (int8_t *)i_conv1_k, 64 * 32 * 3 * 3, abs_max, range);
    quantize_data((float *)c_conv2_k, (int8_t *)i_conv2_k, 64 * 64 * 3 * 3, abs_max, range);
    quantize_data((float *)conv0_b, (int8_t *)i_conv0_b, 32, abs_max, range);
    quantize_data((float *)conv1_b, (int8_t *)i_conv1_b, 64, abs_max, range);
    quantize_data((float *)conv2_b, (int8_t *)i_conv2_b, 64, abs_max, range);
    quantize_data((float *)d0_k, (int8_t *)i_d0_k, 1024 * 64, abs_max, range);
    quantize_data((float *)d1_k, (int8_t *)i_d1_k, 64 * 10, abs_max, range);
    quantize_data((float *)d0_b, (int8_t *)i_d0_b, 64, abs_max, range);
    quantize_data((float *)d1_b, (int8_t *)i_d1_b, 10, abs_max, range);
    return 0;
}

static void quantize_x_data(float *data, int8_t *q_data, int channel, int row, int col, int range)
{
    for (int ci = 0; ci < channel; ci++) {
        for (int ir = 0; ir < row; ir++) {
            for (int ic = 0; ic < col; ic++) {
                int offset = ci * row * col + ir * col + ic;
                q_data[offset] = data[offset] * range;
            }
        }
    }
}

static int x_range = 127;

static int quantize_x()
{
    int range = x_range;
    quantize_x_data((float *)c_x1, (int8_t *)i_x1, 3, 32, 32, range);
    quantize_x_data((float *)c_x2, (int8_t *)i_x2, 3, 32, 32, range);
    quantize_x_data((float *)c_x3, (int8_t *)i_x3, 3, 32, 32, range);
    quantize_x_data((float *)c_x4, (int8_t *)i_x4, 3, 32, 32, range);
    quantize_x_data((float *)c_x5, (int8_t *)i_x5, 3, 32, 32, range);
}

static int result(int8_t *y)
{
    int8_t max = -127;
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (max < y[i]) {
            max = y[i];
            max_index = i;
        }
    }
    return max_index;
}

static void debug(int8_t *in, int channel, int row, int col)
{
    for (int ir = 0; ir < row; ir++) {
        for (int ic = 0; ic < col; ic++) {
            for (int ci = 0; ci < channel; ci++) {
                //int tf_offset = ir * col * channel + ic * channel + ci;
                int in_offset = ci * row * col + ir * col + ic;
                printf("%d ", in[in_offset]);   
            }
            printf("\n");
        }
        printf("\n");
    }
}

static void debug_k(int8_t *in, int kernel_size, int channel, int row, int col)
{
    for (int ir = 0; ir < row; ir++) {
        for (int ic = 0; ic < col; ic++) {
            for (int ci = 0; ci < channel; ci++) {
                for (int ki = 0; ki < kernel_size; ki++) {
                    int in_offset = ir * col * channel * kernel_size + ic * channel * kernel_size + ci * kernel_size + ki;
                    printf("%d ", in[in_offset]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

static int calculate(int8_t *x, int y)
{
    int8_t flatten_tmp[1024];
    int8_t c_flatten[1024];
    int8_t output[10];
    int32_t alpha = 127 * 127;
    int32_t beta = 127;
    int32_t max = 0;

    vm_read_host_memory(0, x, 32 * 32 * 3);

    // conv0 32 * 32 * 3 -> 30 * 30 * 32
    vm_read_weights((int8_t *)i_conv0_k, 32 * 3 * 3 * 3);
    vm_convolve(0, 0, 32, 32, 3, 32, 3, 3, 1, 0);
    vm_read_weights((int8_t *)i_conv0_b, 32);
    vm_conv_bias(0, 30, 30, 32, alpha / 127);
    //vm_debug_acc(0, 32, 30, 30); 
    vm_normalize(0, 30 * 30 * 32, 127, &max);
    beta = max / 127;
    alpha = alpha / beta;
    vm_activate(ACT_TYPE_RELU, 0, 0, 30 * 30 * 32);
    // vm_write_host_memory(debug_data, 0, 30 * 30 * 32);
    // debug(debug_data, 32, 30, 30);
    // vm_read_host_memory(0, debug_data, 30 * 30 * 32);
        
    // max_pooling0 30 * 30 * 32 -> 15 * 15 * 32
    vm_max_pooling(0, 0, 30, 30, 32, 2);
    //vm_debug_acc(0, 32, 15, 15);
    vm_activate(ACT_TYPE_NONE, 0, 0, 15 * 15 * 32);
    // vm_write_host_memory(debug_data, 0, 15 * 15 * 32);
    // debug(debug_data, 32, 15, 15);
    // vm_read_host_memory(0, debug_data, 15 * 15 * 32);

    // conv1 15 * 15 * 32 -> 13 * 13 * 64
    vm_read_weights((int8_t *)i_conv1_k, 64 * 32 * 3 * 3);
    vm_convolve(0, 0, 15, 15, 32, 64, 3, 3, 1, 0);
    vm_read_weights((int8_t *)i_conv1_b, 64);
    vm_conv_bias(0, 13, 13, 64, alpha / 127);
    vm_normalize(0, 13 * 13 * 64, 127, &max);
    beta = max / 127;
    alpha = alpha / beta;
    vm_activate(ACT_TYPE_RELU, 0, 0, 13 * 13 * 64);
    // vm_write_host_memory(debug_data, 0, 13 * 13 * 64);
    // debug(debug_data, 64, 13, 13);
    // vm_read_host_memory(0, debug_data, 13 * 13 * 64);

    // max_pooling1 13 * 13 * 64 -> 6 * 6 * 64
    vm_max_pooling(0, 0, 13, 13, 64, 2);
    vm_activate(ACT_TYPE_NONE, 0, 0, 6 * 6 * 64);
    // vm_write_host_memory(debug_data, 0, 6 * 6 * 64);
    // debug(debug_data, 64, 6, 6);
    // vm_read_host_memory(0, debug_data, 6 * 6 * 64);

    // conv2 6 * 6 * 64 -> 4 * 4 * 64
    vm_read_weights((int8_t *)i_conv2_k, 64 * 64 * 3 * 3);
    vm_convolve(0, 0, 6, 6, 64, 64, 3, 3, 1, 0);
    vm_read_weights((int8_t *)i_conv2_b, 64);
    vm_conv_bias(0, 4, 4, 64, alpha / 127);
    vm_normalize(0, 4 * 4 * 64, 127, &max);
    beta = max / 127;
    alpha = alpha / beta;
    vm_activate(ACT_TYPE_RELU, 0, 0, 4 * 4 * 64); 
    // vm_write_host_memory(debug_data, 0, 4 * 4 * 64);
    // debug(debug_data, 64, 4, 4);
    // vm_read_host_memory(0, debug_data, 4 * 4 * 64);

    // flatten
    vm_write_host_memory(flatten_tmp, 0, 1024);
    convert_flatten((int8_t *)flatten_tmp, (int8_t *)c_flatten, 64, 4, 4);
    vm_read_host_memory(0, (int8_t *)c_flatten, 1024);
    // vm_write_host_memory(debug_data, 0, 1024);
    // debug(debug_data, 1, 1, 1024);
    // vm_read_host_memory(0, debug_data, 1024);

    // d0
    vm_read_weights((int8_t *)i_d0_k, 1024 * 64);
    vm_maxtrix_multiply(0, 0, 1, 1024, 1024, 64);
    vm_read_weights(i_d0_b, 64);
    vm_matmul_bias(0, 64, alpha / 127);
    vm_normalize(0, 64, 127, &max);
    beta = max / 127;
    alpha = alpha / beta;
    vm_activate(ACT_TYPE_RELU, 0, 0, 64);
    // vm_write_host_memory(debug_data, 0, 64);
    // debug(debug_data, 1, 1, 64);
    // vm_read_host_memory(0, debug_data, 64);

    // d1
    vm_read_weights((int8_t *)i_d1_k, 64 * 10);
    vm_maxtrix_multiply(0, 0, 1, 64, 64, 10);
    vm_read_weights(i_d1_b, 10);
    vm_matmul_bias(0, 10, alpha / 127);
    vm_normalize(0, 10, 127, &max);
    beta = max / 127;
    alpha = alpha / beta;
    vm_activate(ACT_TYPE_NONE, 0, 0, 10);
    vm_write_host_memory(output, 0, 10);

    printf("\n");
    printf("predict: %d, %s, test_y: %d, %s \n", result(output), class_names[result(output)], y, class_names[y]);
    printf("detail:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
}

int main(int argc ,char *argv[])
{
    convert_x((float *)x1, (float *)c_x1, 3, 32, 32);
    convert_x((float *)x2, (float *)c_x2, 3, 32, 32);
    convert_x((float *)x3, (float *)c_x3, 3, 32, 32);
    convert_x((float *)x4, (float *)c_x4, 3, 32, 32);
    convert_x((float *)x5, (float *)c_x5, 3, 32, 32);
    convert_conv_k((float *)conv0_k, (float *)c_conv0_k, 32, 3, 3, 3);
    convert_conv_k((float *)conv1_k, (float *)c_conv1_k, 64, 32, 3, 3);
    convert_conv_k((float *)conv2_k, (float *)c_conv2_k, 64, 64, 3, 3);
    quantize_x();
    quantize_wb();
    vm_init();
    //debug((int8_t *)i_x1, 3, 32, 32);
    calculate((int8_t *)i_x1, test_y1);
    calculate((int8_t *)i_x2, test_y2);
    calculate((int8_t *)i_x3, test_y3);
    calculate((int8_t *)i_x4, test_y4);
    calculate((int8_t *)i_x5, test_y5);
}