#include "vm32.h"
#include "cifar10_cnn_variables.h"
#include "cifar10_cnn_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static float c_x1[3][32][32];
static float c_x2[3][32][32];
static float c_x3[3][32][32];
static float c_x4[3][32][32];
static float c_x5[3][32][32];

static float c_conv0_k[32][3][3][3];
static float c_conv1_k[64][32][3][3];
static float c_conv2_k[64][64][3][3];

static float debug_data[32][30][30];

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

static void convert_flatten(float *in, float *out, int channel, int row, int col)
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

static int result(float *y)
{
    float max = -100000;
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (max < y[i]) {
            max = y[i];
            max_index = i;
        }
    }
    return max_index;
}

static void debug(float *in, int channel, int row, int col)
{
    for (int ir = 0; ir < row; ir++) {
        for (int ic = 0; ic < col; ic++) {
            for (int ci = 0; ci < channel; ci++) {
                //int tf_offset = ir * col * channel + ic * channel + ci;
                int in_offset = ci * row * col + ir * col + ic;
                printf("%f ", in[in_offset]);   
            }
            printf("\n");
        }
        printf("\n");
    }
}

static void debug_k(float *in, int kernel_size, int channel, int row, int col)
{
    for (int ir = 0; ir < row; ir++) {
        for (int ic = 0; ic < col; ic++) {
            for (int ci = 0; ci < channel; ci++) {
                for (int ki = 0; ki < kernel_size; ki++) {
                    int in_offset = ir * col * channel * kernel_size + ic * channel * kernel_size + ci * kernel_size + ki;
                    printf("%f ", in[in_offset]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

static int calculate(float *x, int y)
{
    float flatten_tmp[1024];
    float c_flatten[1024];
    float output[10];

    vm32_read_host_memory(0, x, 32 * 32 * 3);

    // conv0 32 * 32 * 3 -> 30 * 30 * 32
    vm32_read_weights((float *)c_conv0_k, 32 * 3 * 3 * 3);
    vm32_convolve(0, 0, 32, 32, 3, 32, 3, 3, 1, 0);
    vm32_read_weights((float *)conv0_b, 32);
    vm32_conv_bias(0, 30, 30, 32);
    vm32_activate(ACT32_TYPE_RELU, 0, 0, 30 * 30 * 32);
    // vm32_write_host_memory(debug_data, 0, 30 * 30 * 32);
    // debug(debug_data, 32, 30, 30);
    // vm32_read_host_memory(0, debug_data, 30 * 30 * 32);

    // max_pooling0 30 * 30 * 32 -> 15 * 15 * 32
    vm32_max_pooling(0, 0, 30, 30, 32, 2);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 15 * 15 * 32);
    // vm32_write_host_memory(debug_data, 0, 15 * 15 * 32);
    // debug(debug_data, 32, 15, 15);
    // vm32_read_host_memory(0, debug_data, 15 * 15 * 32);

    // conv1 15 * 15 * 32 -> 13 * 13 * 64
    vm32_read_weights((float *)c_conv1_k, 64 * 32 * 3 * 3);
    vm32_convolve(0, 0, 15, 15, 32, 64, 3, 3, 1, 0);
    vm32_read_weights((float *)conv1_b, 64);
    vm32_conv_bias(0, 13, 13, 64);
    vm32_activate(ACT32_TYPE_RELU, 0, 0, 13 * 13 * 64); 
    // vm32_write_host_memory(debug_data, 0, 13 * 13 * 64);
    // debug(debug_data, 64, 13, 13);
    // vm32_read_host_memory(0, debug_data, 13 * 13 * 64);

    // max_pooling1 13 * 13 * 64 -> 6 * 6 * 64
    vm32_max_pooling(0, 0, 13, 13, 64, 2);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 6 * 6 * 64);
    // vm32_write_host_memory(debug_data, 0, 6 * 6 * 64);
    // debug(debug_data, 64, 6, 6);
    // vm32_read_host_memory(0, debug_data, 6 * 6 * 64);

    // conv2 6 * 6 * 64 -> 4 * 4 * 64
    vm32_read_weights((float *)c_conv2_k, 64 * 64 * 3 * 3);
    vm32_convolve(0, 0, 6, 6, 64, 64, 3, 3, 1, 0);
    vm32_read_weights((float *)conv2_b, 64);
    vm32_conv_bias(0, 4, 4, 64);
    vm32_activate(ACT32_TYPE_RELU, 0, 0, 4 * 4 * 64); 
    // vm32_write_host_memory(debug_data, 0, 4 * 4 * 64);
    // debug(debug_data, 64, 4, 4);
    // vm32_read_host_memory(0, debug_data, 4 * 4 * 64);

    // flatten
    vm32_write_host_memory(flatten_tmp, 0, 1024);
    convert_flatten((float *)flatten_tmp, (float *)c_flatten, 64, 4, 4);
    vm32_read_host_memory(0, (float *)c_flatten, 1024);
    // vm32_write_host_memory(debug_data, 0, 1024);
    // debug(debug_data, 1, 1, 1024);
    // vm32_read_host_memory(0, debug_data, 1024);

    // d0
    vm32_read_weights((float *)d0_k, 1024 * 64);
    vm32_maxtrix_multiply(0, 0, 1, 1024, 1024, 64);
    vm32_read_weights(d0_b, 64);
    vm32_matmul_bias(0, 64);
    vm32_activate(ACT32_TYPE_RELU, 0, 0, 64);
    // vm32_write_host_memory(debug_data, 0, 64);
    // debug(debug_data, 1, 1, 64);
    // vm32_read_host_memory(0, debug_data, 64);

    // d1
    vm32_read_weights((float *)d1_k, 64 * 10);
    vm32_maxtrix_multiply(0, 0, 1, 64, 64, 10);
    vm32_read_weights(d1_b, 10);
    vm32_matmul_bias(0, 10);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 10);
    vm32_write_host_memory(output, 0, 10);

    printf("\n");
    printf("predict: %d, %s, test_y: %d, %s \n", result(output), class_names[result(output)], y, class_names[y]);
    printf("detail:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", output[i]);
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
    vm32_init();
    calculate((float *)c_x1, test_y1);
    calculate((float *)c_x2, test_y2);
    calculate((float *)c_x3, test_y3);
    calculate((float *)c_x4, test_y4);
    calculate((float *)c_x5, test_y5);
}