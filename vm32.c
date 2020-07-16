#include "vm32.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define DBG(fmt, args...) \
    do { \
        printf("[file %s, line %d, func %s] "fmt, \
                __FILE__, __LINE__, __func__, ##args); \
    } while(0)

//#define VM_INFO
#ifdef VM_INFO
#define INFO(fmt, args...) \
    do { \
        printf(fmt, ##args); \
    } while(0)
#else
#define INFO(fmt, args...) do {} while(0)
#endif

#define WEIGHT_FIFO_MAX_SIZE (256 * 256)

typedef struct {
    float data[WEIGHT_FIFO_MAX_SIZE];
    int read_index;
    int write_index;
    int size;
} weight_fifo_t;

static weight_fifo_t weight_fifo;
static float maxtrix_multiply_unit[256][256] = {0};
static float local_unified_buffer[96 * 1024 * 256] = {0};
static float accumulators[4 * 1024 * 256] = {0};

int vm32_init()
{
    memset(&weight_fifo, 0, sizeof(weight_fifo));
}

int vm32_read_host_memory(uint32_t unified_buffer_addr, float *host_addr, int n)
{
    memcpy(local_unified_buffer + unified_buffer_addr, host_addr, n * sizeof(float));
    return 0;
}

int vm32_read_weights(float *host_addr, int n)
{
    if ((weight_fifo.size + n) > WEIGHT_FIFO_MAX_SIZE) {
        DBG("vm_read_weights failed, weight fifo is full\n");
        return -1;
    }
    if ((weight_fifo.write_index + n) <= WEIGHT_FIFO_MAX_SIZE) {
        memcpy(weight_fifo.data + weight_fifo.write_index, host_addr, n * sizeof(float));
    } else {
        int first_part_size = WEIGHT_FIFO_MAX_SIZE - weight_fifo.write_index;
        memcpy(weight_fifo.data + weight_fifo.write_index, host_addr, first_part_size * sizeof(float));
        memcpy(weight_fifo.data, host_addr + first_part_size, (n - first_part_size) * sizeof(float));
    }
    weight_fifo.write_index = (weight_fifo.write_index + n) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size += n;
    INFO("fifo:\n");
    for (int i = 0; i < weight_fifo.size; i++) {
        INFO("%f ", weight_fifo.data[(weight_fifo.read_index + i) % WEIGHT_FIFO_MAX_SIZE]);
    }
    INFO("\n");
    return 0;
}

int vm32_maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, uint16_t weight_row, uint16_t weight_col)
{
    if (weight_fifo.size < weight_row * weight_col) {
        DBG("vm32_maxtrix_multiply failed, weight fifo is too small\n");
        return -1;
    }
    INFO("maxtrix_multiply:\n");
    for (int lcol = 0; lcol < input_col; lcol++) {
        for (int lrow = 0; lrow < input_row; lrow++) {
            for (int rcol = 0; rcol < weight_col; rcol++) {
                int laddr_offset = lrow * input_col + lcol + unified_buffer_addr;
                int raddr_offset = lcol * weight_col + rcol;
                int fifo_index = (raddr_offset + weight_fifo.read_index) % WEIGHT_FIFO_MAX_SIZE;
                maxtrix_multiply_unit[lrow][rcol] = local_unified_buffer[laddr_offset] * weight_fifo.data[fifo_index];
                accumulators[accumulator_addr + lrow * weight_col + rcol] += maxtrix_multiply_unit[lrow][rcol];
                INFO("%f * %f = %f, sum[%d][%d] = %f\n", local_unified_buffer[laddr_offset], weight_fifo.data[fifo_index],
                        local_unified_buffer[laddr_offset] * weight_fifo.data[fifo_index], 
                        lrow, rcol, maxtrix_multiply_unit[lrow][rcol]);
            }
        }
        INFO(".\n");
    }
    INFO("\n");
    weight_fifo.read_index = (weight_fifo.read_index + weight_row * weight_col) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= weight_row * weight_col;
    return 0;
}

int vm32_convolve(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, int channel, int kernel_size, uint16_t kernel_row, uint16_t kernel_col,
        int stride, int padding)
{
    if (weight_fifo.size < kernel_row * kernel_col * channel * kernel_size) {
        DBG("vm32_convolve failed, weight fifo is too small\n");
        return -1;
    }
    int output_row = (input_row - kernel_row + padding * 2) / stride + 1;
    int output_col = (input_col - kernel_col + padding * 2) / stride + 1;
    int k_row_center = kernel_row / 2;
    int k_col_center = kernel_col / 2;
    INFO("output row %d, output col %d, total %d, k_row_center %d, k_col_center %d\n",
            output_row, output_col, output_row * output_col, k_row_center, k_col_center);
    for (int ki = 0; ki < kernel_size; ki++) {
        for (int ci = 0; ci < channel; ci++) {
            for (int kr = 0; kr < kernel_row; kr++) {
                for (int kc = 0; kc < kernel_col; kc++) {
                    for (int ir = 0; ir < input_row + padding * 2; ir++) {
                        for (int ic = 0; ic < input_col + padding * 2; ic++) {
                            if ((ir - padding) < 0 || (ir - padding) >= input_row
                                    || (ic - padding) < 0 || (ic - padding) >= input_col) {
                                INFO("input[%d][%d], k[%d][%d], padding %d, no need calculate\n", ir, ic, kr, kc, padding);
                                continue;
                            } 
                            if ((ir - kr) < 0 || (ir - kr) >= (input_row - k_row_center - 1 + 2 * padding)
                                    || (ic- kc) < 0 || (ic- kc) >= (input_col - k_col_center - 1 + 2 * padding)) {
                                INFO("input[%d][%d], k[%d][%d], padding %d, invalid\n", ir, ic, kr, kc, padding);
                                continue;
                            }
                            if ((ir - kr) % stride != 0 || (ic - kc) % stride != 0) {
                                INFO("input[%d][%d], k[%d][%d], padding %d, stride %d, invalid\n", ir, ic, kr, kc, padding, stride);
                                continue;
                            }
                            int ub_offset = ci * input_row * input_col + (ir - padding) * input_col + (ic - padding);
                            int k_offset = ki * channel * kernel_row * kernel_col + ci * kernel_row * kernel_col
                                    + kr * kernel_col + kc;
                            int fifo_index = (k_offset + weight_fifo.read_index) % WEIGHT_FIFO_MAX_SIZE;
                            int or = (ir - kr) / stride;
                            int oc = (ic - kc) / stride;
                            int a_offset = ki * output_row * output_col + or * output_col + oc;
                            INFO("ub_offset %d, fifo_index %d, or %d, oc %d, a_offset %d\n", ub_offset, fifo_index, or, oc, a_offset);
                            float input = local_unified_buffer[unified_buffer_addr + ub_offset];
                            float k = weight_fifo.data[fifo_index];
                            accumulators[accumulator_addr + a_offset] += input * k;
                            INFO("input[%d,%d,%d]:%f, k[%d,%d,%d,%d]:%f, a[%d]:%f\n", ci, ir, ic, input, ki, ci, kr, kc, k, a_offset, accumulators[accumulator_addr + a_offset]);
                        }
                    }
                }
            }
        }
    }
    weight_fifo.read_index = (weight_fifo.read_index + kernel_row * kernel_col * channel * kernel_size) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= kernel_row * kernel_col * channel * kernel_size;
    return 0;
}

int vm32_conv_bias(uint16_t accumulator_addr, uint16_t out_row, uint16_t out_col, int out_channel)
{
    if (weight_fifo.size < out_channel) {
        DBG("vm32_conv_bias failed, weight fifo is too small\n");
        return -1;
    }
    for (int ci = 0; ci < out_channel; ci++) {
        for (int i = 0; i < out_row * out_col; i++) {
            int a_offset = ci * out_row * out_col + i;
            int w_offset = ci;
            int fifo_index = (w_offset + weight_fifo.read_index) % WEIGHT_FIFO_MAX_SIZE;
            accumulators[accumulator_addr + a_offset] += weight_fifo.data[fifo_index];
        }
    }
    weight_fifo.read_index = (weight_fifo.read_index + out_channel) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= out_channel;
    return 0;
}

int vm32_max_pooling(uint32_t unified_buffer_addr, uint16_t accumulator_addr, uint16_t row, uint16_t col, int channel, int pool_size)
{
    int a_row = row / pool_size;
    int a_col = col / pool_size;
    for (int ci = 0; ci < channel; ci++) {
        for (int ir = 0; ir < (row - pool_size + 1); ir += pool_size) {
            for (int ic = 0; ic < (col - pool_size + 1); ic += pool_size) {
                int air = ir / pool_size;
                int aic = ic / pool_size;
                int a_offset = ci * a_row * a_col + air * a_col + aic;
                float max = -10000;
                for (int i = 0; i < pool_size; i++) {
                    for (int j = 0; j < pool_size; j++) {
                        int ub_offset = ci * row * col + (ir + i) * col + (ic + j);
                        max = max < local_unified_buffer[unified_buffer_addr + ub_offset] 
                                ? local_unified_buffer[unified_buffer_addr + ub_offset] : max;
                        INFO("[%d,%d,%d]:%f ", ci, ir + i, ic + j, local_unified_buffer[unified_buffer_addr + ub_offset]);
                    }
                }
                accumulators[accumulator_addr + a_offset] = max;
                INFO("[%d] max %f\n", a_offset, accumulators[accumulator_addr + a_offset]);
            }
        }
    }
}

static int relu(uint16_t accumulator_addr, int len)
{
    for (int i = 0; i < len; i++) {
        if (accumulators[accumulator_addr + i] < 0) {
            accumulators[accumulator_addr + i] = 0;
        }
    }
    return 0;
}

static int softmax(uint16_t accumulator_addr, int len)
{
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += exp(accumulators[accumulator_addr + i]);
    }
    for (int i = 0; i < len; i++) {
        accumulators[accumulator_addr + i] = exp(accumulators[accumulator_addr + i]) / sum;
    }
    return 0;
}

int vm32_activate(act32_type_enum_t type, uint16_t accumulator_addr, uint32_t unified_buffer_addr, int n)
{
    int ret;
    switch (type) {
    case ACT32_TYPE_NONE:
        break;
    case ACT32_TYPE_RELU:
        ret = relu(accumulator_addr, n); 
        break;
    case ACT32_TYPE_SOFTMAX:
        ret = softmax(accumulator_addr, n); 
        break;
    default:
        break;
    }
    for (int i = 0; i < n; i ++) {
        local_unified_buffer[unified_buffer_addr + i] = accumulators[accumulator_addr + i];
        accumulators[accumulator_addr + i] = 0;
    }
}

int vm32_write_host_memory(float *host_addr, uint32_t unified_buffer_addr, int n)
{
    memcpy(host_addr, local_unified_buffer + unified_buffer_addr, n * sizeof(float));
    return 0;
}