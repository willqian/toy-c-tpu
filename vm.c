#include "vm.h"

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

#define VM_WARN
#ifdef VM_WARN
#define WARN(fmt, args...) \
    do { \
        printf("[file %s, line %d, func %s] "fmt, \
                __FILE__, __LINE__, __func__, ##args); \
    } while(0)
#else
#define WARN(fmt, args...) do {} while(0)
#endif

// TODO: Sparsity 矩阵运算
// TODO: 根据总线位数和带宽，做32/64bit取input值
// TODO: 实现TPU硬件并行运算虚拟机
// TODO: 实现RNN网络
// TODO: 实现反向传播训练
// TODO: 大矩阵CNN卷积的拆分计算demo

// TODO: weight fifo不只有256*256
#define WEIGHT_FIFO_MAX_SIZE (256 * 256)

typedef struct {
    int8_t data[WEIGHT_FIFO_MAX_SIZE];
    int read_index;
    int write_index;
    int size;
} weight_fifo_t;

static weight_fifo_t weight_fifo;
static int8_t maxtrix_multiply_unit[256][256] = {0}; // 64K MAC
static int8_t local_unified_buffer[96 * 1024 * 256] = {0}; // 24MB
static int32_t accumulators[4 * 1024 * 256] = {0}; // 4MB

int vm_init()
{
    memset(&weight_fifo, 0, sizeof(weight_fifo));
}

int vm_read_host_memory(uint32_t unified_buffer_addr, uint8_t *host_addr, int len)
{
    memcpy(local_unified_buffer + unified_buffer_addr, host_addr, len);
    return 0;
}

int vm_read_weights(uint8_t *host_addr, int len)
{
    if ((weight_fifo.size + len) > WEIGHT_FIFO_MAX_SIZE) {
        DBG("vm_read_weights failed, weight fifo is full\n");
        return -1;
    }
    if ((weight_fifo.write_index + len) <= WEIGHT_FIFO_MAX_SIZE) {
        memcpy(weight_fifo.data + weight_fifo.write_index, host_addr, len);
    } else {
        int first_part_size = WEIGHT_FIFO_MAX_SIZE - weight_fifo.write_index;
        memcpy(weight_fifo.data + weight_fifo.write_index, host_addr, first_part_size);
        memcpy(weight_fifo.data, host_addr + first_part_size, len - first_part_size);
    }
    weight_fifo.write_index = (weight_fifo.write_index + len) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size += len;
    INFO("fifo:\n");
    for (int i = 0; i < weight_fifo.size; i++) {
        INFO("%d ", weight_fifo.data[(weight_fifo.read_index + i) % WEIGHT_FIFO_MAX_SIZE]);
    }
    INFO("\n");
    return 0;
}

// TODO: 实现TPU斜向load input和weight，提高并行计算速度到256
// TODO: 按照TPU论文，重新调整乘法的逻辑
// A matrix operation takes a variable-sized B*256 input, multiplies it by a
// 256x256 constant weight input, and produces a B*256 output, taking B pipelined cycles to complete
int vm_maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, uint16_t weight_row, uint16_t weight_col)
{
    if (weight_fifo.size < weight_row * weight_col) {
        DBG("vm_maxtrix_multiply failed, weight fifo is too small\n");
        return -1;
    }
    INFO("maxtrix_multiply:\n");
    for (int lcol = 0; lcol < input_col; lcol++) {
        for (int lrow = 0; lrow < input_row; lrow++) {
            for (int rcol = 0; rcol < weight_col; rcol++) {
                int laddr_offset = lrow * input_col + lcol + unified_buffer_addr;
                int raddr_offset = lcol * weight_col + rcol;
                int fifo_index = (raddr_offset + weight_fifo.read_index) % WEIGHT_FIFO_MAX_SIZE;
                accumulators[accumulator_addr + lrow * weight_col + rcol] += local_unified_buffer[laddr_offset] * weight_fifo.data[fifo_index];
                INFO("%d * %d = %d, sum[%d][%d] = %d\n", local_unified_buffer[laddr_offset], weight_fifo.data[fifo_index],
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

int vm_convolve(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, int channel, int kernel_size, uint16_t kernel_row, uint16_t kernel_col,
        int stride, int padding)
{
    if (weight_fifo.size < kernel_row * kernel_col * channel * kernel_size) {
        DBG("vm_convolve failed, weight fifo is too small\n");
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
                            int8_t input = local_unified_buffer[unified_buffer_addr + ub_offset];
                            int8_t k = weight_fifo.data[fifo_index];
                            accumulators[accumulator_addr + a_offset] += input * k;
                            INFO("input[%d,%d]:%d, k[%d][%d]:%d, a[%d]:%d\n", ir, ic, input, kr, kc, k, a_offset, accumulators[accumulator_addr + a_offset]);
                        }
                    }
                }
            }
        }
    }
    weight_fifo.read_index = (weight_fifo.read_index + kernel_row * kernel_col * channel * kernel_size) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= kernel_row * kernel_col * channel * kernel_size;
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

static int linear_max(uint16_t accumulator_addr, int len)
{
    int max = accumulators[accumulator_addr];
    int max_index = 0;
    for (int i = 0; i < len; i++) {
        INFO("%d ", accumulators[accumulator_addr + i]);
        if (max < accumulators[accumulator_addr + i]) {
            max = accumulators[accumulator_addr + i];
            max_index = i;
        }
    }
    for (int i = 0; i < len; i++) {
        if (i == max_index) {
            accumulators[accumulator_addr + i] = 1;
        } else {
            accumulators[accumulator_addr + i] = 0;
        }
    }
    return 0;
}

int vm_conv_bias(uint16_t accumulator_addr, uint16_t out_row, uint16_t out_col, int out_channel, int norm_range)
{
    if (weight_fifo.size < out_channel) {
        DBG("vm_conv_bias failed, weight fifo is too small\n");
        return -1;
    }
    for (int ci = 0; ci < out_channel; ci++) {
        for (int i = 0; i < out_row * out_col; i++) {
            int a_offset = ci * out_row * out_col + i;
            int w_offset = ci;
            int fifo_index = (w_offset + weight_fifo.read_index) % WEIGHT_FIFO_MAX_SIZE;
            accumulators[accumulator_addr + a_offset] += weight_fifo.data[fifo_index] * norm_range;
        }
    }
    weight_fifo.read_index = (weight_fifo.read_index + out_channel) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= out_channel;
    return 0;
}

int vm_matmul_bias(uint16_t accumulator_addr, uint16_t out_col, int norm_range)
{
    if (weight_fifo.size < out_col) {
        DBG("vm_matmul_bias failed, weight fifo is too small\n");
        return -1;
    }
    for (int i = 0; i < out_col; i++) {
        int fifo_index = (i + weight_fifo.read_index) % WEIGHT_FIFO_MAX_SIZE;
        accumulators[accumulator_addr + i] += weight_fifo.data[fifo_index] * norm_range;
    }
    weight_fifo.read_index = (weight_fifo.read_index + out_col) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= out_col;
    return 0;
}

int vm_max_pooling(uint32_t unified_buffer_addr, uint16_t accumulator_addr, uint16_t row, uint16_t col, int channel, int pool_size)
{
    int a_row = row / pool_size;
    int a_col = col / pool_size;
    for (int ci = 0; ci < channel; ci++) {
        for (int ir = 0; ir < (row - pool_size + 1); ir += pool_size) {
            for (int ic = 0; ic < (col - pool_size + 1); ic += pool_size) {
                int air = ir / pool_size;
                int aic = ic / pool_size;
                int a_offset = ci * a_row * a_col + air * a_col + aic;
                int8_t max = -127;
                for (int i = 0; i < pool_size; i++) {
                    for (int j = 0; j < pool_size; j++) {
                        int ub_offset = ci * row * col + (ir + i) * col + (ic + j);
                        max = max < local_unified_buffer[unified_buffer_addr + ub_offset] 
                                ? local_unified_buffer[unified_buffer_addr + ub_offset] : max;
                        INFO("[%d,%d,%d]:%d ", ci, ir + i, ic + j, local_unified_buffer[unified_buffer_addr + ub_offset]);
                    }
                }
                accumulators[accumulator_addr + a_offset] = max;
                INFO("[%d] max %d\n", a_offset, accumulators[accumulator_addr + a_offset]);
            }
        }
    }
}

static int get_abs_max(int32_t *data, int len)
{
    int max = 0;
    for (int i = 0; i < len; i ++) {
        if (abs(data[i]) > max) {
            max = abs(data[i]);
        }
    }
    return max;
}

static void debug(int32_t *in, int channel, int row, int col)
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

void vm_debug_acc(uint16_t accumulator_addr, int channel, int row, int col)
{
    debug(accumulators + accumulator_addr, 32, 30, 30);
}

static int normalize(int32_t *data, int len, int max, int range)
{
    for (int i = 0; i < len; i ++) {
        data[i] = data[i] * 1.0f / max * range;
    }
    return 0;
}

int vm_normalize(uint16_t accumulator_addr, int len, int range, int *max)
{
    int abs_max = get_abs_max(accumulators + accumulator_addr, len);
    INFO("norm max %d\n", max);
    normalize(accumulators + accumulator_addr, len, abs_max, range);
    *max = abs_max;
    return 0;
}

int vm_activate(act_type_enum_t type, uint16_t accumulator_addr, uint32_t unified_buffer_addr, int len)
{
    int ret;
    switch (type) {
    case ACT_TYPE_NONE:
        break;
    case ACT_TYPE_RELU:
        ret = relu(accumulator_addr, len); 
        break;
    case ACT_TYPE_MAX:
        ret = linear_max(accumulator_addr, len); 
        break;
    default:
        break;
    }
    for (int i = 0; i < len; i ++) {
        if (accumulators[accumulator_addr + i] >= 128 || accumulators[accumulator_addr + i] <= -128) {
            WARN("accumulators->local_unified_buffer overflow detected %d\n", accumulators[accumulator_addr + i]);
            if (accumulators[accumulator_addr + i] >= 128) {
                accumulators[accumulator_addr + i] = 127;
            } else if (accumulators[accumulator_addr + i] <= -128) {
                accumulators[accumulator_addr + i] = -127;
            }
        }
        local_unified_buffer[unified_buffer_addr + i] = accumulators[accumulator_addr + i];
        accumulators[accumulator_addr + i] = 0;
    }
}

int vm_write_host_memory(uint8_t *host_addr, uint32_t unified_buffer_addr, int len)
{
    memcpy(host_addr, local_unified_buffer + unified_buffer_addr, len);
    return 0;
}