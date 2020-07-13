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
                int fifo_index = raddr_offset + weight_fifo.read_index % WEIGHT_FIFO_MAX_SIZE;
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