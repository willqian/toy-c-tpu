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

//#define VM_WARN
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