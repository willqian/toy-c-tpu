#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// TODO: Sparsity 矩阵运算
// TODO: maxtrix_multiply_unit会不会溢出
// TODO: 数据矩阵小是否分别放到不同的累加器进行累加

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
        printf("vm_read_weights failed, weight fifo is full\n");
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
    return 0;
}

int vm_maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint8_t input_row, uint8_t input_col, uint8_t weight_row, uint8_t weight_col)
{
    for (int lcol = 0; lcol < input_col; lcol++) {
        for (int lrow = 0; lrow < input_row; lrow++) {
            for (int rcol = 0; rcol < weight_col; rcol++) {
                int laddr_offset = lrow * input_col + lcol + unified_buffer_addr;
                int raddr_offset = lcol * weight_col + rcol;
                int fifo_index = raddr_offset + weight_fifo.read_index % WEIGHT_FIFO_MAX_SIZE;
                maxtrix_multiply_unit[lrow][rcol] += local_unified_buffer[laddr_offset] * weight_fifo.data[fifo_index];
            }
        }
        printf("tick\n");
    }
    for (int i = 0; i < input_row; i++) {
        for (int j = 0; j < weight_col; j++) {
            accumulators[accumulator_addr + i * weight_col + j] = maxtrix_multiply_unit[i][j];
        }
    }
    weight_fifo.read_index = (weight_fifo.read_index + weight_row * weight_col) % WEIGHT_FIFO_MAX_SIZE;
    weight_fifo.size -= weight_row * weight_col;
    memset(maxtrix_multiply_unit, 0, 256 * 256);
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

int vm_activate(uint16_t accumulator_addr, int len)
{
    int ret = relu(accumulator_addr, len);
    for (int i = 0; i < len; i ++) {
        local_unified_buffer[i] = accumulators[accumulator_addr + i];
    }
}

int vm_write_host_memory(uint8_t *host_addr, uint32_t unified_buffer_addr, int len)
{
    memcpy(host_addr, local_unified_buffer + unified_buffer_addr, len);
    return 0;
}