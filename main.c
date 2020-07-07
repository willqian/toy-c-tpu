#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// NOTE: MAC Multiply Accumulate

// The CISC MatrixMultiply instruction is 12 bytes, of which 3 are Unified Buffer address; 2 are
// accumulator address; 4 are length (sometimes 2 dimensions for convolutions); and the rest are opcode and flags
// The 4MiB represents 4096, 256-element, 32-bit accumulators.
// The matrix unit produces one 256-element partial sum per clock cycle.
static char *instructions[10] = {
    "Read_Host_Memory", // 3B-unified-buffer-addr 8B-host-addr 3B-size 14B
    "Read_Weights", // 2B-weights-addr 8B-addr 3B-size  13B
    "MatrixMultiplyOrConvolve", // 3B-unified-buffer-addr 2B-accumulator-addr 4B-2dim 1B-MM or Conv flag 12B
    "Activate", // 2B-accumulator-addr 2B-size 4B
    "Write_Host_Memory", // 8B-host-addr 3B-unified-buffer-addr 3B-size 14B
};

// TODO: Sparsity 矩阵运算
// TODO: maxtrix_multiply_unit会不会溢出
// TODO: 数据矩阵小是否分别放到不同的累加器进行累加

static int8_t weight_fifo[256 * 256] = {0};
static int8_t maxtrix_multiply_unit[256][256] = {0}; // 64K MAC
static int8_t local_unified_buffer[96 * 1024 * 256] = {0}; // 24MB
static int32_t accumulators[4 * 1024 * 256] = {0}; // 4MB

static int read_host_memory(uint32_t unified_buffer_addr, uint8_t *host_addr, int len)
{
    memcpy(local_unified_buffer + unified_buffer_addr, host_addr, len);
    return 0;
}

static int read_weights(uint16_t weights_addr, uint8_t *host_addr, int len)
{
    memcpy(weight_fifo, host_addr, len);
    return 0;
}

static int maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint8_t input_row, uint8_t input_col, uint8_t weight_row, uint8_t weight_col)
{
    for (int lcol = 0; lcol < input_col; lcol++) {
        for (int lrow = 0; lrow < input_row; lrow++) {
            for (int rcol = 0; rcol < weight_col; rcol++) {
                int laddr_offset = lrow * input_col + lcol + unified_buffer_addr;
                int raddr_offset = lcol * weight_col + rcol;
                maxtrix_multiply_unit[lrow][rcol] += local_unified_buffer[laddr_offset] * weight_fifo[raddr_offset];
            }
        }
        printf("tick\n");
    }
    for (int i = 0; i < input_row; i++) {
        for (int j = 0; j < weight_col; j++) {
            accumulators[accumulator_addr + i * weight_col + j] = maxtrix_multiply_unit[i][j];
        }
    }
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

static int activate(uint16_t accumulator_addr, int len)
{
    int ret = relu(accumulator_addr, len);
    for (int i = 0; i < len; i ++) {
        local_unified_buffer[i] = accumulators[accumulator_addr + i];
    }
}

static int write_host_memory(uint8_t *host_addr, uint32_t unified_buffer_addr, int len)
{
    memcpy(host_addr, local_unified_buffer + unified_buffer_addr, len);
    return 0;
}

int main(int argc ,char *argv[])
{
    int8_t input[1][2] = {3, 4};
    int8_t weight[2][2] = {
        {1, 2},
        {-1, 0}
    };
    int8_t output[1][2] = {0};
    int ret;
    ret = read_host_memory(0, input, 2);
    ret = read_weights(0, weight, 4);
    ret = maxtrix_multiply(0, 0, 1, 2, 2, 2);
    ret = activate(0, 2);
    ret = write_host_memory(output, 0, 2);
    printf("%d %d\n", output[0][0], output[0][1]);
}