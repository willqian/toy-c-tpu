#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// NOTE: MAC Multiply Accumulate

// The CISC MatrixMultiply instruction is 12 bytes, of which 3 are Unified Buffer address; 2 are
// accumulator address; 4 are length (sometimes 2 dimensions for convolutions); and the rest are opcode and flags
// The 4MiB represents 4096, 256-element, 32-bit accumulators.
// The matrix unit produces one 256-element partial sum per clock cycle.
// We picked 4096 by first noting that the operations per byte need to reach peak performance (roofline knee in Section 4) is ~1350,
// so we rounded that up to 2048 and then duplicated it so that the compiler could use double buffering
// while running at peak performance
static char *instructions[10] = {
    "Read_Host_Memory", // 3B-unified-buffer-addr 8B-host-addr 3B-size 14B
    "Read_Weights", // 2B-weights-addr 8B-addr 3B-size  13B
    "MatrixMultiplyOrConvolve", // 3B-unified-buffer-addr 2B-accumulator-addr 4B-2dim 1B-MM or Conv flag 12B
    "Activate", // 2B-accumulator-addr 2B-size 4B
    "Write_Host_Memory", // 8B-host-addr 3B-unified-buffer-addr 3B-size 14B
};

int main(int argc ,char *argv[])
{
    int8_t input[1][2] = {3, 4};
    int8_t weight[2][2] = {
        {1, 2},
        {-1, 0}
    };
    int8_t output[1][2] = {0};
    int ret;
    ret = vm_init();
    ret = vm_read_host_memory(0, (int8_t *)input, 2);
    ret = vm_read_weights((int8_t *)weight, 4);
    ret = vm_maxtrix_multiply(0, 0, 1, 2, 2, 2);
    ret = vm_activate(ACT_TYPE_RELU, 0, 2);
    ret = vm_write_host_memory((int8_t *)output, 0, 2);
    printf("%d %d\n", output[0][0], output[0][1]);
}