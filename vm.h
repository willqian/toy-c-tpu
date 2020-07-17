#ifndef VM_H
#define VM_H

#include <stdint.h>

typedef enum {
    ACT_TYPE_NONE = 0,
    ACT_TYPE_RELU,
    ACT_TYPE_MAX,
} act_type_enum_t;

int vm_init();

int vm_read_host_memory(uint32_t unified_buffer_addr, uint8_t *host_addr, int len);

int vm_read_weights(uint8_t *host_addr, int len);

int vm_maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, uint16_t weight_row, uint16_t weight_col);

int vm_convolve(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, int channel, int kernel_size, uint16_t kernel_row, uint16_t kernel_col,
        int stride, int padding);

int vm_conv_bias(uint16_t accumulator_addr, uint16_t out_row, uint16_t out_col, int out_channel, int norm_range);

int vm_matmul_bias(uint16_t accumulator_addr, uint16_t out_col, int norm_range);

int vm_max_pooling(uint32_t unified_buffer_addr, uint16_t accumulator_addr, uint16_t row, uint16_t col, int channel, int pool_size);

void vm_debug_acc(uint16_t accumulator_addr, int channel, int row, int col);

int vm_normalize(uint16_t accumulator_addr, int len, int range, int *max);

int vm_activate(act_type_enum_t type, uint16_t accumulator_addr, uint32_t unified_buffer_addr, int len);

int vm_write_host_memory(uint8_t *host_addr, uint32_t unified_buffer_addr, int len);

#endif /* VM_H */