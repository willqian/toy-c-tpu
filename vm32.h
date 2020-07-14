#ifndef VM_H
#define VM_H

#include <stdint.h>

typedef enum {
    ACT32_TYPE_NONE = 0,
    ACT32_TYPE_RELU,
    ACT32_TYPE_SOFTMAX,
} act32_type_enum_t;

int vm32_init();

int vm32_read_host_memory(uint32_t unified_buffer_addr, float *host_addr, int n);

int vm32_read_weights(float *host_addr, int n);

int vm32_maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, uint16_t weight_row, uint16_t weight_col);

int vm32_convolve(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint16_t input_row, uint16_t input_col, int channel, int kernel_size, uint16_t kernel_row, uint16_t kernel_col,
        int stride, int padding);

int vm32_activate(act32_type_enum_t type, uint16_t accumulator_addr, uint32_t unified_buffer_addr, int n);

int vm32_write_host_memory(float *host_addr, uint32_t unified_buffer_addr, int n);

#endif /* VM_H */