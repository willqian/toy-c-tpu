#ifndef VM_H
#define VM_H

#include <stdint.h>

int vm_init();

int vm_read_host_memory(uint32_t unified_buffer_addr, uint8_t *host_addr, int len);

int vm_read_weights(uint8_t *host_addr, int len);

int vm_maxtrix_multiply(uint32_t unified_buffer_addr, uint16_t accumulator_addr,
        uint8_t input_row, uint8_t input_col, uint8_t weight_row, uint8_t weight_col);

int vm_activate(uint16_t accumulator_addr, int len);

int vm_write_host_memory(uint8_t *host_addr, uint32_t unified_buffer_addr, int len);

#endif /* VM_H */