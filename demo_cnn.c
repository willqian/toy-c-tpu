#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc ,char *argv[])
{
    int8_t input[5][5] = {
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {1, 0, 1, 0, 1},
        {2, 3, 4, 1, 1},
        {3, 1, 4, 1, 5},
    };
    int8_t input2[2][5][5] = {{
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {1, 0, 1, 0, 1},
        {2, 3, 4, 1, 1},
        {3, 1, 4, 1, 5},
    }, {
        {3, 2, 1, 0, 4},
        {2, 1, 3, 1, 0},
        {1, 2, 0, 1, 2},
        {3, 2, 4, 3, 1},
        {1, 1, 0, 0, 1},
    }};
    int8_t input7[7][7] = {
        {1, 2, 3, 4, 5, 6, 7},
        {5, 4, 3, 2, 1, 0, 1},
        {1, 0, 1, 0, 1, 0, 1},
        {2, 3, 4, 1, 1, 2, 3},
        {3, 1, 4, 1, 5, 1, 6},
        {7, 5, 2, 9, 1, 8, 5},
        {6, 3, 1, 2, 3, 7, 9}
    };
    int8_t weight[3][3] = {
        {0, 1, 1},
        {1, 1, 1},
        {1, 1, 0},
    };
    int8_t weight2[2][3][3] = {{
        {0, 1, 1},
        {1, 1, 1},
        {1, 1, 0},
    }, {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    }};
    int8_t weight22[2][2][3][3] = { // kernel-channel-row-col
    {
        {
            {0, 1, 1},
            {1, 1, 1},
            {1, 1, 0},
        }, {
            {1, 0, 0},
            {0, 1, 0},
            {0, 1, 1},
        }
    }, {
        {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
        }, {
            {0, 1, 1},
            {1, 1, 1},
            {1, 0, 0},
        }
    }};
    int8_t weight5[5][5] = {
        {0, 1, 1, 0, 1},
        {1, 1, 1, 1, 0},
        {1, 1, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {1, 1, 1, 1, 1}
    };
    int8_t output[3][3] = {0};
    int8_t output_with_padding1[5][5] = {0};
    int8_t output_with_stride2[2][2] = {0};
    int8_t output_with_padding1_stride2[3][3] = {0};
    int8_t output_with_padding1_stride2_k2[2][3][3] = {0};
    int8_t output_with_padding1_stride2_k2_channel2[2][3][3] = {0};
    int8_t output_with_padding2_stride2_w5[4][4] = {0};

    printf("CNN input:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j ++) {
            printf("%3d ", input[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("CNN kernel:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            printf("%3d ", weight[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm_init();
    vm_read_host_memory(0, (int8_t *)input, 25);
    vm_read_weights((int8_t *)weight, 9);
    vm_convolve(0, 0, 5, 5, 1, 1, 3, 3, 1, 0);
    vm_activate(ACT_TYPE_NONE, 0, 0, 9);
    vm_write_host_memory((int8_t *)output, 0, 9);
    printf("CNN normal:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            printf("%3d ", output[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm_read_host_memory(0, (int8_t *)input, 25);
    vm_read_weights((int8_t *)weight, 9);
    vm_convolve(0, 0, 5, 5, 1, 1, 3, 3, 1, 1);
    vm_activate(ACT_TYPE_NONE, 0, 0, 25);
    vm_write_host_memory((int8_t *)output_with_padding1, 0, 25);
    printf("CNN with padding 1:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j ++) {
            printf("%3d ", output_with_padding1[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm_read_host_memory(0, (int8_t *)input, 25);
    vm_read_weights((int8_t *)weight, 9);
    vm_convolve(0, 0, 5, 5, 1, 1, 3, 3, 2, 0);
    vm_activate(ACT_TYPE_NONE, 0, 0, 4);
    vm_write_host_memory((int8_t *)output_with_stride2, 0, 4);
    printf("CNN with stride 2:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j ++) {
            printf("%3d ", output_with_stride2[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm_read_host_memory(0, (int8_t *)input, 25);
    vm_read_weights((int8_t *)weight, 9);
    vm_convolve(0, 0, 5, 5, 1, 1, 3, 3, 2, 1);
    vm_activate(ACT_TYPE_NONE, 0, 0, 9);
    vm_write_host_memory((int8_t *)output_with_padding1_stride2, 0, 9);
    printf("CNN with padding 1 and stride 2:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            printf("%3d ", output_with_padding1_stride2[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm_read_host_memory(0, (int8_t *)input, 25);
    vm_read_weights((int8_t *)weight2, 18);
    vm_convolve(0, 0, 5, 5, 1, 2, 3, 3, 2, 1);
    vm_activate(ACT_TYPE_NONE, 0, 0, 18);
    vm_write_host_memory((int8_t *)output_with_padding1_stride2_k2, 0, 18);
    printf("CNN with padding 1 and stride 2 and kernel 2:\n");
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j ++) {
                printf("%3d ", output_with_padding1_stride2_k2[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    vm_read_host_memory(0, (int8_t *)input2, 50);
    vm_read_weights((int8_t *)weight22, 36);
    vm_convolve(0, 0, 5, 5, 2, 2, 3, 3, 2, 1);
    vm_activate(ACT_TYPE_NONE, 0, 0, 18);
    vm_write_host_memory((int8_t *)output_with_padding1_stride2_k2_channel2, 0, 18);
    printf("CNN with padding 1 and stride 2 and kernel 2 and channel 2:\n");
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j ++) {
                printf("%3d ", output_with_padding1_stride2_k2_channel2[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    vm_read_host_memory(0, (int8_t *)input7, 49);
    vm_read_weights((int8_t *)weight5, 25);
    vm_convolve(0, 0, 7, 7, 1, 1, 5, 5, 2, 2);
    vm_activate(ACT_TYPE_NONE, 0, 0, 16);
    vm_write_host_memory((int8_t *)output_with_padding2_stride2_w5, 0, 16);
    printf("CNN with input 7x7 kernel 5x5, padding 2, and stride 2:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j ++) {
            printf("%3d ", output_with_padding2_stride2_w5[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}