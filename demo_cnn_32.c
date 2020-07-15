#include "vm32.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc ,char *argv[])
{
    float input[5][5] = {
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {1, 0, 1, 0, 1},
        {2, 3, 4, 1, 1},
        {3, 1, 4, 1, 5},
    };
    float input2[2][5][5] = {{
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
    float input7[7][7] = {
        {1, 2, 3, 4, 5, 6, 7},
        {5, 4, 3, 2, 1, 0, 1},
        {1, 0, 1, 0, 1, 0, 1},
        {2, 3, 4, 1, 1, 2, 3},
        {3, 1, 4, 1, 5, 1, 6},
        {7, 5, 2, 9, 1, 8, 5},
        {6, 3, 1, 2, 3, 7, 9}
    };
    float weight[3][3] = {
        {0, 1, 1},
        {1, 1, 1},
        {1, 1, 0},
    };
    float weight2[2][3][3] = {{
        {0, 1, 1},
        {1, 1, 1},
        {1, 1, 0},
    }, {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    }};
    float weight22[2][2][3][3] = { // kernel-channel-row-col
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
    float weight5[5][5] = {
        {0, 1, 1, 0, 1},
        {1, 1, 1, 1, 0},
        {1, 1, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {1, 1, 1, 1, 1}
    };
    float output[3][3] = {0};
    float output_with_padding1[5][5] = {0};
    float output_with_stride2[2][2] = {0};
    float output_with_padding1_stride2[3][3] = {0};
    float output_with_padding1_stride2_k2[2][3][3] = {0};
    float output_with_padding1_stride2_k2_channel2[2][3][3] = {0};
    float output_with_padding2_stride2_w5[4][4] = {0};
    printf("CNN input:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j ++) {
            printf("%f ", input[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("CNN kernel:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            printf("%f ", weight[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm32_init();
    vm32_read_host_memory(0, (float *)input, 25);
    vm32_read_weights((float *)weight, 9);
    vm32_convolve(0, 0, 5, 5, 1, 1, 3, 3, 1, 0);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 9);
    vm32_write_host_memory((float *)output, 0, 9);
    printf("CNN normal:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            printf("%f ", output[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm32_read_host_memory(0, (float *)input, 25);
    vm32_read_weights((float *)weight, 9);
    vm32_convolve(0, 0, 5, 5, 1, 1, 3, 3, 1, 1);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 25);
    vm32_write_host_memory((float *)output_with_padding1, 0, 25);
    printf("CNN with padding 1:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j ++) {
            printf("%f ", output_with_padding1[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm32_read_host_memory(0, (float *)input, 25);
    vm32_read_weights((float *)weight, 9);
    vm32_convolve(0, 0, 5, 5, 1, 1, 3, 3, 2, 0);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 4);
    vm32_write_host_memory((float *)output_with_stride2, 0, 4);
    printf("CNN with stride 2:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j ++) {
            printf("%f ", output_with_stride2[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm32_read_host_memory(0, (float *)input, 25);
    vm32_read_weights((float *)weight, 9);
    vm32_convolve(0, 0, 5, 5, 1, 1, 3, 3, 2, 1);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 9);
    vm32_write_host_memory((float *)output_with_padding1_stride2, 0, 9);
    printf("CNN with padding 1 and stride 2:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j ++) {
            printf("%f ", output_with_padding1_stride2[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    vm32_read_host_memory(0, (float *)input, 25);
    vm32_read_weights((float *)weight2, 18);
    vm32_convolve(0, 0, 5, 5, 1, 2, 3, 3, 2, 1);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 18);
    vm32_write_host_memory((float *)output_with_padding1_stride2_k2, 0, 18);
    printf("CNN with padding 1 and stride 2 and kernel 2:\n");
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j ++) {
                printf("%f ", output_with_padding1_stride2_k2[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    vm32_read_host_memory(0, (float *)input2, 50);
    vm32_read_weights((float *)weight22, 36);
    vm32_convolve(0, 0, 5, 5, 2, 2, 3, 3, 2, 1);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 18);
    vm32_write_host_memory((float *)output_with_padding1_stride2_k2_channel2, 0, 18);
    printf("CNN with padding 1 and stride 2 and kernel 2 and channel 2:\n");
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j ++) {
                printf("%f ", output_with_padding1_stride2_k2_channel2[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    vm32_read_host_memory(0, (float *)input7, 49);
    vm32_read_weights((float *)weight5, 25);
    vm32_convolve(0, 0, 7, 7, 1, 1, 5, 5, 2, 2);
    vm32_activate(ACT32_TYPE_NONE, 0, 0, 16);
    vm32_write_host_memory((float *)output_with_padding2_stride2_w5, 0, 16);
    printf("CNN with input 7x7 kernel 5x5, padding 2, and stride 2:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j ++) {
            printf("%f ", output_with_padding2_stride2_w5[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}