#ifndef CIFAR10_CNN_VARIABLES_H
#define CIFAR10_CNN_VARIABLES_H

extern float conv0_k[3][3][3][32];
extern float conv0_b[32];
extern float conv1_k[3][3][32][64];
extern float conv1_b[64];
extern float conv2_k[3][3][64][64];
extern float conv2_b[64];
extern float d0_k[1024][64];
extern float d0_b[64];
extern float d1_k[64][10];
extern float d1_b[10];

#endif /* CIFAR10_CNN_VARIABLES_H */