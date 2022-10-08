#ifndef DEF_H
#define DEF_H 1

// #define DEBUG_MODE 1

#define ACTIVATION &vrelu
#define ACTIVATION_D &vrelu_d
#define OUT_ACTIVATION &softmax
#define OUT_ACTIVATION_D 0

#define pad(n, k, out) pad_zero(n, k, out)

#define PARAM_MIN -3.5
#define PARAM_MAX 1.0

#define LEARN_RATE 0.01

#define NORMALIZATION_BATCH_SIZE 100

#endif