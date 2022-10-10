#ifndef DEF_H
#define DEF_H 1

// #define DEBUG_MODE 1
#define NDEBUG 1

#define ACTIVATION &vsigmoid
#define ACTIVATION_D &sigmoid_d
#define OUT_ACTIVATION &softmax
#define OUT_ACTIVATION_D 0

#define pad(n, k, out) pad_zero(n, k, out)

#define PARAM_MIN -1.0 // Only for initializing the network.
#define PARAM_MAX 1.0

#define VALUE_MAX 1e9 // Maximum possible activation of a neuron.

#define LEARN_RATE -0.1

#define BATCH_SIZE 125
#define NORMALIZATION_BATCH_SIZE 100

#endif