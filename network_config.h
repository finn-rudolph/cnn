#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H 1

// #define DEBUG_MODE 1
#define NDEBUG 1

#define ACTIVATION &sigmoid
#define ACTIVATION_D &sigmoid_d
#define OUT_ACTIVATION &softmax
#define OUT_ACTIVATION_D 0

#define CONV_FFT 1

#define PARAM_MIN -1.0 // Only for initializing the network.
#define PARAM_MAX 1.0

#define LEARN_RATE 0.1
#define REGULARIZATION_PARAM 1.0

#define BATCH_SIZE 125 // The batch size per thread.
#define NORMALIZATION_BATCH_SIZE 1000

#endif