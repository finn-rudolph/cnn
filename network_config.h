#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H 1

// #define DEBUG_MODE 1
#define NDEBUG 1

#define ACTIVATION &fn_tanh
#define ACTIVATION_D &fn_tanh_d
#define OUT_ACTIVATION &fn_softmax
#define OUT_ACTIVATION_D 0

// #define CONV_FFT 1

#define PARAM_MIN -0.5 // Only for initializing the network.
#define PARAM_MAX 0.5

#define LEARN_RATE 0.1
#define REGULARIZATION_PARAM 0.02

#define BATCH_SIZE 250 // The batch size per thread.
#define NORMALIZATION_BATCH_SIZE 5000

#endif