#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H 1

#define ACTIVATION &fn_tanh
#define ACTIVATION_D &fn_tanh_d
#define OUT_ACTIVATION &fn_softmax // For the output layer.
#define OUT_ACTIVATION_D 0

// Whether to do convolutions with the FFT or the naive way. The naive way is
// faster due to the small image size of 28 x 28.
// #define CONV_FFT 1

#define PARAM_MIN -0.5 // Range for randomly initializing the network.
#define PARAM_MAX 0.5

#define LEARN_RATE 0.1
#define REGULARIZATION_PARAM 0.02

#define BATCH_SIZE 250 // The batch size per thread.
#define NORMALIZATION_BATCH_SIZE 5000

#endif