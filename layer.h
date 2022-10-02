#ifndef LAYER_H
#define LAYER_H 1

#include <stdint.h>
#include <stddef.h>

enum layer_type
{
    TYPE_CONV,
    TYPE_FC
};

typedef struct conv_layer conv_layer;
struct conv_layer
{
    uint8_t layer_type;
    size_t n, m; // layer and kernel size
    double bias;
    double *kernel;
};

typedef struct fc_layer fc_layer;
struct fc_layer
{
    uint8_t layer_type;
    size_t n, p;           // number of nodes, size of w and b
    double *weight, *bias; // weights and biases
};

typedef union layer layer;
union layer
{
    conv_layer conv;
    fc_layer fc;
};

static inline double rand_double(double a, double b)
{
    return a + (rand() / (RAND_MAX / (b - a)));
}

#endif