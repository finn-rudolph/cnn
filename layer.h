#ifndef LAYER_H
#define LAYER_H 1

#include <stdint.h>
#include <stddef.h>

typedef enum layer_type layer_type;
enum layer_type
{
    LTYPE_CONV,
    LTYPE_FC
};

typedef struct conv_layer conv_layer;
struct conv_layer
{
    int ltype;
    size_t n, k; // layer and kernel size
    double bias;
    double **kernel;
};

typedef struct fc_layer fc_layer;
struct fc_layer
{
    int ltype;
    size_t n, m;            // number of nodes in this and the previous layer
    double **weight, *bias; // weights and biases
};

typedef union layer layer;
union layer
{
    conv_layer conv;
    fc_layer fc;
};

#endif