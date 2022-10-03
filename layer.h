#ifndef LAYER_H
#define LAYER_H 1

#include <stdint.h>
#include <stddef.h>

#include "file_io.h"

typedef enum layer_type layer_type;
enum layer_type
{
    LTYPE_INPUT,
    LTYPE_CONV,
    LTYPE_FC
};

typedef struct input_layer input_layer;
struct input_layer
{
    int ltype;
    size_t n;
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
    input_layer inp;
};

void input_layer_init(input_layer *const x, size_t n);

void conv_layer_init(conv_layer *const x, size_t n, size_t k);

void fc_layer_init(fc_layer *const x, size_t n, size_t m);

void conv_layer_destroy(conv_layer *const x);

void fc_layer_destroy(fc_layer *const);

void input_layer_pass(
    input_layer const *const x, example const *const e,
    double *const *const out, size_t padding);

void conv_layer_pass(
    conv_layer const *const x, double const *const *const in,
    double *const *const out);

void fc_layer_pass(
    fc_layer const *const x, double const *const in, double *const out);

void vectorize_matrix(
    size_t n, size_t m, double const *const *const matrix, double *const vector);

#endif