#ifndef LAYER_H
#define LAYER_H 1

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef enum layer_type layer_type;
enum layer_type
{
    LTYPE_INPUT,
    LTYPE_CONV,
    LTYPE_FC
};

typedef void (*activation_fn)(size_t n, double *const);

typedef struct input_layer input_layer;
struct input_layer
{
    uint8_t ltype;
    size_t n, padding;
};

void input_layer_init(input_layer *const x, size_t n, size_t padding);

void input_layer_pass(
    input_layer const *const x, uint8_t const *const image,
    double *const *const out);

void input_layer_read(input_layer *const x, FILE *const net_f);

void input_layer_save(input_layer const *const x, FILE *const net_f);

typedef struct conv_layer conv_layer;
struct conv_layer
{
    uint8_t ltype;
    size_t n, k; // layer and kernel size
    double bias;
    double **kernel;
    activation_fn f;
    double **out, **kernel_gradient, bias_gradient; // buffers for backpropagation
};

void conv_layer_init(conv_layer *const x, size_t n, size_t k);

void conv_layer_init_backprop(conv_layer *const x);

void conv_layer_destroy(conv_layer *const x);

void conv_layer_pass(
    conv_layer const *const x, double *const *const in,
    double *const *const out);

void conv_layer_backprop(
    conv_layer *const x, double *const *const in, double *const *const out);

void conv_layer_read(conv_layer *const x, FILE *const net_f);

void conv_layer_save(conv_layer const *const x, FILE *const net_f);

typedef struct fc_layer fc_layer;
struct fc_layer
{
    uint8_t ltype;
    size_t n, m; // number of nodes in this and the previous layer
    double **weight, *bias;
    activation_fn f;
    double *out, **weight_gradient, *bias_gradient;
};

void fc_layer_init(fc_layer *const x, size_t n, size_t m);

void fc_layer_init_backprop(fc_layer *const x);

void fc_layer_destroy(fc_layer *const);

void fc_layer_pass(fc_layer const *const x, double *const in, double *const out);

void fc_layer_backprop(fc_layer *const x, double *const in, double *const out);

void fc_layer_read(fc_layer *const x, FILE *const net_f);

void fc_layer_save(fc_layer const *const x, FILE *const net_f);

typedef union layer layer;
union layer
{
    input_layer input;
    conv_layer conv;
    fc_layer fc;
};

#endif