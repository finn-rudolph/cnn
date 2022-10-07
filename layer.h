#ifndef LAYER_H
#define LAYER_H 1

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

typedef enum layer_type layer_type;
enum layer_type
{
    LTYPE_INPUT,
    LTYPE_CONV,
    LTYPE_FC,
    LTYPE_FLAT
};

typedef void (*activation_fn)(size_t n, double *const);

typedef struct input_layer input_layer;
struct input_layer
{
    uint8_t ltype;
    size_t n, padding;
    double **out;
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
    size_t n, k;
    double bias, **kernel;
    activation_fn f, fd;
    double **in, **out; // buffers for backpropagation
    double **kernel_gradient, bias_gradient;
};

void conv_layer_init(conv_layer *const x, size_t n, size_t k);

void conv_layer_init_backprop(conv_layer *const x);

void conv_layer_destroy(conv_layer *const x);

void conv_layer_pass(
    conv_layer const *const x, double *const *const in,
    double *const *const out, bool store_intermed);

void conv_layer_backprop(
    conv_layer const *const x, double *const *const delta,
    double *const *const ndelta);

void conv_layer_avg_gradient(conv_layer *const x, size_t t);

void conv_layer_read(conv_layer *const x, FILE *const net_f);

void conv_layer_save(conv_layer const *const x, FILE *const net_f);

typedef struct fc_layer fc_layer;
struct fc_layer
{
    uint8_t ltype;
    size_t n, m;
    double **weight, *bias;
    activation_fn f, fd;
    double *in, *out;
    double **weight_gradient, *bias_gradient;
};

void fc_layer_init(fc_layer *const x, size_t n, size_t m);

void fc_layer_init_backprop(fc_layer *const x);

void fc_layer_destroy(fc_layer *const);

void fc_layer_pass(
    fc_layer const *const x, double *const in, double *const out,
    bool store_intermed);

void fc_layer_backprop(
    fc_layer const *const x, double *const prev_in, double *const prev_out,
    double *const delta, double *const ndelta);

void fc_layer_avg_gradient(fc_layer const *const x, size_t t);

void fc_layer_read(fc_layer *const x, FILE *const net_f);

void fc_layer_save(fc_layer const *const x, FILE *const net_f);

typedef struct flat_layer flat_layer;
struct flat_layer
{
    uint8_t ltype;
    size_t n, padding;
    double *in, *out; // the previous layer's input / output, but flattened
};

void flat_layer_init(flat_layer *const x, size_t n, size_t padding);

void flat_layer_init_backprop(flat_layer *const x);

void flat_layer_destroy(flat_layer *const x);

void flat_layer_pass(
    flat_layer const *const x, double *const *const in, double *const out);

void flat_layer_backprop(flat_layer const *const x);

void flat_layer_read(flat_layer *const x, FILE *const net_f);

void flat_layer_save(flat_layer const *const x, FILE *net_f);

typedef union layer layer;
union layer
{
    input_layer input;
    conv_layer conv;
    fc_layer fc;
    flat_layer flat;
};

#endif