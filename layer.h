#ifndef LAYER_H
#define LAYER_H 1

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

#include "util.h"

typedef enum layer_type layer_type;
enum layer_type
{
    LTYPE_INPUT,
    LTYPE_CONV,
    LTYPE_FC,
    LTYPE_FLAT
};

typedef struct InputLayer InputLayer;
struct InputLayer
{
    uint8_t ltype;
    size_t n;
    double **out;
};

void input_layer_init(InputLayer *const x, size_t n);

void input_layer_init_backprop(InputLayer *const x);

void input_layer_free(InputLayer *const x);

void input_layer_pass(
    InputLayer const *const x, double *const image, double *const *const out,
    bool store_intermed);

void input_layer_read(InputLayer *const x, FILE *const stream);

void input_layer_print(InputLayer const *const x, FILE *const stream);

typedef struct ConvLayer ConvLayer;
struct ConvLayer
{
    uint8_t ltype;
    size_t n, k;
    double bias, **kernel;
    vactivation_fn f;
    activation_fn fd;
    double **in, **out; // buffers for backpropagation
    double **kernel_gradient, bias_gradient;
};

void conv_layer_init(ConvLayer *const x, size_t n, size_t k);

void conv_layer_init_backprop(ConvLayer *const x);

void conv_layer_reset_gradient(ConvLayer *const x);

void conv_layer_free(ConvLayer *const x);

void conv_layer_pass(
    ConvLayer const *const x, double *const *const in,
    double *const *const out, bool store_result);

void conv_layer_update_gradient(
    ConvLayer *const x, double *const *const prev_out,
    double *const *const delta);

void conv_layer_backprop(
    ConvLayer const *const x, double *const *const prev_in,
    activation_fn prev_fd, double *const *const delta,
    double *const *const ndelta);

void conv_layer_descend(ConvLayer *const x, size_t t);

void conv_layer_read(ConvLayer *const x, FILE *const stream);

void conv_layer_print(ConvLayer const *const x, FILE *const stream);

typedef struct FcLayer FcLayer;
struct FcLayer
{
    uint8_t ltype;
    size_t n, m;
    double **weight, *bias;
    vactivation_fn f;
    activation_fn fd;
    double *in, *out;
    double **weight_gradient, *bias_gradient;
};

void fc_layer_init(FcLayer *const x, size_t n, size_t m);

void fc_layer_init_backprop(FcLayer *const x);

void fc_layer_reset_gradient(FcLayer *const x);

void fc_layer_free(FcLayer *const x);

void fc_layer_pass(
    FcLayer const *const x, double *const in, double *const out,
    bool store_result);

void fc_layer_update_gradient(
    FcLayer *const x, double *const prev_out, double *const delta);

void fc_layer_backprop(
    FcLayer const *const x, double *const prev_in, activation_fn prev_fd,
    double *const delta, double *const ndelta);

void fc_layer_descend(FcLayer *const x, size_t t);

void fc_layer_read(FcLayer *const x, FILE *const stream);

void fc_layer_print(FcLayer const *const x, FILE *const stream);

typedef struct FlatLayer FlatLayer;
struct FlatLayer
{
    uint8_t ltype;
    size_t n;
    double *in, *out; // the previous layer's input / output, but flattened
};

void flat_layer_init(FlatLayer *const x, size_t n);

void flat_layer_init_backprop(FlatLayer *const x);

void flat_layer_free(FlatLayer *const x);

void flat_layer_pass(
    FlatLayer const *const x, double *const *const in, double *const out);

void flat_layer_backprop(
    FlatLayer const *const x, double *const delta, double *const *const ndelta);

void flat_layer_read(FlatLayer *const x, FILE *const stream);

void flat_layer_print(
    FlatLayer const *const x, FILE *const stream);

typedef union Layer Layer;
union Layer
{
    InputLayer input;
    ConvLayer conv;
    FcLayer fc;
    FlatLayer flat;
};

#endif