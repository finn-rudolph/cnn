#ifndef NETWORK_H
#define NETWORK_H 1

#include <stdbool.h>

#include "layer.h"
#include "def.h"

typedef struct network network;
struct network
{
    size_t l;
    layer *layers;
};

network network_init(
    size_t num_conv, size_t num_fc, size_t kernel_size, size_t fc_size);

void network_free(network *const net);

double **network_pass_forward(
    network const *const net, size_t t, double **const images);

void network_train(
    network const *const net, size_t epochs, size_t t, double **const images,
    uint8_t *const labels, char const *const fname);

void network_print_results(
    char const *const result_fname, size_t t,
    double *const *const result);

void network_print_accuracy(
    size_t t, double *const *const results,
    uint8_t *const labels);

network network_read(char const *const fname);

void network_print(
    network const *const net, char const *const fname);

#endif