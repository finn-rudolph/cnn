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
    network const *const restrict net, size_t t, double **const restrict images);

void network_train(
    network const *const restrict net, size_t epochs, size_t t,
    double **const restrict images, uint8_t *const restrict labels,
    char const *const restrict fname);

void network_print_results(
    char const *const restrict result_fname, size_t t,
    double *const *const restrict result);

void network_print_accuracy(
    size_t t, double *const *const restrict results,
    uint8_t *const restrict labels);

network network_read(char const *const fname);

void network_print(
    network const *const restrict net, char const *const restrict fname);

#endif