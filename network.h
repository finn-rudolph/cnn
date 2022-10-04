#ifndef NETWORK_H
#define NETWORK_H 1

#include "layer.h"

typedef struct network network;
struct network
{
    size_t l;
    layer *layers;
};

network network_init(size_t l, size_t m);

void network_destroy(network *const net);

network network_read(char const *const fname);

void network_save(network const *const net, char const *const fname);

double **network_pass_forward(
    network const *const net, size_t t, uint8_t *const *const images);

void network_save_results(
    char const *const result_fname, size_t t, double *const *const result);

void network_print_accuracy(
    size_t t, double *const *const results, uint8_t *const labels);

#endif