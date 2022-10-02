#ifndef NETWORK_H
#define NETWORK_H 1

#include "layer.h"

typedef struct network network;
struct network
{
    size_t l;
    layer *layers;
};

network network_init(size_t l, size_t m, double a, double b);

void network_destroy(network *const z);

network network_read(char const *const fname);

void network_save(network const *const z, char const *const fname);

#endif