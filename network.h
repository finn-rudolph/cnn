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

void network_destroy(network *z);

network network_read(char *fname);

void network_save(network *z, char *fname);

#endif