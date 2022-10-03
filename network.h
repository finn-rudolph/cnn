#ifndef NETWORK_H
#define NETWORK_H 1

#include "layer.h"
#include "file_io.h"

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

double *network_feed_forward(network const *const net, example const *const x);

#endif