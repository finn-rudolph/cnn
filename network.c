#include <stdlib.h>
#include <stdio.h>

#include "network.h"
#include "util.h"

network network_init(size_t l, size_t m, double a, double b)
{
    network z;
    z.l = l;
    z.layers = malloc(l * sizeof(layer));

    for (size_t i = 0; i < l - 1; i++)
    {
        z.layers[i].conv = (conv_layer){
            .layer_type = TYPE_CONV,
            .n = 28,
            .m = m,
            .bias = rand_double(a, b),
            .kernel = malloc(m * m * sizeof(double))};

        for (size_t j = 0; j < m * m; j++)
            z.layers[i].conv.kernel[j] = rand_double(a, b);
    }

    z.layers[l - 1].fc = (fc_layer){
        .layer_type = TYPE_FC,
        .n = 10,
        .p = SQUARE(z.layers[l - 2].conv.n) * 10,
        .weight = malloc(SQUARE(z.layers[l - 2].conv.n) * 10 * sizeof(double)),
        .bias = malloc(10 * sizeof(double))};

    for (size_t i = 0; i < z.layers[l - 1].fc.p; i++)
        z.layers[l - 1].fc.weight[i] = rand_double(a, b);
    for (size_t i = 0; i < 10; i++)
        z.layers[l - 1].fc.bias[i] = rand_double(a, b);

    return z;
}

void network_destroy(network *z)
{
    for (size_t i = 0; i < z->l; i++)
    {
        switch (z->layers[i].conv.layer_type)
        {
        case TYPE_CONV:
        {
            free(z->layers[i].conv.kernel);
            break;
        }
        case TYPE_FC:
        {
            free(z->layers[i].fc.weight);
            free(z->layers[i].fc.bias);
            break;
        }
        }
    }

    free(z->layers);
}

void network_save(network *z, char *fname)
{
    FILE *file = fopen(fname, "w");
    if (!file)
    {
        perror("Error while saving network to disk");
        return;
    }

    fprintf(file, "%zu\n", z->l);
    for (size_t i = 0; i < z->l; i++)
    {
        fprintf(file, "%d\n", z->layers[i].conv.layer_type);

        switch (z->layers[i].conv.layer_type)
        {
        case TYPE_CONV:
        {
            fprintf(file, "%zu %zu\n%g\n", z->layers[i].conv.n,
                    z->layers[i].conv.m, z->layers[i].conv.bias);

            for (size_t j = 0; j < SQUARE(z->layers[i].conv.m); j++)
                fprintf(file, "%g ", z->layers[i].conv.kernel[j]);
            fputc('\n', file);

            break;
        }
        case TYPE_FC:
        {
            fprintf(file, "%zu\n", z->layers[i].fc.n);

            for (size_t j = 0; j < z->layers[i].fc.p; j++)
                fprintf(file, "%g ", z->layers[i].fc.weight[j]);
            for (size_t j = 0; j < z->layers[i].fc.n; j++)
                fprintf(file, "%g ", z->layers[i].fc.bias[j]);
            fputc('\n', file);

            break;
        }
        }
    }

    fclose(file);
}