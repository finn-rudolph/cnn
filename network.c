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

    for (size_t j = 0; j < z.layers[l - 1].fc.p; j++)
        z.layers[l - 1].fc.weight[j] = rand_double(a, b);
    for (size_t j = 0; j < z.layers[l - 1].fc.n; j++)
        z.layers[l - 1].fc.bias[j] = rand_double(a, b);

    return z;
}

void network_destroy(network *const z)
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

network network_read(char const *const fname)
{
    network z;
    FILE *file = fopen(fname, "r");
    if (!file)
    {
        perror("Error while reading network from file");
        return z;
    }

    fscanf(file, "%zu", &z.l);
    z.layers = malloc(z.l * sizeof(layer));

    for (size_t i = 0; i < z.l; i++)
    {
        layer *ly = z.layers + i;

        fscanf(file, "%hhu", &ly->conv.layer_type);
        switch (ly->conv.layer_type)
        {
        case TYPE_CONV:
        {
            fscanf(file, "%zu %zu %lg", &ly->conv.n, &ly->conv.m, &ly->conv.bias);

            ly->conv.kernel = malloc(SQUARE(ly->conv.m) * sizeof(double));
            for (size_t j = 0; j < SQUARE(ly->conv.m); j++)
                fscanf(file, "%lg", &ly->conv.kernel[j]);

            break;
        }
        case TYPE_FC:
        {
            fscanf(file, "%zu", &ly->fc.n);
            ly->fc.p = ly->fc.n * SQUARE((ly - 1)->conv.n);
            ly->fc.weight = malloc(ly->fc.p * sizeof(double));
            ly->fc.bias = malloc(ly->fc.n * sizeof(double));

            for (size_t j = 0; j < ly->fc.p; j++)
                fscanf(file, "%lg", &ly->fc.weight[j]);
            for (size_t j = 0; j < ly->fc.n; j++)
                fscanf(file, "%lg", &ly->fc.bias[j]);

            break;
        }
        }
    }

    return z;
}

void network_save(network const *const z, char const *const fname)
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
        layer *ly = z->layers + i;

        fprintf(file, "%hhu\n", ly->conv.layer_type);

        switch (ly->conv.layer_type)
        {
        case TYPE_CONV:
        {
            fprintf(file, "%zu %zu\n%lg\n", ly->conv.n, ly->conv.m, ly->conv.bias);

            for (size_t j = 0; j < SQUARE(ly->conv.m); j++)
                fprintf(file, "%lg ", ly->conv.kernel[j]);
            fputc('\n', file);

            break;
        }
        case TYPE_FC:
        {
            fprintf(file, "%zu\n", ly->fc.n);

            for (size_t j = 0; j < ly->fc.p; j++)
                fprintf(file, "%lg ", ly->fc.weight[j]);
            fputc('\n', file);

            for (size_t j = 0; j < ly->fc.n; j++)
                fprintf(file, "%lg ", ly->fc.bias[j]);
            fputc('\n', file);

            break;
        }
        }
    }

    fclose(file);
}