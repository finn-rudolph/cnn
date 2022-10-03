#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "network.h"
#include "util.h"

network network_init(size_t l, size_t k, double a, double b)
{
    network net;
    net.l = l;
    net.layers = malloc((l + 1) * sizeof(layer));

    for (size_t i = 0; i < l - 1; i++)
    {
        layer *x = net.layers + i;

        x->conv = (conv_layer){
            .ltype = LTYPE_CONV,
            .n = 28,
            .k = k,
            .bias = rand_double(a, b),
            .kernel = malloc(k * sizeof(double *))};

        for (size_t j = 0; j < k; j++)
        {
            x->conv.kernel[j] = malloc(k * sizeof(double));
            for (size_t h = 0; h < k; h++)
                x->conv.kernel[j][h] = rand_double(a, b);
        }
    }

    net.layers[l - 1].fc = (fc_layer){
        .ltype = LTYPE_FC,
        .n = 10,
        .m = SQUARE(net.layers[l - 1].conv.n),
        .weight = malloc(10 * sizeof(double *)),
        .bias = malloc(10 * sizeof(double))};

    for (size_t j = 0; j < net.layers[l - 1].fc.n; j++)
    {
        net.layers[l - 1].fc.weight[j] = malloc(net.layers[l - 1].fc.m * sizeof(double));
        for (size_t k = 0; k < net.layers[l - 1].fc.m; k++)
            net.layers[l - 1].fc.weight[j][k] = rand_double(a, b);
    }
    for (size_t j = 0; j < net.layers[l - 1].fc.n; j++)
        net.layers[l - 1].fc.bias[j] = rand_double(a, b);

    return net;
}

void network_destroy(network *const net)
{
    for (size_t i = 1; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
            conv_layer_destroy(&x->conv);
            break;
        case LTYPE_FC:
            fc_layer_destroy(&x->fc);
            break;
        }
    }

    free(net->layers);
}

network network_read(char const *const fname)
{
    network net;
    FILE *net_f = fopen(fname, "r");
    if (!net_f)
    {
        perror("Error while reading network from file");
        return net;
    }

    fscanf(net_f, "%zu", &net.l);
    net.layers = malloc(net.l * sizeof(layer));

    for (size_t i = 0; i < net.l; i++)
    {
        layer *x = net.layers + i;

        fscanf(net_f, "%d", &x->conv.ltype);
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
        {
            fscanf(net_f, "%zu %zu %lg", &x->conv.n, &x->conv.k, &x->conv.bias);

            x->conv.kernel = malloc(x->conv.k * sizeof(double *));
            for (size_t j = 0; j < x->conv.k; j++)
            {
                x->conv.kernel[j] = malloc(x->conv.k * sizeof(double));
                for (size_t k = 0; k < x->conv.k; k++)
                    fscanf(net_f, "%lg", &x->conv.kernel[j][k]);
            }

            break;
        }
        case LTYPE_FC:
        {
            fscanf(net_f, "%zu", &x->fc.n);
            x->fc.m = SQUARE((x - 1)->conv.n);
            x->fc.weight = malloc(x->fc.n * sizeof(double *));
            x->fc.bias = malloc(x->fc.n * sizeof(double));

            for (size_t j = 0; j < x->fc.n; j++)
            {
                x->fc.weight[j] = malloc(x->fc.m * sizeof(double));
                for (size_t k = 0; k < x->fc.m; k++)
                    fscanf(net_f, "%lg", &x->fc.weight[j][k]);
            }
            for (size_t j = 0; j < x->fc.n; j++)
                fscanf(net_f, "%lg", &x->fc.bias[j]);

            break;
        }
        }
    }

    return net;
}

void network_save(network const *const net, char const *const fname)
{
    FILE *net_f = fopen(fname, "w");
    if (!net_f)
    {
        perror("Error while saving network to disk");
        return;
    }

    fprintf(net_f, "%zu\n", net->l);
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;

        fprintf(net_f, "%d\n", x->conv.ltype);

        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
        {
            fprintf(net_f, "%zu %zu\n%lg\n", x->conv.n, x->conv.k, x->conv.bias);

            for (size_t j = 0; j < x->conv.k; j++)
                for (size_t k = 0; k < x->conv.k; k++)
                    fprintf(net_f, "%lg ", x->conv.kernel[j][k]);
            fputc('\n', net_f);

            break;
        }
        case LTYPE_FC:
        {
            fprintf(net_f, "%zu\n", x->fc.n);

            for (size_t j = 0; j < x->fc.n; j++)
                for (size_t k = 0; k < x->fc.m; k++)
                    fprintf(net_f, "%lg ", x->fc.weight[j][k]);
            fputc('\n', net_f);

            for (size_t j = 0; j < x->fc.n; j++)
                fprintf(net_f, "%lg ", x->fc.bias[j]);
            fputc('\n', net_f);

            break;
        }
        }
    }

    fclose(net_f);
}

double *network_feed_forward(network const *const net, example const *const e)
{
    // Two grid containers for convolutional layers, two linear containers for
    // fully connected layers. Serve as input / output buffer.
    double **u = malloc(28 * sizeof(double *)),
           **v = malloc(28 * sizeof(double *)),
           *p = malloc(SQUARE(28) * sizeof(double)),
           *q = malloc(SQUARE(28) * sizeof(double));

    for (size_t i = 0; i < 28; i++)
    {
        u[i] = malloc(28 * sizeof(double));
        v[i] = malloc(28 * sizeof(double));
    }

    input_layer_pass(&net->layers[0].inp, e, v, net->layers[1].conv.k);

    for (size_t i = 1; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
        {
            conv_layer_pass(&x->conv, u, v);
            swap(&u, &v);
            break;
        }
        case LTYPE_FC:
        {
            if ((x - 1)->conv.ltype != LTYPE_FC)
                // After the first fully connected layer, only such layers come,
                // therefore switch to vectors instead of matrices.
                vectorize_matrix(x->fc.n, x->fc.m, u, p);
            fc_layer_pass(&net->layers[i].fc, p, q);
            swap(&p, &q);
        }
        }
    }

    double *result = malloc(10 * sizeof(double));
    memcpy(result, p, 10 * sizeof(double));

    for (size_t i = 0; i < 28; i++)
    {
        free(v[i]);
        free(u[i]);
    }
    free(v);
    free(u);
    free(p);
    free(q);
    return result;
}