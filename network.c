#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "network.h"
#include "util.h"

#define PARAM_MIN 0.0
#define PARAM_MAX 10.0

network network_init(size_t num_layers, size_t kernel_size)
{
    network net;
    net.l = num_layers;
    net.layers = malloc((num_layers + 1) * sizeof(layer));

    input_layer_init(&net.layers[0].input, 28);

    for (size_t i = 1; i < num_layers; i++)
    {
        layer *x = net.layers + i;
        conv_layer_init(&x->conv, 28, kernel_size);
        x->conv.bias = rand_double(PARAM_MIN, PARAM_MAX);

        for (size_t j = 0; j < kernel_size; j++)
        {
            for (size_t k = 0; k < kernel_size; k++)
            {
                x->conv.kernel[j][k] = rand_double(PARAM_MAX, PARAM_MAX);
            }
        }
    }

    fc_layer_init(&net.layers[net.l - 1].fc, 10,
                  SQUARE(net.layers[net.l - 2].conv.n));

    for (size_t j = 0; j < net.layers[net.l - 1].fc.n; j++)
    {
        for (size_t k = 0; k < net.layers[net.l - 1].fc.m; k++)
        {
            net.layers[net.l - 1].fc.weight[j][k] =
                rand_double(PARAM_MIN, PARAM_MAX);
        }
    }

    for (size_t j = 0; j < net.layers[net.l - 1].fc.n; j++)
    {
        net.layers[net.l - 1].fc.bias[j] = rand_double(PARAM_MIN, PARAM_MAX);
    }

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
            conv_layer_init(&x->conv, x->conv.n, x->conv.k);

            for (size_t j = 0; j < x->conv.k; j++)
            {
                for (size_t k = 0; k < x->conv.k; k++)
                {
                    fscanf(net_f, "%lg", &x->conv.kernel[j][k]);
                }
            }

            break;
        }
        case LTYPE_FC:
        {
            fscanf(net_f, "%zu %zu", &x->fc.n, &x->fc.m);
            fc_layer_init(&x->fc, x->fc.n, x->fc.m);

            for (size_t j = 0; j < x->fc.n; j++)
            {
                for (size_t k = 0; k < x->fc.m; k++)
                {
                    fscanf(net_f, "%lg", &x->fc.weight[j][k]);
                }
            }

            for (size_t j = 0; j < x->fc.n; j++)
            {
                fscanf(net_f, "%lg", &x->fc.bias[j]);
            }

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
            {
                for (size_t k = 0; k < x->conv.k; k++)
                {
                    fprintf(net_f, "%lg ", x->conv.kernel[j][k]);
                }
            }
            fputc('\n', net_f);

            break;
        }
        case LTYPE_FC:
        {
            fprintf(net_f, "%zu %zu\n", x->fc.n, x->fc.m);

            for (size_t j = 0; j < x->fc.n; j++)
            {
                for (size_t k = 0; k < x->fc.m; k++)
                {
                    fprintf(net_f, "%lg ", x->fc.weight[j][k]);
                }
            }
            fputc('\n', net_f);

            for (size_t j = 0; j < x->fc.n; j++)
            {
                fprintf(net_f, "%lg ", x->fc.bias[j]);
            }
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

    input_layer_pass(&net->layers[0].input, e, v, net->layers[1].conv.k);

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
            {
                // After the first fully connected layer, only such layers come,
                // therefore switch to vectors instead of matrices.
                vectorize_matrix(x->fc.n, x->fc.m, u, p);
            }
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