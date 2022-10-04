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
                x->conv.kernel[j][k] = rand_double(PARAM_MIN, PARAM_MAX);
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
        case LTYPE_INPUT:
            input_layer_read(&x->input, net_f);
            break;

        case LTYPE_CONV:
            conv_layer_read(&x->conv, net_f);
            break;

        case LTYPE_FC:
            fc_layer_read(&x->fc, net_f);
            break;
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
        case LTYPE_INPUT:
            input_layer_save(&x->input, net_f);
            break;

        case LTYPE_CONV:
            conv_layer_save(&x->conv, net_f);
            break;

        case LTYPE_FC:
            fc_layer_save(&x->fc, net_f);
            break;
        }
    }

    fclose(net_f);
}

double **network_pass_forward(
    network const *const net, size_t t, uint8_t *const *const images)
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

    double **result = malloc(t * sizeof(double *));
    for (size_t i = 0; i < t; i++)
    {
        result[i] = malloc(10 * sizeof(double));
    }

    for (size_t i = 0; i < t; i++)
    {
        input_layer_pass(&net->layers[0].input, images[0], v,
                         net->layers[1].conv.k / 2);

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
                    // After the first fully connected layer, only such layers
                    // come, therefore switch to vectors instead of matrices.
                    vectorize_matrix(x->fc.n, x->fc.m, u, p);
                }
                fc_layer_pass(&net->layers[i].fc, p, q);
                swap(&p, &q);
            }
            }
        }

        memcpy(result[i], p, 10 * sizeof(double));
    }

    destroy_matrix(28, u);
    destroy_matrix(28, v);
    free(p);
    free(q);
    return result;
}

void network_save_results(
    char const *const result_fname, size_t t, double *const *const results)
{
    FILE *result_f = fopen(result_fname, "w");
    if (!result_f)
    {
        perror("Error while saving results");
        return;
    }

    for (size_t i = 0; i < t; i++)
    {
        uint8_t max_digit;
        double max_val = 0.0;
        for (size_t j = 0; j < 10; j++)
        {
            if (results[i][j] > max_val)
            {
                max_val = results[i][j];
                max_digit = j + 1;
            }
        }

        fprintf(result_f, "%huu\n", max_digit);
        for (size_t j = 0; j < 10; j++)
        {
            fprintf(result_f, "%lg ", results[i][j]);
        }
    }
}