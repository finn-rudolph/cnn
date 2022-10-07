#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <stdbool.h>

#include "network.h"
#include "util.h"
#include "def.h"

network network_init(
    size_t num_conv, size_t num_fc, size_t kernel_size, size_t fc_size)
{
    network net;
    net.l = num_conv + num_fc + 2;
    net.layers = malloc(net.l * sizeof(layer));

    input_layer_init(&net.layers[0].input, 28, kernel_size / 2);

    for (size_t i = 1; i < num_conv + 1; i++)
    {
        conv_layer *x = &net.layers[i].conv;
        conv_layer_init(x, 28, kernel_size);
        x->bias = rand_double(PARAM_MIN, PARAM_MAX);

        for (size_t j = 0; j < x->k; j++)
        {
            for (size_t k = 0; k < x->k; k++)
            {
                x->kernel[j][k] = rand_double(PARAM_MIN, PARAM_MAX);
            }
        }
    }

    flat_layer_init(
        &net.layers[num_conv + 1].flat, net.layers[num_conv].conv.n,
        kernel_size - 1);

    for (size_t i = num_conv + 2; i < net.l; i++)
    {
        fc_layer *x = &net.layers[i].fc;
        fc_layer_init(
            x, (i == net.l - 1) ? 10 : fc_size,
            (i == num_conv + 2) ? square(net.layers[i - 1].flat.n) : fc_size);

        if (i != net.l - 1)
        {
            x->f = ACTIVATION;
            x->fd = ACTIVATION_D;
        }
        else
        {
            x->f = OUT_ACTIVATION;
            x->fd = 0;
        }

        for (size_t j = 0; j < x->n; j++)
        {
            for (size_t k = 0; k < x->m; k++)
            {
                x->weight[j][k] = rand_double(PARAM_MIN, PARAM_MAX);
            }
            x->bias[j] = rand_double(PARAM_MIN, PARAM_MAX);
        }
    }

    return net;
}

void network_init_backprop(network const *const net)
{
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_INPUT:
            break;

        case LTYPE_CONV:
            conv_layer_init_backprop(&x->conv);
            break;

        case LTYPE_FC:
            fc_layer_init_backprop(&x->fc);
            break;

        case LTYPE_FLAT:
            flat_layer_init_backprop(&x->flat);
            break;
        }
    }
}

void network_destroy(network *const net)
{
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_INPUT:
            break;

        case LTYPE_CONV:
            conv_layer_destroy(&x->conv);
            break;

        case LTYPE_FC:
            fc_layer_destroy(&x->fc);
            break;

        case LTYPE_FLAT:
            flat_layer_destroy(&x->flat);
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

        fscanf(net_f, "%hhu", &x->conv.ltype);

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

        case LTYPE_FLAT:
            flat_layer_read(&x->flat, net_f);
            break;
        }
    }

    net.layers[net.l - 1].fc.f = OUT_ACTIVATION;
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

        fprintf(net_f, "%hhu\n", x->conv.ltype);

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

        case LTYPE_FLAT:
            flat_layer_save(&x->flat, net_f);
            break;
        }
    }

    fclose(net_f);
}

double *network_pass_one(
    network const *const net, uint8_t *const image, double **u, double **v,
    double *p, double *q, bool store_intermed)
{
    input_layer_pass(&net->layers[0].input, image, u);

    for (size_t i = 1; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
        {
            conv_layer_pass(&x->conv, u, v, store_intermed);
            swap(&u, &v);
            break;
        }
        case LTYPE_FC:
        {
            fc_layer_pass(&x->fc, p, q, store_intermed);
            swap(&p, &q);
            break;
        }
        case LTYPE_FLAT:
        {
            flat_layer_pass(&x->flat, u, p);
            if (store_intermed)
            {
                for (size_t i = 0; i < x->flat.n; i++)
                {
                    memcpy(
                        x->flat.in + i * x->flat.n, (x - 1)->conv.in[i],
                        x->flat.n * sizeof(double));
                }
                memcpy(x->flat.out, p, square(x->flat.n) * sizeof(double));
            }
            break;
        }
        }
    }

    double *result = malloc(10 * sizeof(double));
    memcpy(result, p, 10 * sizeof(double));
    return result;
}

double **network_pass_forward(
    network const *const net, size_t t, uint8_t *const *const images)
{
    size_t const grid_size = 28 + 2 * net->layers[0].input.padding;
    // Two grid containers for convolutional layers, two linear containers for
    // fully connected layers. Serve as input / output buffer.
    double **u = malloc(grid_size * sizeof(double *)),
           **v = malloc(grid_size * sizeof(double *)),
           *p = malloc(square(grid_size) * sizeof(double)),
           *q = malloc(square(grid_size) * sizeof(double));

    for (size_t i = 0; i < grid_size; i++)
    {
        u[i] = malloc(grid_size * sizeof(double));
        v[i] = malloc(grid_size * sizeof(double));
    }

    double **result = malloc(t * sizeof(double *));
    for (size_t i = 0; i < t; i++)
    {
        result[i] = malloc(10 * sizeof(double));
    }

    for (size_t i = 0; i < t; i++)
    {
        result[i] = network_pass_one(net, images[i], u, v, p, q, 0);
    }

    destroy_matrix(grid_size, u);
    destroy_matrix(grid_size, v);
    free(p);
    free(q);
    return result;
}

double *vget_prev_in(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_INPUT:
    case LTYPE_CONV:
        return 0; // Does not occur.

    case LTYPE_FC:
        return x->fc.in;

    case LTYPE_FLAT:
        return x->flat.in;
    }

    return 0;
}

double *vget_prev_out(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_INPUT:
    case LTYPE_CONV:
        return 0; // Does not occur.

    case LTYPE_FC:
        return x->fc.out;

    case LTYPE_FLAT:
        return x->flat.out;
    }

    return 0;
}

double **mget_prev_in(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_INPUT:
        return x->input.out;

    case LTYPE_CONV:
        return x->conv.in;

    case LTYPE_FC:
    case LTYPE_FLAT:
        return 0; // Does not occur.
    }

    return 0;
}

double **mget_prev_out(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_INPUT:
        return x->input.out;

    case LTYPE_CONV:
        return x->conv.out;

    case LTYPE_FC:
    case LTYPE_FLAT:
        return 0; // Does not occur.
    }

    return 0;
}

// The delta vector of the last layer must be in the first 10 positions of p.
void network_backprop(
    network const *const net, double **u, double **v, double *p, double *q)
{
    for (size_t i = net->l - 1; i; i--)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
        {
            conv_layer_backprop(&x->conv, u, v);
            swap(&u, &v);
            break;
        }
        case LTYPE_FC:
        {
            fc_layer_backprop(
                &x->fc, vget_prev_in(net, i), vget_prev_out(net, i), p, q);
            swap(&p, &q);
            break;
        }
        case LTYPE_FLAT:
        {
            flat_layer_backprop(&x->flat);
            break;
        }
        }
    }
}

void network_descend(network const *const net)
{
}

// Softmax in combination with the cross entropy cost function is used,
// therefore computing the loss simplifies dramatically.
void network_get_loss(double *p, uint8_t label)
{
    for (size_t i = 0; i < 10; i++)
    {
        p[i] = p[i] - (double)(label == i);
    }
}

void network_train(
    network const *const net, size_t r, size_t t, uint8_t *const *const images,
    uint8_t *const labels)
{
    size_t const grid_size = 28 + 2 * net->layers[0].input.padding;

    double **u = malloc(grid_size * sizeof(double *)),
           **v = malloc(grid_size * sizeof(double *)),
           *p = malloc(square(grid_size) * sizeof(double)),
           *q = malloc(square(grid_size) * sizeof(double));

    for (size_t i = 0; i < grid_size; i++)
    {
        u[i] = malloc(grid_size * sizeof(double));
        v[i] = malloc(grid_size * sizeof(double));
    }

    for (size_t e = 0; e < r; e++)
    {
        for (size_t i = 0; i < t; i++)
        {
            double *result = network_pass_one(net, images[i], u, v, p, q, 1);
            memcpy(p, result, 10 * sizeof(double));
            network_get_loss(p, labels[i]);
            network_backprop(net, u, v, p, q);
        }

        for (size_t i = 0; i < net->l; i++)
        {
            layer *x = net->layers + i;

            switch (x->conv.ltype)
            {
            case LTYPE_INPUT:
                break;

            case LTYPE_CONV:
                conv_layer_avg_gradient(&x->conv, t);
                break;

            case LTYPE_FC:
                fc_layer_avg_gradient(&x->fc, t);
                break;

            case LTYPE_FLAT:
                break;
            }
        }

        network_descend(net);
    }

    destroy_matrix(grid_size, u);
    destroy_matrix(grid_size, v);
    free(p);
    free(q);
}

uint8_t *calc_max_digits(size_t t, double *const *const results)
{
    uint8_t *max_digits = malloc(t * sizeof(uint8_t));

    for (size_t i = 0; i < t; i++)
    {
        double max_val = 0.0;
        for (size_t j = 0; j < 10; j++)
        {
            if (results[i][j] > max_val)
            {
                max_val = results[i][j];
                max_digits[i] = j;
            }
        }
    }

    return max_digits;
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

    uint8_t *max_digits = calc_max_digits(t, results);

    for (size_t i = 0; i < t; i++)
    {
        fprintf(result_f, "%hhu\n", max_digits[i]);
        for (size_t j = 0; j < 10; j++)
        {
            fprintf(result_f, "%lg ", results[i][j]);
        }
        fputc('\n', result_f);
    }

    free(max_digits);
    fclose(result_f);
}

void network_print_accuracy(
    size_t t, double *const *const results, uint8_t *const labels)
{
    unsigned digit_correct[10], digit_occ[10];
    memset(digit_correct, 0, 10 * sizeof(unsigned));
    memset(digit_occ, 0, 10 * sizeof(unsigned));
    unsigned total_correct = 0;

    uint8_t *max_digits = calc_max_digits(t, results);

    for (size_t i = 0; i < t; i++)
    {
        if (max_digits[i] == labels[i])
        {
            digit_correct[max_digits[i]]++;
            total_correct++;
        }
        digit_occ[labels[i]]++;
    }

    printf("  Total accuracy: %lg\n", (double)total_correct / (double)t);
    for (uint8_t i = 0; i < 10; i++)
    {
        printf("  %hhu: %lg\n", i, (double)digit_correct[i] / (double)digit_occ[i]);
    }

    free(max_digits);
}