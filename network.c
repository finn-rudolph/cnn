#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "network.h"
#include "util.h"
#include "def.h"

#define PARAM_MIN -1.0
#define PARAM_MAX 1.0

network network_init(
    size_t num_conv, size_t num_fc, size_t kernel_size, size_t fc_size)
{
    network net;
    net.l = num_conv + num_fc + 1;
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

    for (size_t i = num_conv + 1; i < net.l; i++)
    {
        fc_layer *x = &net.layers[i].fc;
        fc_layer_init(
            x, (i == net.l - 1) ? 10 : fc_size,
            (i == num_conv + 1) ? square(net.layers[i - 1].conv.n) : fc_size);
        x->f = (i == net.l - 1) ? &out_actiavtion : &activation;

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
        }
    }

    net.layers[net.l - 1].fc.f = &out_actiavtion;
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
        }
    }

    fclose(net_f);
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
    for (size_t z = 0; z < t; z++)
    {
        result[z] = malloc(10 * sizeof(double));
    }

    for (size_t z = 0; z < t; z++)
    {
        input_layer_pass(&net->layers[0].input, images[z], u);

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
                    vectorize_matrix((x - 1)->conv.n, (x - 1)->conv.n,
                                     (x - 1)->conv.k / 2, u, p);
                }
                fc_layer_pass(&net->layers[i].fc, p, q);
                swap(&p, &q);
                break;
            }
            }
        }

        memcpy(result[z], p, 10 * sizeof(double));
    }

    destroy_matrix(grid_size, u);
    destroy_matrix(grid_size, v);
    free(p);
    free(q);
    return result;
}

void network_train(
    network const *const net, size_t r, size_t t, uint8_t *const *const images,
    uint8_t *const labels)
{
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