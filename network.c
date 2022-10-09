#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>

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

    srand(time(0));

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
        kernel_size / 2);

    for (size_t i = num_conv + 2; i < net.l; i++)
    {
        fc_layer *x = &net.layers[i].fc;
        fc_layer_init(
            x, (i == net.l - 1) ? 10 : fc_size,
            (i == num_conv + 2) ? square(net.layers[i - 1].flat.n) : fc_size);

        x->f = (i != net.l - 1) ? ACTIVATION : OUT_ACTIVATION;
        x->fd = (i != net.l - 1) ? ACTIVATION_D : OUT_ACTIVATION_D;

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
            input_layer_init_backprop(&x->input);
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

void network_free(network *const net)
{
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_INPUT:
            input_layer_free(&x->input);
            break;

        case LTYPE_CONV:
            conv_layer_free(&x->conv);
            break;

        case LTYPE_FC:
            fc_layer_free(&x->fc);
            break;

        case LTYPE_FLAT:
            flat_layer_free(&x->flat);
            break;
        }
    }

    free(net->layers);
}

// Feeds the specified image through the network. u, v, p and q must be user
// provided buffers large endough to store intermediate results of any layer.
double *network_pass_one(
    network const *const net, double *const image, double **u, double **v,
    double *p, double *q, bool store_intermed)
{
    input_layer_pass(&net->layers[0].input, image, u, store_intermed);

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
                for (size_t j = 0; j < x->flat.n; j++)
                {
                    memcpy(
                        x->flat.in + j * x->flat.n, (x - 1)->conv.in[j],
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
    network const *const net, size_t t, double *const *const images)
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

    double **results = malloc(t * sizeof(double *));

    for (size_t i = 0; i < t; i++)
    {
        results[i] = network_pass_one(net, images[i], u, v, p, q, 0);
    }

    matrix_free(grid_size, u);
    matrix_free(grid_size, v);
    free(p);
    free(q);
    return results;
}

// Resets the buffers of all layers storing the accumulated gradient to 0.
void network_reset_gradient(network const *const net)
{
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
            conv_layer_reset_gradient(&x->conv);
            break;

        case LTYPE_FC:
            fc_layer_reset_gradient(&x->fc);
            break;

        default:
            break;
        }
    }
}

void network_avg_gradient(network const *const net, size_t t)
{
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_CONV:
            conv_layer_avg_gradient(&x->conv, t);
            break;

        case LTYPE_FC:
            fc_layer_avg_gradient(&x->fc, t);
            break;

        default:
            break;
        }
    }
}

double *vget_prev_in(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_FC:
        return x->fc.in;

    case LTYPE_FLAT:
        return x->flat.in;

    default:
        return 0;
    }
}

double *vget_prev_out(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_FC:
        return x->fc.out;

    case LTYPE_FLAT:
        return x->flat.out;

    default:
        return 0;
    }
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

    default:
        return 0;
    }
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

    default:
        return 0;
    }
}

activation_fn get_prev_fd(network const *const net, size_t i)
{
    layer *x = net->layers + i - 1;
    switch (x->conv.ltype)
    {
    case LTYPE_CONV:
        return x->conv.fd;

    case LTYPE_FC:
        return x->fc.fd;

    case LTYPE_FLAT:
        return (x - 1)->conv.fd;

    default:
        return 0;
    }
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
            conv_layer_update_gradient(&x->conv, mget_prev_out(net, i), u);
            if (i > 1) // Propagating to the input layer is unnecessary.
            {
                conv_layer_backprop(
                    &x->conv, mget_prev_in(net, i), get_prev_fd(net, i), u, v);
                swap(&u, &v);
            }
            break;
        }
        case LTYPE_FC:
        {
            fc_layer_update_gradient(&x->fc, vget_prev_out(net, i), p);
            fc_layer_backprop(
                &x->fc, vget_prev_in(net, i), get_prev_fd(net, i), p, q);
            swap(&p, &q);
            break;
        }
        case LTYPE_FLAT:
        {
            flat_layer_backprop(&x->flat, p, u);
            break;
        }
        }
    }
}

void network_descend(network const *const net)
{
    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;
        switch (x->conv.ltype)
        {
        case LTYPE_INPUT:
        case LTYPE_FLAT:
            break;

        case LTYPE_CONV:
            conv_layer_descend(&x->conv);
            break;

        case LTYPE_FC:
            fc_layer_descend(&x->fc);
            break;
        }
    }
}

// Softmax in combination with the log-likelihood cost function is used,
// therefore computing the loss simplifies dramatically.
void network_get_loss(double *result, uint8_t label)
{
    for (size_t i = 0; i < 10; i++)
    {
        result[i] = result[i] - (double)(label == i);
    }
}

double get_cost(double *result, uint8_t label)
{
    return -log(result[label]);
}

void network_train(
    network const *const net, size_t epochs, size_t t, double **const images,
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

    network_init_backprop(net);

    for (size_t e = 0; e < epochs; e++)
    {
        vector_random_shuffle(t, images);
        long double cost = 0.0;

        for (size_t i = 0; i < t; i++)
        {
            double *result = network_pass_one(net, images[i], u, v, p, q, 1);

            cost += get_cost(result, labels[i]);
            network_get_loss(result, labels[i]);

            memcpy(p, result, 10 * sizeof(double));
            free(result);
            network_backprop(net, u, v, p, q);

            if (!((i + 1) % BATCH_SIZE))
            {
                network_avg_gradient(net, t);
                network_descend(net);
                network_reset_gradient(net);

                printf("%Lg\n", cost);
                cost = 0.0;
            }
        }

        if (t % BATCH_SIZE)
        {
            network_avg_gradient(net, t);
            network_descend(net);
            printf("%Lg\n", cost);
        }
    }

    matrix_free(grid_size, u);
    matrix_free(grid_size, v);
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
    FILE *stream = fopen(result_fname, "w");
    if (!stream)
    {
        perror("Error while saving results");
        return;
    }

    uint8_t *max_digits = calc_max_digits(t, results);

    for (size_t i = 0; i < t; i++)
    {
        fprintf(stream, "%hhu\n", max_digits[i]);
        vector_print(10, results[i], stream);
    }

    free(max_digits);
    fclose(stream);
}

void network_print_accuracy(
    size_t t, double *const *const results, uint8_t *const labels)
{
    size_t digit_correct[10], digit_occ[10];
    memset(digit_correct, 0, 10 * sizeof(size_t));
    memset(digit_occ, 0, 10 * sizeof(size_t));
    size_t total_correct = 0;

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

network network_read(char const *const fname)
{
    network net;
    FILE *stream = fopen(fname, "r");
    if (!stream)
    {
        perror("Error while reading network from file");
        memset(&net, 0, sizeof(network));
        return net;
    }

    fscanf(stream, "%zu", &net.l);
    net.layers = malloc(net.l * sizeof(layer));

    for (size_t i = 0; i < net.l; i++)
    {
        layer *x = net.layers + i;

        fscanf(stream, "%hhu", &x->conv.ltype);

        switch (x->conv.ltype)
        {
        case LTYPE_INPUT:
            input_layer_read(&x->input, stream);
            break;

        case LTYPE_CONV:
            conv_layer_read(&x->conv, stream);
            break;

        case LTYPE_FC:
            fc_layer_read(&x->fc, stream);
            break;

        case LTYPE_FLAT:
            flat_layer_read(&x->flat, stream);
            assert(x->flat.n == (x - 1)->conv.n);
            break;
        }
    }

    net.layers[net.l - 1].fc.f = OUT_ACTIVATION;
    net.layers[net.l - 1].fc.fd = OUT_ACTIVATION_D;
    return net;
}

void network_print(network const *const net, char const *const fname)
{
    FILE *stream = fopen(fname, "w");
    if (!stream)
    {
        perror("Error while saving network to disk");
        return;
    }

    fprintf(stream, "%zu\n\n", net->l);

    for (size_t i = 0; i < net->l; i++)
    {
        layer *x = net->layers + i;

        fprintf(stream, "%hhu\n", x->conv.ltype);

        switch (x->conv.ltype)
        {
        case LTYPE_INPUT:
            input_layer_print(&x->input, stream);
            break;

        case LTYPE_CONV:
            conv_layer_print(&x->conv, stream);
            break;

        case LTYPE_FC:
            fc_layer_print(&x->fc, stream);
            break;

        case LTYPE_FLAT:
            flat_layer_print(&x->flat, stream);
            break;
        }
    }

    fclose(stream);
}