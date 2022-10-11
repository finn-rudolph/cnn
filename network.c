#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <threads.h>
#include <sys/sysinfo.h>

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

void shuffle_images(size_t n, double **const images, uint8_t *const labels)
{
    srand(time(0));
    for (size_t i = n - 1; i; i--)
    {
        size_t j = rand() % (i + 1);
        swap(images + i, images + j);
        swap(labels + i, labels + j);
    }
}

// Softmax in combination with the log-likelihood cost function is used,
// therefore computing the loss simplifies dramatically.
void get_loss(double *const result, uint8_t label)
{
    for (size_t i = 0; i < 10; i++)
    {
        result[i] = result[i] - (double)(label == i);
    }
}

double get_cost(double const *const result, uint8_t label)
{
    return -log(result[label]);
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

// Creates duplicates and initializes separate buffers for backpropagation.
network *replicate_net(network const *const net, size_t n)
{
    network *replicas = malloc(n * sizeof(network));

    for (size_t i = 0; i < n; i++)
    {
        network *const z = replicas + i;
        z->l = net->l;
        z->layers = malloc(z->l * sizeof(layer));
        memcpy(z->layers, net->layers, net->l * sizeof(layer));

        network_init_backprop(replicas + i);
    }

    return replicas;
}

void free_replicas(size_t n, network *const replicas)
{
    // Must be done manually as the layer functions would also free the weights
    // and biases containers.
    for (size_t i = 0; i < n; i++)
    {
        network *const z = replicas + i;
        for (size_t j = 0; j < z->l; j++)
        {
            layer *const x = z->layers + j;

            switch (x->conv.ltype)
            {
            case LTYPE_INPUT:
            {
                matrix_free(x->input.n + 2 * x->input.padding, x->input.out);
                break;
            }
            case LTYPE_CONV:
            {
                matrix_free(x->conv.n, x->conv.in);
                matrix_free(x->conv.n + x->conv.k - 1, x->conv.out);
                matrix_free(x->conv.k, x->conv.kernel_gradient);
                break;
            }
            case LTYPE_FC:
            {
                free(x->fc.in);
                free(x->fc.out);
                matrix_free(x->fc.n, x->fc.weight_gradient);
                free(x->fc.bias_gradient);
                break;
            }
            case LTYPE_FLAT:
            {
                free(x->flat.in);
                free(x->flat.out);
                break;
            }
            }
        }

        free(z->layers);
    }
}

// Assumes the gradients in net are all set to 0.
void sum_replica_gradients(
    network const *const net, size_t n, network const *const replicas)
{
    for (size_t i = 0; i < n; i++)
    {
        network const *const z = replicas + i;

        for (size_t j = 0; j < z->l; j++)
        {
            layer *const x = z->layers + j, *const y = net->layers + j;

            switch (x->conv.ltype)
            {
            case LTYPE_CONV:
                matrix_add(
                    x->conv.k, x->conv.k, x->conv.kernel_gradient,
                    y->conv.kernel_gradient);
                y->conv.bias_gradient += x->conv.bias_gradient;
                break;

            case LTYPE_FC:
                matrix_add(
                    x->fc.n, x->fc.m, x->fc.weight_gradient,
                    y->fc.weight_gradient);
                vector_add(x->fc.n, x->fc.bias_gradient, y->fc.bias_gradient);
                break;

            default:
                break;
            }
        }
    }
}

typedef struct parallel_args parallel_args;
struct parallel_args
{
    network const *net;
    size_t t;
    double **images;
    uint8_t *labels;
    double **u, **v, *p, *q;
    double *cost, **results;
    bool store_cost, store_results, do_backprop;
};

int pass_parallel(void *args)
{
    parallel_args *a = args;
    for (size_t i = 0; i < a->t; i++)
    {
        double *result = network_pass_one(
            a->net, a->images[i], a->u, a->v, a->p, a->q, a->do_backprop);

        if (a->store_cost)
        {
            *a->cost += get_cost(result, a->labels[i]);
        }
        if (a->store_results)
        {
            a->results[i] = result;
        }
        if (a->do_backprop)
        {
            get_loss(result, a->labels[i]);
            memcpy(a->p, result, 10 * sizeof(double));
            network_backprop(a->net, a->u, a->v, a->p, a->q);
        }

        if (!a->store_results)
        {
            free(result);
        }
    }
    return 0;
}

double **network_pass_forward(
    network const *const restrict net, size_t t, double **const restrict images)
{
    size_t const num_threads = get_nprocs();
    printf("Using %zu threads.\n", num_threads);

    size_t const grid_size = 28 + 2 * net->layers[0].input.padding;

    double ***u = malloc(num_threads * sizeof(double **)),
           ***v = malloc(num_threads * sizeof(double **)),
           **p = matrix_alloc(num_threads, square(grid_size)),
           **q = matrix_alloc(num_threads, square(grid_size));

    for (size_t i = 0; i < num_threads; i++)
    {
        u[i] = matrix_alloc(grid_size, grid_size);
        v[i] = matrix_alloc(grid_size, grid_size);
    }

    network_init_backprop(net);
    double **results = malloc(t * sizeof(double *));

    for (size_t i = 0; i < t; i += num_threads * BATCH_SIZE)
    {
        thrd_t threads[num_threads];
        parallel_args args[num_threads];

        for (size_t j = 0; j < num_threads && i + j * BATCH_SIZE < t; j++)
        {
            args[j] = (parallel_args){
                .net = net,
                .t = min(BATCH_SIZE, t - i - j * BATCH_SIZE),
                .images = images + i + j * BATCH_SIZE,
                .labels = 0,
                .u = u[j],
                .v = v[j],
                .p = p[j],
                .q = q[j],
                .cost = 0,
                .results = results + i + j * BATCH_SIZE,
                .store_cost = 0,
                .store_results = 1,
                .do_backprop = 0};

            thrd_create(threads + j, pass_parallel, args + j);
        }

        for (size_t j = 0; j < num_threads && i + j * BATCH_SIZE < t; j++)
        {
            thrd_join(threads[j], 0);
        }
    }

    for (size_t i = 0; i < num_threads; i++)
    {
        matrix_free(grid_size, u[i]);
        matrix_free(grid_size, v[i]);
    }
    free(u);
    free(v);
    matrix_free(num_threads, p);
    matrix_free(num_threads, q);

    return results;
}

void network_train(
    network const *const restrict net, size_t epochs, size_t t,
    double **const restrict images, uint8_t *const restrict labels,
    char const *const restrict fname)
{
    size_t const num_threads = get_nprocs();
    printf("Using %zu threads.\n", num_threads);

    network *replicas = replicate_net(net, num_threads);

    size_t const grid_size = 28 + 2 * net->layers[0].input.padding;

    double ***u = malloc(num_threads * sizeof(double **)),
           ***v = malloc(num_threads * sizeof(double **)),
           **p = matrix_alloc(num_threads, square(grid_size)),
           **q = matrix_alloc(num_threads, square(grid_size));

    for (size_t i = 0; i < num_threads; i++)
    {
        u[i] = matrix_alloc(grid_size, grid_size);
        v[i] = matrix_alloc(grid_size, grid_size);
    }

    network_init_backprop(net);

    for (size_t e = 0; e < epochs; e++)
    {
        shuffle_images(t, images, labels);

        for (size_t i = 0; i < t; i += num_threads * BATCH_SIZE)
        {
            network_reset_gradient(net);
            for (size_t j = 0; j < num_threads; j++)
            {
                network_reset_gradient(replicas + j);
            }

            double costs[num_threads];
            thrd_t threads[num_threads];
            parallel_args args[num_threads];

            for (size_t j = 0; j < num_threads && i + j * BATCH_SIZE < t; j++)
            {
                costs[j] = 0.0;
                args[j] = (parallel_args){
                    .net = replicas + j,
                    .t = min(BATCH_SIZE, t - i - j * BATCH_SIZE),
                    .images = images + i + j * BATCH_SIZE,
                    .labels = labels + i + j * BATCH_SIZE,
                    .u = u[j],
                    .v = v[j],
                    .p = p[j],
                    .q = q[j],
                    .cost = costs + j,
                    .results = 0,
                    .store_cost = 1,
                    .store_results = 0,
                    .do_backprop = 1};

                thrd_create(threads + j, pass_parallel, args + j);
            }

            double total_cost = 0.0;
            for (size_t j = 0; j < num_threads && i + j * BATCH_SIZE < t; j++)
            {
                thrd_join(threads[j], 0);
                total_cost += costs[j];
            }

            sum_replica_gradients(net, num_threads, replicas);
            network_avg_gradient(net, min(BATCH_SIZE * num_threads, t - i));
            network_descend(net);

            printf("%lg\n",
                   total_cost / (double)(min(BATCH_SIZE * num_threads, t - i)));
        }

        network_print(net, fname);
    }

    free_replicas(num_threads, replicas);
    free(replicas);

    for (size_t i = 0; i < num_threads; i++)
    {
        matrix_free(grid_size, u[i]);
        matrix_free(grid_size, v[i]);
        free(p[i]);
        free(q[i]);
    }
    free(u);
    free(v);
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

void network_print_results(
    char const *const restrict result_fname, size_t t,
    double *const *const restrict results)
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
    size_t t, double *const *const restrict results,
    uint8_t *const restrict labels)
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
        printf("  %hhu: %lg\n", i,
               (double)digit_correct[i] / (double)digit_occ[i]);
    }

    free(max_digits);
}

network network_read(char const *const fname)
{
    network net;
    FILE *const stream = fopen(fname, "r");
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

void network_print(
    network const *const restrict net, char const *const restrict fname)
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