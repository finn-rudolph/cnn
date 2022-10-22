#include <memory.h>
#include <assert.h>

#include "convolution.h"
#include "network_config.h"
#include "layer.h"
#include "util.h"

void input_layer_init(input_layer *const x, size_t n)
{
    x->ltype = LTYPE_INPUT;
    x->n = n;
    x->out = 0;
}

void input_layer_init_backprop(input_layer *const x)
{
    x->out = matrix_alloc(x->n, x->n);
}

void input_layer_free(input_layer *const x)
{
    if (x->out)
    {
        matrix_free(x->n, x->out);
    }
}

void input_layer_pass(
    input_layer const *const x, double *const image,
    double *const *const out, bool store_intermed)
{
    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            out[i][j] = image[i * x->n + j];
        }
    }

    if (store_intermed)
    {
        matrix_copy(x->n, x->n, out, x->out);
    }

#ifdef DEBUG_MODE

    printf("Input image:\n");
    matrix_print(x->n, x->n, out, stdout);

#endif
}

void input_layer_read(input_layer *const x, FILE *const stream)
{
    fscanf(stream, "%zu", &x->n);
    input_layer_init(x, x->n);
}

void input_layer_print(input_layer const *const x, FILE *const stream)
{
    fprintf(stream, "%zu\n\n", x->n);
}

void conv_layer_init(conv_layer *const x, size_t n, size_t k)
{
    x->ltype = LTYPE_CONV;
    x->n = n;
    x->k = k;
    x->f = ACTIVATION;
    x->fd = ACTIVATION_D;
    x->kernel = matrix_alloc(k, k);

    x->in = 0;
    x->out = 0;
    x->kernel_gradient = 0;
    x->bias_gradient = 0.0;
}

void conv_layer_init_backprop(conv_layer *const x)
{
    x->in = matrix_alloc(x->n, x->n);
    x->out = matrix_alloc(x->n, x->n);
    x->kernel_gradient = matrix_alloc(x->k, x->k);
    x->bias_gradient = 0.0;
}

void conv_layer_reset_gradient(conv_layer *const x)
{
    assert(x->kernel_gradient);

    for (size_t i = 0; i < x->k; i++)
    {
        for (size_t j = 0; j < x->k; j++)
        {
            x->kernel_gradient[i][j] = 0.0;
        }
    }
    x->bias_gradient = 0.0;
}

void conv_layer_free(conv_layer *const x)
{
    matrix_free(x->k, x->kernel);

    if (x->in)
    {
        matrix_free(x->n, x->in);
    }
    if (x->out)
    {
        matrix_free(x->n, x->out);
    }
    if (x->kernel_gradient)
    {
        matrix_free(x->k, x->kernel_gradient);
    }
}

void conv_layer_pass(
    conv_layer const *const x, double *const *const in,
    double *const *const out, bool store_intermed)
{

#ifdef CONV_FFT

    double **flipped_kernel = flip_kernel(x->k, x->kernel);
    convolve_fft(x->n, x->k, in, out, flipped_kernel, 0);
    matrix_free(x->k, flipped_kernel);

#else

    convolve(x->n, x->k, in, out, x->kernel, 0);

#endif

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            out[i][j] += x->bias;
        }
    }

    if (store_intermed)
    {
        matrix_copy(x->n, x->n, out, x->in);
    }

#ifdef DEBUG_MODE

    printf("Input values:\n");
    matrix_print(x->n, x->n, out, stdout);

#endif

    for (size_t i = 0; i < x->n; i++)
    {
        (*x->f)(x->n, out[i]);
    }

    if (store_intermed)
    {
        matrix_copy(x->n, x->n, out, x->out);
    }

#ifdef DEBUG_MODE

    printf("Output values:\n");
    matrix_print(x->n, x->n, out, stdout);

#endif
}

void conv_layer_update_gradient(
    conv_layer *const x, double *const *const prev_out,
    double *const *const delta)
{
    assert(x->kernel_gradient);

    // Convolve the current layer's deltas with the previous layer's outputs to
    // get the gradient for the kernel.

#ifdef CONV_FFT

    double **flipped_delta = flip_kernel(x->n, delta);
    convolve_fft_offset(
        x->n, x->n, prev_out, x->kernel_gradient, flipped_delta, x->k / 2, 1);
    matrix_free(x->n, flipped_delta);

#else

    convolve_offset(
        x->n, x->n, prev_out, x->kernel_gradient, delta, x->k / 2, 1);

#endif

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            x->bias_gradient += delta[i][j];
        }
    }
}

void conv_layer_backprop(
    conv_layer const *const x, double *const *const prev_in,
    activation_fn prev_fd, double *const *const delta,
    double *const *const ndelta)
{
    // The kernel rotated by 180° convolved with the current layer's deltas are
    // the previous layer's deltas.

#ifdef CONV_FFT

    convolve_fft(x->n, x->k, delta, ndelta, x->kernel, 0);

#else

    double **flipped_kernel = flip_kernel(x->k, x->kernel);
    convolve(x->n, x->k, delta, ndelta, flipped_kernel, 0);
    matrix_free(x->k, flipped_kernel);

#endif

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            ndelta[i][j] *= (*prev_fd)(prev_in[i][j]);
        }
    }

#ifdef DEBUG_MODE

    printf("Delta:\n");
    matrix_print(x->n, x->n, delta, stdout);

#endif
}

void conv_layer_descend(conv_layer *const x, size_t t)
{
    assert(x->kernel_gradient);

    for (size_t i = 0; i < x->k; i++)
    {
        for (size_t j = 0; j < x->k; j++)
        {
            x->kernel[i][j] *=
                (1.0 - (LEARN_RATE * REGULARIZATION_PARAM) / (double)t);
            x->kernel[i][j] -=
                (x->kernel_gradient[i][j] / (double)t) * LEARN_RATE;
        }
    }
    x->bias -= (x->bias_gradient / (double)t) * LEARN_RATE;

#ifdef DEBUG_MODE

    printf("Kernel gradient:\n");
    matrix_print(x->k, x->k, x->kernel_gradient, stdout);
    printf("Bias gradient:\n %lg\n", x->bias_gradient);

#endif
}

void conv_layer_read(conv_layer *const x, FILE *const stream)
{
    fscanf(stream, "%zu %zu %lg", &x->n, &x->k, &x->bias);
    conv_layer_init(x, x->n, x->k);

    for (size_t i = 0; i < x->k; i++)
    {
        for (size_t j = 0; j < x->k; j++)
        {
            fscanf(stream, "%lg", &x->kernel[i][j]);
        }
    }
}

void conv_layer_print(conv_layer const *const x, FILE *const stream)
{
    fprintf(stream, "%zu %zu\n%lg\n", x->n, x->k, x->bias);
    matrix_print(x->k, x->k, x->kernel, stream);
}

void fc_layer_init(fc_layer *const x, size_t n, size_t m)
{
    x->ltype = LTYPE_FC;
    x->n = n;
    x->m = m;
    x->f = ACTIVATION;
    x->fd = ACTIVATION_D;
    x->weight = matrix_alloc(n, m);
    x->bias = malloc(n * sizeof(double));

    x->in = 0;
    x->out = 0;
    x->weight_gradient = 0;
    x->bias_gradient = 0;
}

void fc_layer_init_backprop(fc_layer *const x)
{
    x->out = malloc(x->n * sizeof(double));
    x->in = malloc(x->n * sizeof(double));
    x->weight_gradient = matrix_alloc(x->n, x->m);
    x->bias_gradient = malloc(x->n * sizeof(double));
}

void fc_layer_reset_gradient(fc_layer *const x)
{
    assert(x->weight_gradient && x->bias_gradient);

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->m; j++)
        {
            x->weight_gradient[i][j] = 0.0;
        }
        x->bias_gradient[i] = 0.0;
    }
}

void fc_layer_free(fc_layer *const x)
{
    matrix_free(x->n, x->weight);
    free(x->bias);

    if (x->in)
    {
        free(x->in);
    }
    if (x->out)
    {
        free(x->out);
    }
    if (x->weight_gradient)
    {
        matrix_free(x->n, x->weight_gradient);
    }
    if (x->bias_gradient)
    {
        free(x->bias_gradient);
    }
}

void fc_layer_pass(
    fc_layer const *const x, double *const in,
    double *const out, bool store_intermed)
{
    vector_mul_matrix(x->n, x->m, in, x->weight, out);

    for (size_t i = 0; i < x->n; i++)
    {
        out[i] += x->bias[i];
    }

    if (store_intermed)
    {
        memcpy(x->in, out, x->n * sizeof(double));
    }

#ifdef DEBUG_MODE

    printf("Input values:\n");
    vector_print(x->n, out, stdout);
    putchar('\n');

#endif

    (*x->f)(x->n, out);

    if (store_intermed)
    {
        memcpy(x->out, out, x->n * sizeof(double));
    }

#ifdef DEBUG_MODE

    printf("Output values:\n");
    vector_print(x->n, out, stdout);
    putchar('\n');

#endif
}

void fc_layer_update_gradient(
    fc_layer *const x, double *const prev_out,
    double *const delta)
{
    assert(x->weight_gradient && x->bias_gradient);

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->m; j++)
        {
            x->weight_gradient[i][j] += prev_out[j] * delta[i];
        }
        x->bias_gradient[i] += delta[i];
    }
}

void fc_layer_backprop(
    fc_layer const *const x, double *const prev_in,
    activation_fn prev_fd, double *const delta,
    double *const ndelta)
{
    for (size_t j = 0; j < x->m; j++)
    {
        ndelta[j] = 0.0;
        for (size_t i = 0; i < x->n; i++)
        {
            ndelta[j] += delta[i] * x->weight[i][j];
        }
        ndelta[j] *= (*prev_fd)(prev_in[j]);
    }

#ifdef DEBUG_MODE

    printf("Delta:\n");
    vector_print(x->n, delta, stdout);

#endif
}

void fc_layer_descend(fc_layer *const x, size_t t)
{
    assert(x->weight_gradient && x->bias_gradient);

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->m; j++)
        {
            x->weight[i][j] *=
                (1.0 - (LEARN_RATE * REGULARIZATION_PARAM) / (double)t);
            x->weight[i][j] -=
                (x->weight_gradient[i][j] / (double)t) * LEARN_RATE;
        }
        x->bias[i] -= (x->bias_gradient[i] / (double)t) * LEARN_RATE;
    }

#ifdef DEBUG_MODE

    printf("Weight gradients:\n");
    matrix_print(x->n, x->m, x->weight_gradient, stdout);
    printf("Bias gradients:\n");
    vector_print(x->n, x->bias_gradient, stdout);

#endif
}

void fc_layer_read(fc_layer *const x, FILE *const stream)
{
    fscanf(stream, "%zu %zu", &x->n, &x->m);
    fc_layer_init(x, x->n, x->m);

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->m; j++)
        {
            fscanf(stream, "%lg", &x->weight[i][j]);
        }
    }

    for (size_t i = 0; i < x->n; i++)
    {
        fscanf(stream, "%lg", &x->bias[i]);
    }
}

void fc_layer_print(fc_layer const *const x, FILE *const stream)
{
    fprintf(stream, "%zu %zu\n", x->n, x->m);
    matrix_print(x->n, x->m, x->weight, stream);
    vector_print(x->n, x->bias, stream);
    fputc('\n', stream);
}

void flat_layer_init(flat_layer *const x, size_t n)
{
    x->ltype = LTYPE_FLAT;
    x->n = n;
    x->in = 0;
    x->out = 0;
}

void flat_layer_init_backprop(flat_layer *const x)
{
    x->in = malloc(square(x->n) * sizeof(double));
    x->out = malloc(square(x->n) * sizeof(double));
}

void flat_layer_free(flat_layer *const x)
{
    if (x->in)
    {
        free(x->in);
    }
    if (x->out)
    {
        free(x->out);
    }
}

void flat_layer_pass(
    flat_layer const *const x, double *const *const in, double *const out)
{
    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            out[i * x->n + j] = in[i][j];
        }
    }
}

void flat_layer_backprop(
    flat_layer const *const x, double *const delta, double *const *const ndelta)
{
    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            ndelta[i][j] = delta[i * x->n + j];
        }
    }
}

void flat_layer_read(flat_layer *const x, FILE *const stream)
{
    fscanf(stream, "%zu", &x->n);
    flat_layer_init(x, x->n);
}

void flat_layer_print(flat_layer const *const x, FILE *const stream)
{
    fprintf(stream, "%zu\n\n", x->n);
}