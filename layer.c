#include "layer.h"
#include "convolution.h"
#include "util.h"
#include "file_io.h"

void input_layer_init(input_layer *const x, size_t n)
{
    x->ltype = LTYPE_CONV;
    x->n = n;
}

void conv_layer_init(conv_layer *const x, size_t n, size_t k)
{
    x->ltype = LTYPE_CONV;
    x->n = n;
    x->k = k;
    x->kernel = malloc(k * sizeof(double *));
    for (size_t i = 0; i < k; i++)
    {
        x->kernel[i] = malloc(k * sizeof(double));
    }
}

void fc_layer_init(fc_layer *const x, size_t n, size_t m)
{
    x->ltype = LTYPE_FC;
    x->n = n;
    x->m = m;
    x->weight = malloc(n * sizeof(double *));
    x->bias = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
    {
        x->weight[i] = malloc(m * sizeof(double));
    }
}

void conv_layer_destroy(conv_layer *const x)
{
    for (size_t i = 0; i < x->k; i++)
    {
        free(x->kernel[i]);
    }
    free(x->kernel);
}

void fc_layer_destroy(fc_layer *const x)
{
    for (size_t i = 0; i < x->n; i++)
    {
        free(x->weight[i]);
    }
    free(x->weight);
    free(x->bias);
}

// Assumes the output of the former layer is layed out such that a margin of
// half the kernel size just fits in.
void pad_avg(size_t n, size_t k, double *const *const out)
{
    size_t const s = k / 2;

    for (size_t d = 0; d < s; d++)
    {
        // Extend sides.
        for (size_t j = s - d; j < n + d; j++)
        {
            out[s - d - 1][j] = out[s - d][j];
            out[n + d + 1][j] = out[n + d][j];
        }

        for (size_t i = s - d; i < n + d; i++)
        {
            out[i][s - d - 1] = out[i][s - d];
            out[i][n + d + 1] = out[i][n + d];
        }

        // Fill corners.
        out[s - d - 1][s - d - 1] =
            (out[s - d - 1][s - d] + out[s - d][s - d - 1]) / 2.0;
        out[s - d - 1][n + d + 1] =
            (out[s - d - 1][n + d] + out[s - d][n + d + 1]) / 2.0;
        out[n + d + 1][s - d - 1] =
            (out[n + d + 1][s - d] + out[n + d][s - d - 1]) / 2.0;
        out[n + d + 1][n + d + 1] =
            (out[n + d + 1][n + d] + out[n + d][n + d + 1]) / 2.0;
    }
}

void pad_zero(size_t n, size_t k, double *const *out)
{
    size_t const s = k / 2;

    for (size_t d = 0; d < s; d++)
    {
        for (size_t j = s - d; j < n + d; j++)
        {
            out[s - d - 1][j] = out[n + d + 1][j] = 0.0;
        }

        for (size_t i = s - d; i < n + d; i++)
        {
            out[i][s - d - 1] = out[i][n + d + 1] = 0.0;
        }

        out[s - d - 1][s - d - 1] = out[s - d - 1][n + d + 1] =
            out[n + d + 1][s - d - 1] = out[n + d + 1][n + d + 1] = 0.0;
    }
}

void conv_layer_pass(
    conv_layer const *const x, double *const *const in,
    double *const *const out)
{
    convolve(x->n, x->k, in, out, x->kernel);
    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->n; j++)
        {
            out[i][j] = relu(out[i][j] + x->bias);
        }
    }
    pad_avg(x->n, x->k, out);
}

void input_layer_pass(
    input_layer const *const x, example const *const e,
    double *const *const out, size_t padding)
{
    for (size_t i = 0; i < SQUARE(x->n); i++)
    {
        out[i / x->n + padding][i % x->n + padding] = e->image[i];
    }
    pad_avg(x->n, padding, out);
}

void fc_layer_pass(
    fc_layer const *const x, double *const in, double *const out)
{
    mul_matrix_vector(x->n, x->m, in, x->weight, out);
    for (size_t i = 0; i < x->n; i++)
    {
        out[i] += x->bias[i];
    }
    softmax(x->n, out);
}

void input_layer_read(input_layer *const x, FILE *const net_f)
{
    fscanf(net_f, "%zu", &x->n);
    input_layer_init(x, x->n);
}

void conv_layer_read(conv_layer *const x, FILE *const net_f)
{
    fscanf(net_f, "%zu %zu %lg", &x->n, &x->k, &x->bias);
    conv_layer_init(x, x->n, x->k);

    for (size_t i = 0; i < x->k; i++)
    {
        for (size_t j = 0; j < x->k; j++)
        {
            fscanf(net_f, "%lg", &x->kernel[i][j]);
        }
    }
}

void fc_layer_read(fc_layer *const x, FILE *const net_f)
{
    fscanf(net_f, "%zu %zu", &x->n, &x->m);
    fc_layer_init(x, x->n, x->m);

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->m; j++)
        {
            fscanf(net_f, "%lg", &x->weight[i][j]);
        }
    }

    for (size_t i = 0; i < x->n; i++)
    {
        fscanf(net_f, "%lg", &x->bias[i]);
    }
}

void input_layer_save(input_layer const *const x, FILE *const net_f)
{
    fprintf(net_f, "%zu\n", x->n);
}

void conv_layer_save(conv_layer const *const x, FILE *const net_f)
{
    fprintf(net_f, "%zu %zu\n%lg\n", x->n, x->k, x->bias);

    for (size_t i = 0; i < x->k; i++)
    {
        for (size_t j = 0; j < x->k; j++)
        {
            fprintf(net_f, "%lg ", x->kernel[i][j]);
        }
    }
    fputc('\n', net_f);
}

void fc_layer_save(fc_layer const *const x, FILE *const net_f)
{
    fprintf(net_f, "%zu %zu\n", x->n, x->m);

    for (size_t i = 0; i < x->n; i++)
    {
        for (size_t j = 0; j < x->m; j++)
        {
            fprintf(net_f, "%lg ", x->weight[i][j]);
        }
    }
    fputc('\n', net_f);

    for (size_t i = 0; i < x->n; i++)
    {
        fprintf(net_f, "%lg ", x->bias[i]);
    }
    fputc('\n', net_f);
}

void vectorize_matrix(
    size_t n, size_t m, double *const *const matrix, double *const vector)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            vector[i * m + j] = matrix[i][j];
        }
    }
}
