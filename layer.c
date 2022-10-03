#include "layer.h"
#include "convolution.h"
#include "util.h"
#include "file_io.h"

void conv_layer_pass(
    conv_layer const *const x, double const *const *const in,
    double *const *const out)
{
    convolve(x->n, x->k, in, out, x->kernel);
    for (size_t i = 0; i < x->n; i++)
        for (size_t j = 0; j < x->n; j++)
            out[i][j] = relu(out[i][j] + x->bias);
    pad_avg(x->n, x->k, out);
}

void input_layer_pass(
    input_layer const *const x, example const *const e,
    double *const *const out, size_t padding)
{
    for (size_t i = 0; i < SQUARE(x->n); i++)
        out[i / x->n + padding][i % x->n + padding] = e->image[i];
    pad_avg(x->n, padding, out);
}

void fc_layer_pass(
    fc_layer const *const x, double const *const in, double *const out)
{
    mul_matrix_vector(x->n, x->m, in, x->weight, out);
    for (size_t i = 0; i < x->n; i++)
        out[i] += x->bias[i];
    softmax(x->weight, out);
}

void vectorize_matrix(
    size_t n, size_t m, double const *const *const matrix, double *const vector)
{
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            vector[i * m + j] = matrix[i][j];
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
            out[s - d - 1][j] = out[n + d + 1][j] = 0.0;

        for (size_t i = s - d; i < n + d; i++)
            out[i][s - d - 1] = out[i][n + d + 1] = 0.0;

        out[s - d - 1][s - d - 1] =
            out[s - d - 1][n + d + 1] =
                out[n + d + 1][s - d - 1] =
                    out[n + d + 1][n + d + 1] = 0.0;
    }
}
