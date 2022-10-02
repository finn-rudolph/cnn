#include "layer.h"
#include "convolution.h"
#include "activations.h"

void layer_forward(layer const *const x, layer const *const y,
                   double *in, double *out)
{
    switch (y->conv.layer_type)
    {
    case TYPE_CONV:
    {
        convolve(x->conv.n, y->conv.m, in, out, y->conv.kernel);
        for (size_t i = 0; i < y->conv.n; i++)
            for (size_t j = 0; j < y->conv.n; j++)
            {
                out[i * y->conv.n + j] += y->conv.bias;
                out[i * y->conv.n + j] = relu(out[i * y->conv.n + j]);
            }
    }
    case TYPE_FC:
    {
    }
    }
}

// Assumes the output of the former layer is layed out such that a margin of
// half the kernel size causes all data to be in a continuous range.
void padding_avg(conv_layer const *const y, double *const out)
{
    size_t const s = y->m / 2, n = y->n + 2 * s;

    for (size_t k = 0; k < s; k++)
    {
        // Extend sides.
        for (size_t j = s - k; j < y->n + k; j++)
        {
            out[(s - k - 1) * n + j] = out[(s - k) * n + j];
            out[(y->n + k + 1) * n + j] = out[(y->n + k) * n + j];
        }

        for (size_t i = s - k; i < y->n + k; i++)
        {
            out[i * n + s - k - 1] = out[i * n + s - k];
            out[i * n + y->n + k + 1] = out[i * n + y->n + k];
        }

        // Fill corners.
        out[(s - k - 1) * n + s - k - 1] = (out[(s - k - 1) * n + s - k] +
                                            out[(s - k) * n + s - k - 1]) /
                                           2.0;
        out[(s - k - 1) * n + y->n + k + 1] = (out[(s - k - 1) * n + y->n + k] +
                                               out[(s - k) * n + y->n + k + 1]) /
                                              2.0;
        out[(y->n + k + 1) * n + s - k - 1] = (out[(y->n + k + 1) * n + s - k] +
                                               out[(y->n + k) * n + s - k - 1]) /
                                              2.0;
        out[(y->n + k + 1) * n + y->n + k + 1] = (out[(y->n + k + 1) * n + y->n + k] +
                                                  out[(y->n + k) * n + y->n + k + 1]) /
                                                 2.0;
    }
}

void padding_zero(conv_layer const *const y, double *out)
{
    size_t const s = y->m / 2, n = y->n + 2 * s;

    for (size_t k = 0; k < s; k++)
    {
        for (size_t j = s - k; j < y->n + k; j++)
            out[(s - k - 1) * n + j] = out[(y->n + k + 1) * n + j] = 0.0;

        for (size_t i = s - k; i < y->n + k; i++)
            out[i * n + s - k - 1] = out[i * n + y->n + k + 1] = 0;

        out[(s - k - 1) * n + s - k - 1] =
            out[(s - k - 1) * n + y->n + k + 1] =
                out[(y->n + k + 1) * n + s - k - 1] =
                    out[(y->n + k + 1) * n + y->n + k + 1] = 0;
    }
}
