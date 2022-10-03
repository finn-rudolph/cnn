#include "layer.h"
#include "convolution.h"
#include "util.h"

void layer_pass_forward(layer const *const x, layer const *const y,
                        double const *const *const in, double *const *const out)
{
    switch (y->conv.ltype)
    {
    case LTYPE_CONV:
    {
        convolve(x->conv.n, y->conv.k, in, out, y->conv.kernel);
        for (size_t i = 0; i < y->conv.n; i++)
            for (size_t j = 0; j < y->conv.n; j++)
                out[i][j] = relu(out[i][j] + y->conv.bias);
    }
    case LTYPE_FC:
    {
    }
    }
}

// Assumes the output of the former layer is layed out such that a margin of
// half the kernel size just fits in.
void pad_avg(conv_layer const *const y, double **const out)
{
    size_t const s = y->k / 2;

    for (size_t d = 0; d < s; d++)
    {
        // Extend sides.
        for (size_t j = s - d; j < y->n + d; j++)
        {
            out[(s - d - 1)][j] = out[(s - d)][j];
            out[(y->n + d + 1)][j] = out[(y->n + d)][j];
        }

        for (size_t i = s - d; i < y->n + d; i++)
        {
            out[i][s - d - 1] = out[i][s - d];
            out[i][y->n + d + 1] = out[i][y->n + d];
        }

        // Fill corners.
        out[s - d - 1][s - d - 1] =
            (out[s - d - 1][s - d] + out[s - d][s - d - 1]) / 2.0;
        out[s - d - 1][y->n + d + 1] =
            (out[s - d - 1][y->n + d] + out[s - d][y->n + d + 1]) / 2.0;
        out[y->n + d + 1][s - d - 1] =
            (out[y->n + d + 1][s - d] + out[y->n + d][s - d - 1]) / 2.0;
        out[y->n + d + 1][y->n + d + 1] =
            (out[y->n + d + 1][y->n + d] + out[y->n + d][y->n + d + 1]) / 2.0;
    }
}

void pad_zero(conv_layer const *const y, double **out)
{
    size_t const s = y->k / 2;

    for (size_t d = 0; d < s; d++)
    {
        for (size_t j = s - d; j < y->n + d; j++)
            out[s - d - 1][j] = out[y->n + d + 1][j] = 0.0;

        for (size_t i = s - d; i < y->n + d; i++)
            out[i][s - d - 1] = out[i][y->n + d + 1] = 0.0;

        out[s - d - 1][s - d - 1] =
            out[s - d - 1][y->n + d + 1] =
                out[y->n + d + 1][s - d - 1] =
                    out[y->n + d + 1][y->n + d + 1] = 0.0;
    }
}
