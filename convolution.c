#include "convolution.h"

// Assumes a margin of half the kernel size. The output is layed out with
// the same margin.
void convolve(
    size_t n, size_t m, double const *const *const in,
    double *const *const out, double *const *const kernel)
{
    size_t const s = m / 2;
    for (size_t i = s; i < n - s; i++)
        for (size_t j = s; j < n - s; j++)
        {
            out[i][j] = 0.0;
            for (size_t a = 0; a < m; a++)
                for (size_t b = 0; b < m; b++)
                    out[i][j] += in[i - s + a][j - s + b] * kernel[a][b];
        }
}