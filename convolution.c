#include "convolution.h"

// Assumes a margin of half the kernel size. The output is layed out with
// the same margin.
void convolve(
    size_t n, size_t m,
    double const *const x, double *const y, double const *const kernel)
{
    size_t const s = m / 2;
    for (size_t i = s; i < n - s; i++)
        for (size_t j = s; j < n - s; j++)
        {
            y[i * n + j] = 0.0;
            for (size_t a = 0; a < m; a++)
                for (size_t b = 0; b < m; b++)
                    y[i * n + j] += x[(i - s + a) * n + (j - s + b)] *
                                    kernel[a * m + b];
        }
}