#include "convolution.h"
#include "util.h"

void convolve(
    size_t n, size_t k, double *const *const restrict in,
    double *const *const restrict out, double *const *const restrict kernel,
    bool additive)
{
    for (size_t i = 0; i < n - k + 1; i++)
    {
        for (size_t j = 0; j < n - k + 1; j++)
        {
            if (!additive)
                out[i][j] = 0.0;
            for (size_t a = 0; a < k; a++)
            {
                for (size_t b = 0; b < k; b++)
                {
                    out[i][j] += in[i + a][j + b] * kernel[a][b];
                }
            }
        }
    }
}

void convolve_pad(
    size_t n, size_t k, double *const *const restrict in,
    double *const *const restrict out, double *const *const restrict kernel,
    bool additive)
{
    size_t const s = k / 2;
    for (size_t i = s; i < n - s; i++)
    {
        for (size_t j = s; j < n - s; j++)
        {
            if (!additive)
                out[i][j] = 0.0;
            for (size_t a = 0; a < k; a++)
            {
                for (size_t b = 0; b < k; b++)
                {
                    out[i][j] += in[i - s + a][j - s + b] * kernel[a][b];
                }
            }
        }
    }
}
