#include "convolution.h"
#include "util.h"

// n must be the size of the input matrix including possible padding.
void convolve(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool add_to_out)
{
    size_t const s = k / 2;
    for (size_t i = s; i < n - s; i++)
    {
        for (size_t j = s; j < n - s; j++)
        {
            if (!add_to_out)
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