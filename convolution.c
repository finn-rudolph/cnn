#include <complex.h>
#include <math.h>

#include "convolution.h"
#include "util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void convolve(
    size_t n, size_t k, double *const *const in,
    double *const *const out, double *const *const kernel,
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
    size_t n, size_t k, double *const *const in,
    double *const *const out, double *const *const kernel,
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

void bit_reverse(
    size_t n, size_t lgn, double *const vec, complex double *const out)
{
    for (size_t i = 0; i < n; i++)
    {
        size_t rev_i = 0;
        for (size_t j = 0; j < lgn; j++)
        {
            rev_i |= (i & (1 << j) ? 1 : 0) << (lgn - j - 1);
        }
        out[rev_i] = vec[i];
    }
}

// Returns the Discrete Fourier Transform of vec, where n must be a power of 2.
complex double *fft(size_t n, double *const vec)
{
    size_t lgn = 0;
    while (1 << lgn < n)
        lgn++;

    complex double *res = malloc(n * sizeof(complex double));
    bit_reverse(n, lgn, vec, res);

    for (size_t s = 1; s <= lgn; s++)
    {
        size_t l = 1 << s;
        complex double omega_m = exp(2.0 * M_PI * I / (complex double)l);

        for (size_t i = 0; i < n; i += s)
        {
            complex double omega = 1.0;
            for (size_t j = i; j < i + l / 2; j++)
            {
                complex double u = res[j], v = omega * res[j + l / 2];
                res[j] = u + v;
                res[j + l / 2] = u - v;
                omega *= omega_m;
            }
        }
    }

    return res;
}
