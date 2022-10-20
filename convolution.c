#include <complex.h>
#include <math.h>
#include <assert.h>

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

void bit_reverse(size_t n, size_t lgn, complex double *const vec)
{
    bool swapped[n];
    memset(swapped, 0, n * sizeof(bool));

    for (size_t i = 0; i < n; i++)
    {
        if (!swapped[i])
        {
            size_t rev_i = 0;
            for (size_t j = 0; j < lgn; j++)
            {
                rev_i |= (i & (1 << j) ? 1 : 0) << (lgn - j - 1);
            }

            swap(vec + i, vec + rev_i);
            swapped[i] = 1;
            swapped[rev_i] = 1;
        }
    }
}

// Computes the Discrete Fourier Transform of vec in place. n must be a power
// of 2.
void fft(size_t n, complex double *const vec)
{
    size_t lgn = 0;
    while (1 << lgn < n)
        lgn++;

    assert(n == 1 << lgn);

    bit_reverse(n, lgn, vec);

    for (size_t s = 1; s <= lgn; s++)
    {
        size_t m = 1 << s;
        complex double omega_m = exp(2.0 * M_PI * I / (complex double)m);

        for (size_t i = 0; i < n; i += s)
        {
            complex double omega = 1.0;
            for (size_t j = i; j < i + m / 2; j++)
            {
                complex double u = vec[j], v = omega * vec[j + m / 2];
                vec[j] = u + v;
                vec[j + m / 2] = u - v;
                omega *= omega_m;
            }
        }
    }
}

void ifft(size_t n, complex double *const vec)
{
    size_t lgn = 0;
    while (1 << lgn < n)
        lgn++;

    assert(n == 1 << lgn);

    bit_reverse(n, lgn, vec);

    for (size_t s = 1; s <= lgn; s++)
    {
        size_t m = 1 << s;
        complex double omega_m = exp(-2.0 * M_PI * I / (complex double)m);

        for (size_t i = 0; i < n; i += s)
        {
            complex double omega = 1.0;
            for (size_t j = i; j < i + m / 2; j++)
            {
                complex double u = vec[j], v = omega * vec[j + m / 2];
                vec[j] = u + v;
                vec[j + m / 2] = u - v;
                omega *= omega_m;
            }
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        vec[i] /= (complex double)n;
    }
}

// The two-dimensional DFT, both n and m must be powers of 2.
complex double **fft_2d(size_t n, size_t m, double *const *const matrix)
{
    complex double **transpose = malloc(m * sizeof(complex double *));

    for (size_t i = 0; i < m; i++)
    {
        transpose[i] = malloc(n * sizeof(complex double));
        for (size_t j = 0; j < n; j++)
        {
            transpose[i][j] = matrix[j][i];
        }
        fft(n, transpose[i]);
    }

    complex double **res = malloc(n * sizeof(complex double *));

    for (size_t i = 0; i < n; i++)
    {
        res[i] = malloc(m * sizeof(complex double));
        for (size_t j = 0; j < m; j++)
        {
            res[i][j] = transpose[j][i];
        }
        fft(m, res[i]);
    }

    matrix_free(m, transpose);
    return res;
}

complex double **ifft_2d(size_t n, size_t m, complex double *const *const matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        ifft(m, matrix[i]);
    }

    complex double **transpose = malloc(m * sizeof(complex double *));

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; i < n; i++)
        {
            transpose[i][j] = matrix[j][i];
        }
        ifft(n, transpose[i]);
    }

    complex double **res = malloc(n * sizeof(complex double *));

    for (size_t i = 0; i < n; i++)
    {
        res[i] = malloc(m * sizeof(complex double));
        for (size_t j = 0; j < m; j++)
        {
            res[i][j] = transpose[j][i];
        }
    }

    matrix_free(m, transpose);
    return res;
}

void convolve_fft(
    size_t n, size_t k, double *const *const in,
    double *const *const out, double *const *const kernel)
{
}
