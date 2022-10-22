#include <complex.h>
#include <math.h>
#include <assert.h>

#include "convolution.h"
#include "util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void convolve(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool const additive)
{
    assert(k & 1);
    ptrdiff_t s = k / 2;

    // i, j: output coordinates; a, b: current coordinates in the input
    for (ptrdiff_t i = 0; i < n; i++)
    {
        for (ptrdiff_t j = 0; j < n; j++)
        {
            if (!additive)
                out[i][j] = 0.0;

            for (ptrdiff_t a = max(0, i - s); a < i + s + 1 && a < n; a++)
            {
                for (ptrdiff_t b = max(0, j - s); b < j + s + 1 && b < n; b++)
                {
                    out[i][j] += in[a][b] * kernel[a - i + s][b - j + s];
                }
            }
        }
    }
}

void convolve_offset(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, size_t margin, bool const additive)
{
    ptrdiff_t s = (k - 1) / 2;

    for (ptrdiff_t i = 0; i < 2 * margin + 1; i++)
    {
        for (ptrdiff_t j = 0; j < 2 * margin + 1; j++)
        {
            if (!additive)
                out[i][j] = 0.0;

            for (ptrdiff_t a = max(0, i - s); a < i + k - s && a < n; a++)
            {
                for (ptrdiff_t b = max(0, j - s); b < j + k - s && b < n; b++)
                {
                    out[i][j] += in[a][b] * kernel[a - i + s][b - j + s];
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
        size_t const m = 1 << s;
        complex double const omega_m = cexp((2.0 * M_PI * I) / (complex double)m);

        for (size_t i = 0; i < n; i += m)
        {
            complex double omega = 1.0;
            for (size_t j = 0; j < m / 2; j++)
            {
                assert(i + j + m / 2 < n);

                complex double u = vec[i + j], v = omega * vec[i + j + m / 2];
                vec[i + j] = u + v;
                vec[i + j + m / 2] = u - v;
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
        size_t const m = 1 << s;
        complex double const omega_m = cexp((-2.0 * M_PI * I) / (complex double)m);

        for (size_t i = 0; i < n; i += m)
        {
            complex double omega = 1.0;
            for (size_t j = 0; j < m / 2; j++)
            {
                complex double u = vec[i + j], v = omega * vec[i + j + m / 2];
                vec[i + j] = u + v;
                vec[i + j + m / 2] = u - v;
                omega *= omega_m;
            }
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        vec[i] /= (complex double)n;
    }
}

// n is the size of the input matrix, m the size to which it must be padded.
complex double **fft_2d(size_t n, size_t m, double *const *const matrix)
{
    complex double **transpose = malloc(m * sizeof(complex double *));

    for (size_t i = 0; i < m; i++)
    {
        transpose[i] = calloc(m, sizeof(complex double));
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            transpose[i][j] = matrix[j][i];
        }

        // Only doing the FFT on the nonzero rows is fine, as the result of the
        // other rows would just be a null-vector.
        fft(m, transpose[i]);
    }

    complex double **res = malloc(m * sizeof(complex double *));

    for (size_t i = 0; i < m; i++)
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

complex double **ifft_2d(size_t m, complex double *const *const matrix)
{
    complex double **transpose = malloc(m * sizeof(complex double *));

    for (size_t i = 0; i < m; i++)
    {
        transpose[i] = calloc(m, sizeof(complex double));
    }

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            transpose[i][j] = matrix[j][i];
        }
        ifft(m, transpose[i]);
    }

    complex double **res = malloc(m * sizeof(complex double *));

    for (size_t i = 0; i < m; i++)
    {
        res[i] = malloc(m * sizeof(complex double));
        for (size_t j = 0; j < m; j++)
        {
            res[i][j] = transpose[j][i];
        }
        ifft(m, res[i]);
    }

    matrix_free(m, transpose);
    return res;
}

double **cyclic_sift(size_t k, size_t m, double *const *const kernel)
{
    double **shifted = malloc(m * sizeof(double *));
    for (size_t i = 0; i < m; i++)
    {
        shifted[i] = calloc(m, sizeof(double));
    }

    // The center of the kernel is at (s, s).
    size_t const s = (k - 1) / 2;

    for (size_t i = s; i < k; i++)
    {
        for (size_t j = s; j < k; j++)
        {
            shifted[i - s][j - s] = kernel[i][j];
        }
    }

    for (size_t i = 0; i < s; i++)
    {
        for (size_t j = s; j < k; j++)
        {
            shifted[m - s + i][j - s] = kernel[i][j];
        }
    }

    for (size_t i = s; i < k; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            shifted[i - s][m - s + j] = kernel[i][j];
        }
    }

    for (size_t i = 0; i < s; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            shifted[m - s + i][m - s + j] = kernel[i][j];
        }
    }

    return shifted;
}

// Returns the convolution of in and kernel, padded to size m and in complex
// numbers. This function's purpose is avoiding code duplication in convolve_fft
// and convolve_fft_offset.
complex double **get_convolution(
    size_t n, size_t m, size_t k, double *const *const in,
    double *const *const kernel)
{
    double **shifted_kernel = cyclic_sift(k, m, kernel);

    complex double **in_dft = fft_2d(n, m, in),
                   **kernel_dft = fft_2d(m, m, shifted_kernel);

    matrix_free(m, shifted_kernel);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            in_dft[i][j] *= kernel_dft[i][j];
        }
    }

    matrix_free(m, kernel_dft);
    complex double **res = ifft_2d(m, in_dft);
    matrix_free(m, in_dft);

    return res;
}

void convolve_fft(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool const additive)
{
    size_t lgm = 0;
    while (1 << lgm < n + k - 1)
        lgm++;
    size_t m = 1 << lgm;

    complex double **res = get_convolution(n, m, k, in, kernel);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (additive)
                out[i][j] += creal(res[i][j]);
            else
                out[i][j] = creal(res[i][j]);
        }
    }

    matrix_free(m, res);
}

void convolve_fft_offset(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, size_t margin, bool const additive)
{
    size_t lgm = 0;
    while (1 << lgm < n + k - 1)
        lgm++;
    size_t m = 1 << lgm;

    complex double **res = get_convolution(n, m, k, in, kernel);
    size_t const shift = (k - 1) / 2 - margin;

    for (size_t i = 0; i < 2 * margin + 1; i++)
    {
        for (size_t j = 0; j < 2 * margin + 1; j++)
        {
            if (additive)
                out[i][j] += creal(res[i + shift][j + shift]);
            else
                out[i][j] = creal(res[i + shift][j + shift]);
        }
    }

    matrix_free(m, res);
}