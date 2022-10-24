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

    // i, j: output coordinates; a, b: input coordinates
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

void bit_reverse(size_t n, size_t lgn, complex double *const vector)
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

            swap(vector + i, vector + rev_i);
            swapped[i] = 1;
            swapped[rev_i] = 1;
        }
    }
}

// Computes the Discrete Fourier Transform of the input vector in place. n must
// be a power of 2.
void fft(size_t n, complex double *const vector)
{
    size_t lgn = 0;
    while (1 << lgn < n)
        lgn++;

    assert(n == 1 << lgn);
    bit_reverse(n, lgn, vector);

    for (size_t s = 1; s <= lgn; s++)
    {
        size_t const m = 1 << s;
        complex double const omega_m =
            cexp((2.0 * M_PI * I) / (complex double)m);

        for (size_t i = 0; i < n; i += m)
        {
            complex double omega = 1.0;
            for (size_t j = 0; j < m / 2; j++)
            {
                assert(i + j + m / 2 < n);

                complex double u = vector[i + j],
                               v = omega * vector[i + j + m / 2];
                vector[i + j] = u + v;
                vector[i + j + m / 2] = u - v;
                omega *= omega_m;
            }
        }
    }
}

void ifft(size_t n, complex double *const vector)
{
    size_t lgn = 0;
    while (1 << lgn < n)
        lgn++;

    assert(n == 1 << lgn);
    bit_reverse(n, lgn, vector);

    for (size_t s = 1; s <= lgn; s++)
    {
        size_t const m = 1 << s;
        complex double const omega_m =
            cexp((-2.0 * M_PI * I) / (complex double)m);

        for (size_t i = 0; i < n; i += m)
        {
            complex double omega = 1.0;
            for (size_t j = 0; j < m / 2; j++)
            {
                complex double u = vector[i + j],
                               v = omega * vector[i + j + m / 2];
                vector[i + j] = u + v;
                vector[i + j + m / 2] = u - v;
                omega *= omega_m;
            }
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        vector[i] /= (complex double)n;
    }
}

// n is the size of the input matrix, target_n the size to which it shall be
// padded. This allows passing smaller matrices without having to pad them.
complex double **fft_2d(size_t n, size_t target_n, double *const *const matrix)
{
    complex double **transpose = malloc(target_n * sizeof(complex double *));

    for (size_t i = 0; i < target_n; i++)
    {
        transpose[i] = calloc(target_n, sizeof(complex double));
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            transpose[i][j] = matrix[j][i];
        }

        // Only doing the FFT on the nonzero rows is fine, as the result of the
        // other rows would just be a null-vector.
        fft(target_n, transpose[i]);
    }

    complex double **res = malloc(target_n * sizeof(complex double *));

    for (size_t i = 0; i < target_n; i++)
    {
        res[i] = malloc(target_n * sizeof(complex double));
        for (size_t j = 0; j < target_n; j++)
        {
            res[i][j] = transpose[j][i];
        }
        fft(target_n, res[i]);
    }

    matrix_free(target_n, transpose);
    return res;
}

complex double **ifft_2d(size_t n, complex double *const *const matrix)
{
    complex double **transpose = malloc(n * sizeof(complex double *));

    for (size_t i = 0; i < n; i++)
    {
        transpose[i] = calloc(n, sizeof(complex double));
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            transpose[i][j] = matrix[j][i];
        }
        ifft(n, transpose[i]);
    }

    complex double **res = malloc(n * sizeof(complex double *));

    for (size_t i = 0; i < n; i++)
    {
        res[i] = malloc(n * sizeof(complex double));
        for (size_t j = 0; j < n; j++)
        {
            res[i][j] = transpose[j][i];
        }
        ifft(n, res[i]);
    }

    matrix_free(n, transpose);
    return res;
}

double **cyclic_sift(size_t k, size_t target_n, double *const *const kernel)
{
    double **shifted = malloc(target_n * sizeof(double *));
    for (size_t i = 0; i < target_n; i++)
    {
        shifted[i] = calloc(target_n, sizeof(double));
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
            shifted[target_n - s + i][j - s] = kernel[i][j];
        }
    }

    for (size_t i = s; i < k; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            shifted[i - s][target_n - s + j] = kernel[i][j];
        }
    }

    for (size_t i = 0; i < s; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            shifted[target_n - s + i][target_n - s + j] = kernel[i][j];
        }
    }

    return shifted;
}

// Returns the convolution of in and kernel, padded to size target_n and in
// complex numbers. This function's purpose is avoiding code duplication in
// convolve_fft and convolve_fft_offset.
complex double **get_convolution(
    size_t n, size_t target_n, size_t k, double *const *const in,
    double *const *const kernel)
{
    double **shifted_kernel = cyclic_sift(k, target_n, kernel);

    complex double **in_dft = fft_2d(n, target_n, in),
                   **kernel_dft = fft_2d(target_n, target_n, shifted_kernel);

    matrix_free(target_n, shifted_kernel);

    for (size_t i = 0; i < target_n; i++)
    {
        for (size_t j = 0; j < target_n; j++)
        {
            in_dft[i][j] *= kernel_dft[i][j];
        }
    }

    matrix_free(target_n, kernel_dft);
    complex double **res = ifft_2d(target_n, in_dft);
    matrix_free(target_n, in_dft);

    return res;
}

void convolve_fft(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool const additive)
{
    size_t lg_target_n = 0;
    while (1 << lg_target_n < n + k - 1)
        lg_target_n++;
    size_t target_n = 1 << lg_target_n;

    complex double **res = get_convolution(n, target_n, k, in, kernel);

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

    matrix_free(target_n, res);
}

void convolve_fft_offset(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, size_t margin, bool const additive)
{
    size_t lg_target_n = 0;
    while (1 << lg_target_n < n + k - 1)
        lg_target_n++;
    size_t target_n = 1 << lg_target_n;

    complex double **res = get_convolution(n, target_n, k, in, kernel);
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

    matrix_free(target_n, res);
}