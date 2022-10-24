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

void bit_reverse_rows(
    size_t n1, size_t n2, size_t lgn2, complex double *const *const matrix)
{
    bool swapped[n1][n2];
    memset(swapped, 0, n1 * n2 * sizeof(bool));

    for (size_t i = 0; i < n1; i++)
    {
        for (size_t j = 0; j < n2; j++)
        {
            if (!swapped[i][j])
            {
                size_t rev_j = 0;
                for (size_t k = 0; k < lgn2; k++)
                {
                    rev_j |= (j & (1 << k) ? 1 : 0) << (lgn2 - k - 1);
                }

                swap(matrix[i] + j, matrix[i] + rev_j);
                swapped[i][j] = 1;
                swapped[i][rev_j] = 1;
            }
        }
    }
}

void bit_reverse_cols(
    size_t n1, size_t lgn1, size_t n2, complex double *const *const matrix)
{
    bool swapped[n1][n2];
    memset(swapped, 0, n1 * n2 * sizeof(bool));

    for (size_t j = 0; j < n2; j++)
    {
        for (size_t i = 0; i < n1; i++)
        {
            if (!swapped[i][j])
            {
                size_t rev_i = 0;
                for (size_t k = 0; k < lgn1; k++)
                {
                    rev_i |= (i & (1 << k) ? 1 : 0) << (lgn1 - k - 1);
                }

                swap(matrix[i] + j, matrix[rev_i] + j);
                swapped[i][j] = 1;
                swapped[rev_i][j] = 1;
            }
        }
    }
}

// Computes the two-dimensional discrete fourier transform in place. n1 and n2
// must be powers of 2.
void fft_2d(size_t n1, size_t n2, complex double *const *const matrix)
{
    size_t lgn1 = 0, lgn2 = 0;
    while (1 << lgn1 < n1)
        lgn1++;
    while (1 << lgn2 < n2)
        lgn2++;

    assert(n1 == 1 << lgn1);
    assert(n2 == 1 << lgn2);

    bit_reverse_cols(n1, lgn1, n2, matrix);

    for (size_t j = 0; j < n2; j++)
    {
        for (size_t s = 1; s <= lgn1; s++)
        {
            size_t const m = 1 << s;
            complex double const omega_m =
                cexp((2.0 * M_PI * I) / (complex double)m);

            for (size_t i = 0; i < n1; i += m)
            {
                complex double omega = 1.0;
                for (size_t k = 0; k < m / 2; k++)
                {
                    assert(i + k + m / 2 < n1);

                    complex double u = matrix[i + k][j],
                                   v = omega * matrix[i + k + m / 2][j];
                    matrix[i + k][j] = u + v;
                    matrix[i + k + m / 2][j] = u - v;
                    omega *= omega_m;
                }
            }
        }
    }

    bit_reverse_rows(n1, n2, lgn2, matrix);

    for (size_t i = 0; i < n1; i++)
    {
        for (size_t s = 1; s <= lgn2; s++)
        {
            size_t const m = 1 << s;
            complex double const omega_m =
                cexp((2.0 * M_PI * I) / (complex double)m);

            for (size_t j = 0; j < n2; j += m)
            {
                complex double omega = 1.0;
                for (size_t k = 0; k < m / 2; k++)
                {
                    assert(j + k + m / 2 < n2);

                    complex double u = matrix[i][j + k],
                                   v = omega * matrix[i][j + k + m / 2];
                    matrix[i][j + k] = u + v;
                    matrix[i][j + k + m / 2] = u - v;
                    omega *= omega_m;
                }
            }
        }
    }
}

void ifft_2d(size_t n1, size_t n2, complex double *const *const matrix)
{
    size_t lgn1 = 0, lgn2 = 0;
    while (1 << lgn1 < n1)
        lgn1++;
    while (1 << lgn2 < n2)
        lgn2++;

    assert(n1 == 1 << lgn1);
    assert(n2 == 1 << lgn2);

    bit_reverse_cols(n1, lgn1, n2, matrix);

    for (size_t j = 0; j < n2; j++)
    {
        for (size_t s = 1; s <= lgn1; s++)
        {
            size_t const m = 1 << s;
            complex double const omega_m =
                cexp((-2.0 * M_PI * I) / (complex double)m);

            for (size_t i = 0; i < n1; i += m)
            {
                complex double omega = 1.0;
                for (size_t k = 0; k < m / 2; k++)
                {
                    assert(i + k + m / 2 < n1);

                    complex double u = matrix[i + k][j],
                                   v = omega * matrix[i + k + m / 2][j];
                    matrix[i + k][j] = u + v;
                    matrix[i + k + m / 2][j] = u - v;
                    omega *= omega_m;
                }
            }
        }

        for (size_t i = 0; i < n1; i++)
        {
            matrix[i][j] /= (complex double)n1;
        }
    }

    bit_reverse_rows(n1, n2, lgn2, matrix);

    for (size_t i = 0; i < n1; i++)
    {
        for (size_t s = 1; s <= lgn2; s++)
        {
            size_t const m = 1 << s;
            complex double const omega_m =
                cexp((-2.0 * M_PI * I) / (complex double)m);

            for (size_t j = 0; j < n2; j += m)
            {
                complex double omega = 1.0;
                for (size_t k = 0; k < m / 2; k++)
                {
                    assert(j + k + m / 2 < n2);

                    complex double u = matrix[i][j + k],
                                   v = omega * matrix[i][j + k + m / 2];
                    matrix[i][j + k] = u + v;
                    matrix[i][j + k + m / 2] = u - v;
                    omega *= omega_m;
                }
            }
        }

        for (size_t j = 0; j < n2; j++)
        {
            matrix[i][j] /= (complex double)n2;
        }
    }
}

complex double **cyclic_sift(
    size_t k, size_t target_n, double *const *const kernel)
{
    complex double **shifted = malloc(target_n * sizeof(complex double *));
    for (size_t i = 0; i < target_n; i++)
    {
        shifted[i] = calloc(target_n, sizeof(complex double));
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

inline size_t next_pow2(size_t x)
{
    size_t lg_y = 0;
    while (1 << lg_y < x)
        lg_y++;
    return 1 << lg_y;
}

// Returns the convolution of in and kernel, padded to size target_n and in
// complex numbers. This function avoids duplication in the convolution
// function with / without offset.
complex double **get_convolution(
    size_t n, size_t k, double *const *const in, double *const *const kernel)
{
    size_t const target_n = next_pow2(n + k - 1);

    complex double **shifted_kernel = cyclic_sift(k, target_n, kernel);
    complex double **padded_in = malloc(target_n * sizeof(complex double *));

    for (size_t i = 0; i < target_n; i++)
    {
        padded_in[i] = calloc(target_n, sizeof(complex double));
    }
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            padded_in[i][j] = in[i][j];
        }
    }

    fft_2d(target_n, target_n, padded_in);
    fft_2d(target_n, target_n, shifted_kernel);

    for (size_t i = 0; i < target_n; i++)
    {
        for (size_t j = 0; j < target_n; j++)
        {
            padded_in[i][j] *= shifted_kernel[i][j];
        }
    }

    matrix_free(target_n, shifted_kernel);
    ifft_2d(target_n, target_n, padded_in);

    return padded_in;
}

void convolve_fft(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool const additive)
{
    complex double **res = get_convolution(n, k, in, kernel);

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

    matrix_free(next_pow2(n + k - 1), res);
}

void convolve_fft_offset(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, size_t margin, bool const additive)
{
    complex double **res = get_convolution(n, k, in, kernel);
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

    matrix_free(next_pow2(n + k - 1), res);
}