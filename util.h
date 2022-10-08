#ifndef UTIL_H
#define UTIL_H 1

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "def.h"

#define square(x) (x * x)
#define min(x, y) (x < y ? x : y)

static inline double rand_double(double a, double b)
{
    return a + (rand() / (RAND_MAX / (b - a)));
}

static inline void destroy_matrix_uint8(size_t n, uint8_t **const matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

static inline void destroy_matrix_double(size_t n, double **const matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

#define destroy_matrix(n, matrix)    \
    _Generic(matrix,                 \
             uint8_t * *             \
             : destroy_matrix_uint8, \
               double **             \
             : destroy_matrix_double)(n, matrix)

static inline void swap_dp(double **x, double **y)
{
    double *z = *x;
    *x = *y;
    *y = z;
}

static inline void swap_dpp(double ***x, double ***y)
{
    double **z = *x;
    *x = *y;
    *y = z;
}

// x and y must be pointers to the base type.
#define swap(x, y)        \
    _Generic(x,           \
             double **    \
             : swap_dp,   \
               double *** \
             : swap_dpp)(x, y)

static inline void rev_uint16(uint16_t *x)
{
    uint16_t y = (((*x & 0x00FF) << 8) |
                  ((*x & 0xFF00) >> 8));
    *x = y;
}

static inline void rev_uint32(uint32_t *x)
{
    uint32_t y = (((*x & 0x000000FF) << 24) |
                  ((*x & 0x0000FF00) << 8) |
                  ((*x & 0x00FF0000) >> 8) |
                  ((*x & 0xFF000000) >> 24));
    *x = y;
}

#define rev_int(x)         \
    _Generic(x,            \
             uint16_t *    \
             : rev_uint16, \
               uint32_t *  \
             : rev_uint32)(x)

// in must be a row vector of length m.
static inline void mul_matrix_vector(
    size_t n, size_t m, double const *const in, double *const *const matrix,
    double *const out)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = 0.0;
        for (size_t j = 0; j < m; j++)
        {
            out[i] += in[j] * matrix[i][j];
        }
    }
}

static inline double **flip_kernel(size_t k, double *const *const kernel)
{
    double **flipped = malloc(k * sizeof(double *));
    for (size_t i = 0; i < k; i++)
    {
        flipped[i] = malloc(k * sizeof(double));
    }

    for (size_t i = 0; i < k; i++)
    {
        for (size_t j = 0; j < k; j++)
        {
            flipped[i][j] = kernel[k - i - 1][k - j - 1];
        }
    }

    return flipped;
}

// Suffix _d means derivative. Prefix v means the function uses a vector, not
// only a scalar.

static inline double relu(double x)
{
    return x > 0.0 ? x : 0.0;
}

static inline double relu_d(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}

static inline double relu_smooth(double x)
{
    return log1p(1.0 + exp(x));
}

static inline double relu_smooth_d(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline void vrelu(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = x[i] > 0.0 ? x[i] : 0.0;
    }
}

static inline void vrelu_d(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = x[i] > 0.0 ? 1.0 : 0.0;
    }
}

static inline void vrelu_smooth(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = log1p(1.0 + exp(x[i]));
    }
}

static inline void vrelu_smooth_d(size_t n, double *x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1.0 / (1.0 + exp(-x[i]));
    }
}

static inline void softmax(size_t n, double *const x)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += exp(x[i]);
    }
    for (size_t i = 0; i < n; i++)
    {
        x[i] = exp(x[i]) / sum;
    }
}

static inline void videntity(size_t n, double *const x)
{
}

#endif