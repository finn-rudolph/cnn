#ifndef UTIL_H
#define UTIL_H 1

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>

#include "def.h"

#define square(x) (x * x)
#define min(x, y) (x < y ? x : y)

static inline double rand_double(double a, double b)
{
    return a + (rand() / (RAND_MAX / (b - a)));
}

static inline void uint8_matrix_free(size_t n, uint8_t **const matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

static inline void double_matrix_free(size_t n, double **const matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

#define matrix_free(n, matrix)    \
    _Generic(matrix,              \
             uint8_t * *          \
             : uint8_matrix_free, \
               double **          \
             : double_matrix_free)(n, matrix)

static inline uint8_t **uint8_matrix_alloc(size_t n, size_t m)
{
    uint8_t **matrix = malloc(n * sizeof(uint8_t *));
    for (size_t i = 0; i < n; i++)
    {
        matrix[i] = malloc(m * sizeof(uint8_t));
    }
    return matrix;
}

static inline double **double_matrix_alloc(size_t n, size_t m)
{
    double **matrix = malloc(n * sizeof(double *));
    for (size_t i = 0; i < n; i++)
    {
        matrix[i] = malloc(m * sizeof(double));
    }
    return matrix;
}

static inline void double_matrix_copy(
    size_t n, size_t m, double *const *const in, double *const *const out)
{
    for (size_t i = 0; i < n; i++)
    {
        memcpy(out[i], in[i], m * sizeof(double));
    }
}

#define matrix_copy(n, m, in, out) \
    _Generic(in,                   \
             double *const *       \
             : double_matrix_copy, \
               double **           \
             : double_matrix_print)(n, m, in, out)

static inline void double_matrix_print(
    size_t n, size_t m, double *const *const matrix, FILE *stream)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            fprintf(stream, "%lg ", matrix[i][j]);
        }
        fputc('\n', stream);
    }
    fputc('\n', stream);
}

#define matrix_print(n, m, matrix, stream) \
    _Generic(matrix,                       \
             double *const *               \
             : double_matrix_print,        \
               double **                   \
             : double_matrix_print)(n, m, matrix, stream)

static inline void double_vector_print(
    size_t n, double *const vector, FILE *stream)
{
    for (size_t i = 0; i < n; i++)
    {
        fprintf(stream, "%lg ", vector[i]);
    }
    fputc('\n', stream);
}

static inline void uint8_vector_print(
    size_t n, uint8_t *const vector, FILE *stream)
{
    for (size_t i = 0; i < n; i++)
    {
        fprintf(stream, "%hhu ", vector[i]);
    }
    fputc('\n', stream);
}

#define vector_print(n, vector, stream) \
    _Generic(vector,                    \
             double *                   \
             : double_vector_print,     \
               uint8_t *                \
             : uint8_vector_print)(n, vector, stream)

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

#endif