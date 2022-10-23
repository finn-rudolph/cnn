#ifndef UTIL_H
#define UTIL_H 1

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <complex.h>

#define square(x) ((x) * (x))
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(x, y) (((x) > (y)) ? (x) : (y))

static inline double rand_double(double a, double b)
{
    return a + (rand() / (RAND_MAX / (b - a)));
}

// Swapping functions.

static inline void swap_cdouble(complex double *x, complex double *y)
{
    complex double z = *x;
    *x = *y;
    *y = z;
}

static inline void swap_doublep(double **x, double **y)
{
    double *z = *x;
    *x = *y;
    *y = z;
}

static inline void swap_doublepp(double ***x, double ***y)
{
    double **z = *x;
    *x = *y;
    *y = z;
}

static inline void swap_uint8(uint8_t *x, uint8_t *y)
{
    uint8_t z = *x;
    *x = *y;
    *y = z;
}

#define swap(x, y)                    \
    _Generic(x,                       \
             complex double *         \
             : swap_cdouble,          \
               double **              \
             : swap_doublep,          \
               double const **        \
             : swap_doublep,          \
               double ***             \
             : swap_doublepp,         \
               double const ***       \
             : swap_doublepp,         \
               double const *const ** \
             : swap_doublepp,         \
               double *const **       \
             : swap_doublepp,         \
               uint8_t *              \
             : swap_uint8)(x, y)

// Matrix utility functions.

static inline double **matrix_alloc(size_t n, size_t m)
{
    double **matrix = malloc(n * sizeof(double *));
    for (size_t i = 0; i < n; i++)
    {
        matrix[i] = malloc(m * sizeof(double));
    }
    return matrix;
}

static inline void matrix_free_double(size_t n, double **matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

static inline void matrix_free_cdouble(size_t n, complex double **matrix)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

#define matrix_free(n, matrix)     \
    _Generic(matrix,               \
             double **             \
             : matrix_free_double, \
               complex double **   \
             : matrix_free_cdouble)(n, matrix)

static inline void matrix_copy(
    size_t n, size_t m, double *const *const matrix,
    double *const *const out)
{
    for (size_t i = 0; i < n; i++)
    {
        memcpy(out[i], matrix[i], m * sizeof(double));
    }
}

static inline void matrix_print(
    size_t n, size_t m, double *const *const matrix, FILE *const stream)
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

static inline void matrix_add(
    size_t n, size_t m, double *const *const matrix, double *const *const out)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            out[i][j] += matrix[i][j];
        }
    }
}

// Vector utility functions.

static inline void vector_print(
    size_t n, double const *const vector, FILE *const stream)
{
    for (size_t i = 0; i < n; i++)
    {
        fprintf(stream, "%lg ", vector[i]);
    }
    fputc('\n', stream);
}

static inline void vector_add(size_t n, double *const vector, double *const out)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] += vector[i];
    }
}

// Multiplies a vector of length m with a matrix of size n x m and stores the
// resulting vector of length n in out.
static inline void vector_mul_matrix(
    size_t n, size_t m, double const *const in,
    double *const *const matrix, double *const out)
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

// Endianess inversion functions.

static inline void rev_uint16(uint16_t *const x)
{
    uint16_t y = (((*x & 0x00FF) << 8) |
                  ((*x & 0xFF00) >> 8));
    *x = y;
}

static inline void rev_uint32(uint32_t *const x)
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

// Activation functions. The actual activation functions are defined for
// vectors, while their derivatives are defined for scalars. This is useful, as
// softmax (counting as an activation function) needs the wohle vector of
// values.

typedef double (*activation_fn)(double x);
typedef void (*vactivation_fn)(size_t n, double *const);

static inline void fn_relu(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = max(0.0, x[i]);
    }
}

static inline double fn_relu_d(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}

static inline void fn_relu_smooth(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = log1p(exp(x[i]));
    }
}

static inline double fn_relu_smooth_d(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline void fn_sigmoid(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1.0 / (1.0 + exp(-x[i]));
    }
}

static inline double fn_sigmoid_d(double x)
{
    return 1.0 / (2.0 + exp(x) + exp(-x));
}

static inline void fn_tanh(size_t n, double *const x)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = tanh(x[i]);
    }
}

static inline double fn_tanh_d(double x)
{
    return 4.0 / (2.0 + exp(2.0 * x) + exp(-2.0 * x));
}

static inline void fn_softmax(size_t n, double *const x)
{
    // Subtract the maximum to avoid overflow.
    double max_val = x[0];
    for (size_t i = 0; i < n; i++)
    {
        max_val = max(max_val, x[i]);
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += exp(x[i] - max_val);
    }
    for (size_t i = 0; i < n; i++)
    {
        x[i] = exp(x[i] - max_val) / sum;
    }
}

#endif