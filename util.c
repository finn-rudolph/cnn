#include "util.h"

#include <assert.h>

void mul_matrix_vector(
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