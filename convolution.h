#ifndef CONVOLUTION_H
#define CONVOLUTION_H 1

#include <stddef.h>
#include <stdbool.h>

#include "def.h"

void convolve(
    size_t n, size_t k, double *const *const restrict in,
    double *const *const restrict out, double *const *const restrict kernel,
    bool additive);

// Lays out the output with a padding of k / 2 in all directions.
void convolve_pad(
    size_t n, size_t k, double *const *const restrict in,
    double *const *const restrict out, double *const *const restrict kernel,
    bool additive);

#endif