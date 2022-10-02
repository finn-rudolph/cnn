#ifndef CONVOLUTION_H
#define CONVOLUTION_H 1

#include <stddef.h>

void convolve(
    size_t n, size_t m,
    double const *const x, double *const y, double const *const kernel);

#endif