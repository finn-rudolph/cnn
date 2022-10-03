#ifndef CONVOLUTION_H
#define CONVOLUTION_H 1

#include <stddef.h>

void convolve(
    size_t n, size_t m, double *const *const in, double *const *const out,
    double *const *const kernel);

#endif