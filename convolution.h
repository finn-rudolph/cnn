#ifndef CONVOLUTION_H
#define CONVOLUTION_H 1

#include <stddef.h>
#include <stdbool.h>

void convolve(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool add_to_out);

#endif