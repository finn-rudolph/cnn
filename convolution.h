#ifndef CONVOLUTION_H
#define CONVOLUTION_H 1

#include <stddef.h>
#include <stdbool.h>

// All convolution functions assume, that the data starts at index 0 in both
// dimensions and is layed out continuously.

// Ouputs a matrix of the same size as the input matrix by first centering the
// kernel in the top left corner and cutting off when necessary. k must be odd.
void convolve(
    size_t n, size_t k, double *const *const in,
    double *const *const out, double *const *const kernel,
    bool additive);

// Performs a convolution in the usual sense. The margin is used as a maximum
// margin around the input image, such that the ouput is of size 2 * margin + 1.
void convolve_offset(
    size_t n, size_t k, double *const *const in,
    double *const *const out, double *const *const kernel, size_t margin,
    bool additive);

void convolve_fft(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel);

#endif