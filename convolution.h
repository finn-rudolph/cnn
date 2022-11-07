#ifndef CONVOLUTION_H
#define CONVOLUTION_H 1

#include <stddef.h>
#include <stdbool.h>

// Stores a matrix of the same size as the input matrix in out. The kernel is
// first centered in the top left corner and cut off when necessary. k must be
// odd.
void convolve(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool const additive);

// Stores a matrix of size 2 * margin + 1 in out. margin specifies the maximum
// amount the kernel may hang out of the input matrix.
void convolve_offset(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, size_t margin, bool const additive);

void convolve_fft(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, bool const additive);

void convolve_fft_offset(
    size_t n, size_t k, double *const *const in, double *const *const out,
    double *const *const kernel, size_t margin, bool const additive);

#endif