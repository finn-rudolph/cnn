#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H 1

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

// Images from a to b (exclusive) are read. An image is represented as a
// one-dimensional array organized by rows.
double **read_images(char const *const image_fname, size_t a, size_t b);

uint8_t *read_labels(char const *const label_fname, size_t a, size_t b);

// Subtracts the mean and divides by the standard deviation. Only use this
// function on mini-batches, not the whole data, to avoid precision loss.
void normalize_mini(size_t t, size_t n1, size_t n2, double *const *const images);

// Divides images into groups of size NORMALIZATION_BATCH_SIZE and normalizes
// them.
void normalize(size_t t, size_t n1, size_t n2, double *const *const images);

#endif