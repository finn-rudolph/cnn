#ifndef DATA_IO_H
#define DATA_IO_H 1

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "def.h"

// Images from a to b (exclusive) are read. An image is represented as a
// one-dimensional array organized by rows.
uint8_t **read_images(char const *const image_fname, size_t a, size_t b);

uint8_t *read_labels(char const *const label_fname, size_t a, size_t b);

#endif