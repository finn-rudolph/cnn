#ifndef FILE_IO_H
#define FILE_IO_H 1

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

typedef struct example example;
struct example
{
    uint8_t *image;
    uint8_t solution;
};

example *read_range(char const *const image_fname, char const *const label_fname,
                    size_t a, size_t b);

#endif