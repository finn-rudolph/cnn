#ifndef FILE_IO_H
#define FILE_IO_H 1

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

typedef struct example example;
struct example
{
    uint8_t *z;
    uint8_t d;
};

example *read_range(char *image_file, char *label_file, size_t i, size_t j);

static inline void reverse16(uint16_t *x)
{
    uint16_t y = (((*x & 0x00FF) << 8) |
                  ((*x & 0xFF00) >> 8));
    *x = y;
}

static inline void reverse32(uint32_t *x)
{
    uint32_t y = (((*x & 0x000000FF) << 24) |
                  ((*x & 0x0000FF00) << 8) |
                  ((*x & 0x00FF0000) >> 8) |
                  ((*x & 0xFF000000) >> 24));
    *x = y;
}

#define reverseb(x)       \
    _Generic(x,           \
             uint16_t *   \
             : reverse16, \
               uint32_t * \
             : reverse32)(x)

#endif