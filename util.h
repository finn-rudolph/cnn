#ifndef UTIL_H
#define UTIL_H 1

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define SQUARE(x) (x * x)

static inline double rand_double(double a, double b)
{
    return a + (rand() / (RAND_MAX / (b - a)));
}

static inline void rev_uint16(uint16_t *x)
{
    uint16_t y = (((*x & 0x00FF) << 8) |
                  ((*x & 0xFF00) >> 8));
    *x = y;
}

static inline void rev_uint32(uint32_t *x)
{
    uint32_t y = (((*x & 0x000000FF) << 24) |
                  ((*x & 0x0000FF00) << 8) |
                  ((*x & 0x00FF0000) >> 8) |
                  ((*x & 0xFF000000) >> 24));
    *x = y;
}

#define rev_int(x)         \
    _Generic(x,            \
             uint16_t *    \
             : rev_uint16, \
               uint32_t *  \
             : rev_uint32)(x)

// Suffix _d means derivative.

static inline double relu(double x)
{
    return x > 0.0 ? x : 0.0;
}

static inline double relu_d(double x)
{
    return x > 0.0 ? 1 : 0;
}

static inline double relu_smooth(double x)
{
    return log1p(1 + exp(x));
}

static inline double relu_smooth_d(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

#endif