#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H 1

#include <math.h>

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