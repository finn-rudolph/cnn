#include <stdlib.h>

#define SQUARE(x) (x * x)

static inline double rand_double(double a, double b)
{
    return a + (rand() / (RAND_MAX / (b - a)));
}