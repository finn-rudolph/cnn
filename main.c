#include <stdio.h>

#include "file_io.h"

int main()
{
    example *e = read_range("train-images.idx3-ubyte", 0, 1);
    for (size_t i = 0; i < 28 * 28; i++)
        printf("%u\n", (unsigned)e[0].z[i]);
}