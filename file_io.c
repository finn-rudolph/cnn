#include "file_io.h"

// Reads all examples in [i, j)
example *read_range(char *filename, size_t i, size_t j)
{
    assert(i <= j);
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Error on opening example file. ");
        return 0;
    }

    unsigned magic_num, t, n, m;
    fread(&magic_num, 4, 1, file);
    fread(&t, 4, 1, file);
    fread(&n, 4, 1, file);
    fread(&m, 4, 1, file);

    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    {
        reverseb(&magic_num);
        reverseb(&t);
        reverseb(&n);
        reverseb(&m);
    }

    assert(magic_num == 2051);
    assert(j <= t);

    example *e = malloc((j - i) * sizeof(example));
    fseek(file, i * n * m, SEEK_CUR);

    for (size_t k = 0; k < (j - i); k++)
    {
        e[k].z = malloc(n * m * sizeof(uint8_t));
        fread(e[k].z, 1, n * m, file);
    }

    fclose(file);
    return e;
}