#include "file_io.h"

// Reads all examples in [i, j)
example *read_range(char const *const image_fname, char const *const label_fname,
                    size_t i, size_t j)
{
    assert(i <= j);
    FILE *image_f = fopen(image_fname, "rb");
    if (!image_f)
    {
        perror("Error on opening image file");
        return 0;
    }
    FILE *label_f = fopen(label_fname, "rb");
    if (!label_f)
    {
        perror("Error on opening label file");
        return 0;
    }

    unsigned magic_num1, t1, n, m, magic_num2, t2;
    fread(&magic_num1, 4, 1, image_f);
    fread(&t1, 4, 1, image_f);
    fread(&n, 4, 1, image_f);
    fread(&m, 4, 1, image_f);
    fread(&magic_num2, 4, 1, label_f);
    fread(&t2, 4, 1, label_f);

    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    {
        reverseb(&magic_num1);
        reverseb(&t1);
        reverseb(&n);
        reverseb(&m);
        reverseb(&magic_num2);
        reverseb(&t2);
    }

    assert(magic_num1 == 2051);
    assert(magic_num2 == 2049);
    assert(t1 == t2);
    assert(j <= t1);

    example *e = malloc((j - i) * sizeof(example));
    fseek(image_f, i * n * m, SEEK_CUR);
    fseek(label_f, i, SEEK_CUR);

    for (size_t k = 0; k < (j - i); k++)
    {
        e[k].z = malloc(n * m * sizeof(uint8_t));
        fread(e[k].z, 1, n * m, image_f);
        fread(&e[k].d, 1, 1, label_f);
    }

    fclose(image_f);
    fclose(label_f);
    return e;
}