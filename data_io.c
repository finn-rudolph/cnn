#include "data_io.h"
#include "util.h"

double **read_images(char const *const image_fname, size_t a, size_t b)
{
    assert(a <= b);
    FILE *stream = fopen(image_fname, "rb");
    if (!stream)
    {
        perror("Error on opening image file");
        return 0;
    }

    unsigned magic_num, t, n, m;
    fread(&magic_num, 4, 1, stream);
    fread(&t, 4, 1, stream);
    fread(&n, 4, 1, stream);
    fread(&m, 4, 1, stream);

    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    {
        rev_int(&magic_num);
        rev_int(&t);
        rev_int(&n);
        rev_int(&m);
    }

    assert(magic_num == 2051);
    assert(b <= t);

    double **images = double_matrix_alloc(b - a, n * m);
    fseek(stream, a * n * m, SEEK_CUR);

    for (size_t i = 0; i < (b - a); i++)
    {
        for (size_t j = 0; j < n * m; j++)
        {
            uint8_t x;
            fread(&x, 1, 1, stream);
            images[i][j] = x;
        }
    }

    fclose(stream);
    return images;
}

uint8_t *read_labels(char const *const label_fname, size_t a, size_t b)
{
    assert(a <= b);

    FILE *stream = fopen(label_fname, "rb");
    if (!stream)
    {
        perror("Error on opening label file");
        return 0;
    }

    unsigned magic_num, t;
    fread(&magic_num, 4, 1, stream);
    fread(&t, 4, 1, stream);

    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    {
        rev_int(&magic_num);
        rev_int(&t);
    }

    assert(magic_num == 2049);
    assert(b <= t);

    uint8_t *labels = malloc((b - a) * sizeof(uint8_t));

    fseek(stream, a, SEEK_CUR);
    for (size_t i = 0; i < (b - a); i++)
    {
        fread(labels + i, 1, 1, stream);
    }

    fclose(stream);
    return labels;
}
