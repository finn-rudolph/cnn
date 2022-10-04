#include "data_io.h"
#include "util.h"

uint8_t **read_images(char const *const image_fname, size_t a, size_t b)
{
    assert(a <= b);
    FILE *image_f = fopen(image_fname, "rb");
    if (!image_f)
    {
        perror("Error on opening image file");
        return 0;
    }

    unsigned magic_num, t, n, m;
    fread(&magic_num, 4, 1, image_f);
    fread(&t, 4, 1, image_f);
    fread(&n, 4, 1, image_f);
    fread(&m, 4, 1, image_f);

    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    {
        rev_int(&magic_num);
        rev_int(&t);
        rev_int(&n);
        rev_int(&m);
    }

    assert(magic_num == 2051);
    assert(b <= t);

    uint8_t **images = malloc((b - a) * sizeof(uint8_t *));

    fseek(image_f, a * n * m, SEEK_CUR);
    for (size_t i = 0; i < (b - a); i++)
    {
        images[i] = malloc(n * m * sizeof(uint8_t));
        fread(images[i], 1, n * m, image_f);
    }

    fclose(image_f);
    return images;
}

uint8_t *read_labels(char const *const label_fname, size_t a, size_t b)
{
    assert(a <= b);

    FILE *label_f = fopen(label_fname, "rb");
    if (!label_f)
    {
        perror("Error on opening label file");
        return 0;
    }

    unsigned magic_num, t;
    fread(&magic_num, 4, 1, label_f);
    fread(&t, 4, 1, label_f);

    if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    {
        rev_int(&magic_num);
        rev_int(&t);
    }

    assert(magic_num == 2049);
    assert(b <= t);

    uint8_t *labels = malloc((b - a) * sizeof(uint8_t));

    fseek(label_f, a, SEEK_CUR);
    for (size_t i = 0; i < (b - a); i++)
    {
        fread(labels + i, 1, 1, label_f);
    }

    fclose(label_f);
    return labels;
}
