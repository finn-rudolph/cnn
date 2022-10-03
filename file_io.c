#include "file_io.h"
#include "util.h"

// Reads all examples in [i, j)
example *read_range(char const *const image_fname, char const *const label_fname,
                    size_t a, size_t b)
{
    assert(a <= b);
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
        rev_int(&magic_num1);
        rev_int(&t1);
        rev_int(&n);
        rev_int(&m);
        rev_int(&magic_num2);
        rev_int(&t2);
    }

    assert(magic_num1 == 2051);
    assert(magic_num2 == 2049);
    assert(t1 == t2);
    assert(b <= t1);

    example *e = malloc((b - a) * sizeof(example));
    fseek(image_f, a * n * m, SEEK_CUR);
    fseek(label_f, a, SEEK_CUR);

    for (size_t i = 0; i < (b - a); i++)
    {
        e[i].image = malloc(n * m * sizeof(uint8_t));
        fread(e[i].image, 1, n * m, image_f);
        fread(&e[i].solution, 1, 1, label_f);
    }

    fclose(image_f);
    fclose(label_f);
    return e;
}