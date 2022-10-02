#include <stdio.h>
#include <string.h>

#include "file_io.h"
#include "network.h"

enum action
{
    INIT_NET
};

_Noreturn void print_help()
{
    printf("Please specify an action.\n");
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
    if (argc == 1)
        print_help();

    enum action ac;
    if (!strcmp(argv[1], "init"))
        ac = INIT_NET;
    else
        print_help();

    switch (ac)
    {
    case INIT_NET:
    {
        size_t l, m;
        double a, b;
        printf("Number of layers: ");
        scanf("%zu", &l);
        printf("Kernel size: ");
        scanf("%zu", &m);
        printf("Parameter initialization bounds: ");
        scanf("%lg %lg", &a, &b);

        network z = network_init(l, m, a, b);
        network_save(&z, "net.txt");
        network_destroy(&z);
    }
    }
}