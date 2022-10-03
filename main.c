#include <stdio.h>
#include <string.h>

#include "file_io.h"
#include "network.h"

typedef enum command command;
enum command
{
    INIT_NET
};

// TODO: help command, more information
_Noreturn void print_help()
{
    printf("Please specify an action.\n");
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
    if (argc == 1)
        print_help();

    command com;
    if (!strcmp(argv[1], "init"))
        com = INIT_NET;
    else
        print_help();

    switch (com)
    {
    case INIT_NET:
    {
        size_t l, m;
        double a, b;
        char name[100];
        memset(name, 0, 100);

        printf("Number of layers: ");
        scanf("%zu", &l);
        printf("Kernel size: ");
        scanf("%zu", &m);
        printf("Parameter initialization bounds: ");
        scanf("%lg %lg", &a, &b);
        printf("Network name: ");
        scanf("%s", name);

        network z = network_init(l, m, a, b);
        network_save(&z, name);
        network_destroy(&z);
    }
    }
}