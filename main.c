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
        size_t num_layers, kernel_size;
        char name[100];
        memset(name, 0, 100);

        printf("Number of layers: ");
        scanf("%zu", &num_layers);
        printf("Kernel size: ");
        scanf("%zu", &kernel_size);
        while (!(kernel_size & 1))
        {
            printf("Kernel size must be odd. Enter again: ");
            scanf("%zu", &kernel_size);
        }

        printf("Network name: ");
        scanf("%s", name);

        network net = network_init(num_layers, kernel_size);
        network_save(&net, name);
        network_destroy(&net);
    }
    }
}