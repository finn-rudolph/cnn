#include <stdio.h>
#include <string.h>

#include "file_io.h"
#include "network.h"
#include "util.h"

enum net_command
{
    NET_INIT,
    NET_EVALUATE,
    NET_TEST,
    NET_TRAIN,
};

// TODO: help command, more information
_Noreturn void print_help()
{
    printf("Please specify an action.\n");
    exit(EXIT_SUCCESS);
}

enum net_command get_command(int argc, char **argv)
{
    enum net_command arg;
    if (!strcmp(argv[1], "init"))
        arg = NET_INIT;
    else if (!strcmp(argv[1], "evaluate"))
        arg = NET_EVALUATE;
    else
        print_help();

    return arg;
}

int main(int argc, char **argv)
{
    if (argc == 1)
        print_help();

    enum net_command command = get_command(argc, argv);

    switch (command)
    {
    case NET_INIT:
    {
        size_t num_layers, kernel_size;
        char net_fname[100];

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
        scanf("%s", net_fname);

        network net = network_init(num_layers, kernel_size);
        network_save(&net, net_fname);
        network_destroy(&net);

        break;
    }
    case NET_EVALUATE:
    {
        char net_fname[100];
        printf("Network file name: ");
        scanf("%s", net_fname);
        network net = network_read(net_fname);
        if (!net.layers)
            exit(EXIT_FAILURE);

        char image_fname[100], result_fname[100];
        size_t a, b;
        printf("Images file name: ");
        scanf("%s", image_fname);
        printf("Interval [a b) of images to read: ");
        scanf("%zu %zu", &a, &b);
        uint8_t **const images = read_images(image_fname, a, b);
        printf("Output file name: ");
        scanf("%s", result_fname);

        double **const results = network_pass_forward(&net, b - a, images);
        network_save_results(result_fname, b - a, results);

        destroy_matrix(b - a, images);
        destroy_matrix(b - a, results);
        network_destroy(&net);

        break;
    }
    case NET_TEST:
    {
        break;
    }
    case NET_TRAIN:
    {
        break;
    }
    }
}