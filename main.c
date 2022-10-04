#include <stdio.h>
#include <string.h>

#include "data_io.h"
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
    enum net_command command;
    if (!strcmp(argv[1], "init"))
        command = NET_INIT;
    else if (!strcmp(argv[1], "evaluate") || !strcmp(argv[1], "eval"))
        command = NET_EVALUATE;
    else if (!strcmp(argv[1], "test"))
        command = NET_TEST;
    else
        print_help();

    return command;
}

static inline network request_network()
{
    char net_fname[100];
    printf("Network file name: ");
    scanf("%s", net_fname);

    network net = network_read(net_fname);
    if (!net.layers)
        exit(EXIT_FAILURE);

    return net;
}

static inline uint8_t **request_images(size_t *a, size_t *b)
{
    printf("Interval [a b) of images to read: ");
    scanf("%zu %zu", a, b);

    char image_fname[100];
    printf("Images file name: ");
    scanf("%s", image_fname);

    uint8_t **const images = read_images(image_fname, *a, *b);
    if (!images)
        exit(EXIT_FAILURE);

    return images;
}

static inline uint8_t *request_labels(size_t a, size_t b)
{
    char label_fname[100];
    printf("Labels file name: ");
    scanf("%s", label_fname);

    uint8_t *const labels = read_labels(label_fname, a, b);
    if (!labels)
        exit(EXIT_FAILURE);

    return labels;
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
        char net_fname[100];
        size_t num_layers, kernel_size;

        printf("Network name: ");
        scanf("%s", net_fname);
        printf("Number of layers: ");
        scanf("%zu", &num_layers);
        printf("Kernel size: ");
        scanf("%zu", &kernel_size);

        while (!(kernel_size & 1))
        {
            printf("Kernel size must be odd. Enter again: ");
            scanf("%zu", &kernel_size);
        }

        network net = network_init(num_layers, kernel_size);
        network_save(&net, net_fname);
        network_destroy(&net);

        break;
    }
    case NET_EVALUATE:
    {
        network net = request_network();
        size_t a, b;
        uint8_t **images = request_images(&a, &b);

        char result_fname[100];
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
        network net = request_network();
        size_t a, b;
        uint8_t **images = request_images(&a, &b);
        uint8_t *labels = request_labels(a, b);

        double **const results = network_pass_forward(&net, b - a, images);
        network_print_accuracy(b - a, results, labels);

        destroy_matrix(b - a, images);
        free(labels);
        destroy_matrix(b - a, results);
        network_destroy(&net);

        break;
    }
    case NET_TRAIN:
    {
        break;
    }
    }
}