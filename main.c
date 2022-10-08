#include <stdio.h>
#include <string.h>

#include "data_io.h"
#include "network.h"
#include "util.h"
#include "def.h"

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
    else if (!strcmp(argv[1], "train"))
        command = NET_TRAIN;
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
        size_t num_conv, num_fc, kernel_size, fc_size;

        printf("Network name: ");
        scanf("%s", net_fname);
        printf("Number of convolutional layers: ");
        scanf("%zu", &num_conv);
        printf("Kernel size: ");
        scanf("%zu", &kernel_size);

        while (!(kernel_size & 1))
        {
            printf("Kernel size must be odd. Enter again: ");
            scanf("%zu", &kernel_size);
        }

        printf("Number of fully connected layers: ");
        scanf("%zu", &num_fc);
        printf("Number of nodes in fully connected layers: ");
        scanf("%zu", &fc_size);

        assert(num_conv && num_fc);
        assert(kernel_size > 2);
        assert(fc_size >= 10);

        network net = network_init(num_conv, num_fc, kernel_size, fc_size);
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

        matrix_free(b - a, images);
        matrix_free(b - a, results);
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

        matrix_free(b - a, images);
        free(labels);
        matrix_free(b - a, results);
        network_destroy(&net);

        break;
    }
    case NET_TRAIN:
    {
        network net = request_network();
        size_t a, b;
        uint8_t **images = request_images(&a, &b);
        uint8_t *labels = request_labels(a, b);

        char new_fname[100];
        printf("Output file of the trained network: ");
        scanf("%s", new_fname);
        size_t epochs;
        printf("Number of training epochs: ");
        scanf("%zu", &epochs);

        network_train(&net, epochs, b - a, images, labels);
        network_save(&net, new_fname);

        matrix_free(b - a, images);
        free(labels);
        network_destroy(&net);
        break;
    }
    }
}