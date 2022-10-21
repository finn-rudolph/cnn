# A convolutional neural network for classifying hadwritten digits

The network consists of several convolutional layers followed by fully connected layers. A short description of the architecture and important details follows.

## Data representation

An image is represented as an array of size 784. The input layer converts this to a 28 times 28 two-dimensional array. Data always starts at index 0 in both dimensions. As the fully connected layers use one dimension, a flattening layer in between is needed.

## Control flow

The control flow when performing a usual operation on the network such as training or evaluating images starts in the respective function in `network.c`. Layer-specific actions are performed in `layer.c`. Usually, the function on network level iterates over all layers and invokes layer-specific functions by layer type.