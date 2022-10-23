# A convolutional neural network for classifying hadwritten digits

The network consists of several convolutional layers followed by fully connected layers. A short description of the architecture and important details follows.

## File Structure

`main.c` contains the `main` function and is the starting point for all interaction with the network. In `network.c`, network level functions such as a feed-forward and training procedure are defined. Also reading a network from a file and saving it is handled here. `layer.c` contains layer-specific functions. There are four types of layers:

- Input layer: This layer arranges an input image in a matrix.

- Convolutional layers: Nodes in these layers are organized in a grid. The feed-forward operation is a convolution of the previous layers values with the layer's kernel. Learning modifies the kernel values and the bias.

- Fully connected layers: These layers have a weight matrix and bias vector. Feed forward is a matrix-vector product of the previous layer's value vector with the weight matrix. Then the bias vector is added.

- Flattening layer: Lies in between the convolutional and fully connected layers and flattens the convolutional layer's matrix into a vector.

There is always one input layer, some number of convolutional layers, then one flattening layer and finally some number of fully connected layers. After passing the values forward to the next layer, they are piped through the activiation function, which can be specified by the user in `network_config.h`. Activation functions and their derivatives are defined in `util.h`.

`convolution.c` contains the straight-forward $O(n^2)$ convolution algorithm as well as convolution via the Fast-Fourier-Transform. `image_data.c` reads in images and labels in the [MNIST format](http://yann.lecun.com/exdb/mnist/). In this file, also a function for normalizing the images is contained. Images are normalized in mini-batches, whose size is specified by `NORMALIZATION_BATCH_SIZE` in `network_config.h`.

## Data representation

An image is represented as an array of size 784. The input layer converts this to a 28 times 28 two-dimensional array. Data always starts at index 0 in both dimensions. As the fully connected layers use one dimension, a flattening layer between the convolutional and fully connected layers is needed.

## Formulas for Gradient Descent

The equations used to implement backpropagation and gradient descent are summarized here.

### List of Symbols

$C:$ Overall cost.

$f:$ Activation function.

$a_i^l:$ Input of neuron $i$ in layer $l$.

$z_i^l:$ Output of neuron $i$ in layer $l$. Defined as $f(a_i^l)$.

$\delta_i^l:$ The gradient of the cost function with respect to node $i$ in layer $l$. Defined as $\frac {\partial C} {\partial a_i^l}$. In the context of convolutional layers, nodes are arranged in a matrix, so $\delta_{i, j}^l$ is used to denote the gradient of the node in the $i$th row and $j$th column. The same notation is also used for $a$ and $z$.

$b_i^l:$ The bias of node $i$ in layer $l$. In case of convolutional layers, the subscript is omitted.

$y:$ The correct digit in the current example.

$L:$ The number of layers in the network.

$k:$ The kernel side length (always odd).

$n:$ Side length of the node matrix in context of a convolutional layer, so each convolutional layer has $n^2$ nodes. In context of a fully connected layer, $n$ is it's number of nodes.

$w_{i, j}^l:$ The weight in the $l$th layer's kernel in the context of convolutional layers. For fully connected layers, the weight between the $i$th node of the $l$th layer and the $j$th node of the $(l-1)$-th layer.

$t:$ Mini-batch size.

Zero-indexing is used in all equations. When $\delta, a, z$ etc. is used without subscript, the vector or matrix of all nodes in the respective layer is meant.

### Cost Function

The log-likelihood cost function with a regularization parameter $\lambda$ was used. The last layer is a softmax layer.

$$
C = -\ln z_y^{L-1} + \frac {\lambda} {2t} \sum_{w} w^2
$$

$w$ stands for all weights appearing in the network, in convolutional as well as fully connected layers. The first term is the actual cost, the second for regularization.
### Convolutional Layers

$*$ denotes a convolution.

$$
\delta^l = \delta^{l+1} * \text{flipped} (w^{l+1})
$$

$$
\frac {\partial C} {\partial w^l} = (\delta^{l} * z^{l-1}) + \frac \lambda t w^l
$$

$$
\frac {\partial C} {\partial b^l} = \sum_{i = 0}^{n-1} \sum_{j= 0}^{n-1} \delta_{i, j}^l
$$

In the second equation, $\delta^l$ must be padded with $\lfloor k/2 \rfloor$ zeros evenly, such that the result of the convolution has size $k$.

### Fully Connected Layers

$$
\delta^l = \delta^{l+1} w^{l+1}
$$

$$
\frac {\partial C} {\partial w^l_{i, j}} = z^{l-1}_j \delta^l_i + \frac \lambda t {w^l_{i, j}}
$$

$$
\frac {\partial C} {\partial b^l_i} = \delta^l_i
$$

In the first equation, the $\delta$-vector of the $(l+1)$-th layer is multiplied with this layer's weight matrix on the left.