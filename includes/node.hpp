#ifndef NODE_HPP
#define NODE_HPP

/**
 * Represents a node in a neural network
 */
struct Node {
    /// The activation or value of the node
    float value;

    /// The gradient of the node (used during backpropagation)
    float gradient;
};

#endif