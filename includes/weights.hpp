#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include "node.hpp"

/**
 * Stores the weights, biases, and gradients as one-dimensional arrays
 */
class Weights {
    public:
        Weights() = delete;
        Weights(int size);
        int getSize();
        void forward_apply(Node* n1, Node* n2, int idx);
        void reverse_apply(Node* n1, Node* n2, int idx);
    
    private:
        int size;
        float* weights;
        float* gradients;
};

#endif