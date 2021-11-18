#include "sgdoptimizer.hpp"

SGDOptimizer::SGDOptimizer(float learning_rate): learning_rate(learning_rate) {}

void SGDOptimizer::optimize(float* weights, float* gradients, int size) {
    for (int i=0; i<size; i++) {
        weights[i] += learning_rate * gradients[i];
    }
}