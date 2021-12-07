#include "sgdoptimizer.hpp"

SGDOptimizer::SGDOptimizer(): learningRate(0.01) {}
SGDOptimizer::SGDOptimizer(float learningRate): learningRate(learningRate) {}

void SGDOptimizer::optimize(float* weights, float* gradients, int size) {
    for (int i=0; i<size; i++) {
        weights[i] -= learningRate * (gradients[i]/batch_size);
    }
}