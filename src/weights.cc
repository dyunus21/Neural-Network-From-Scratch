#include "weights.hpp"

/**
 * Initializes size, weights, and gradients
 * 
 * @param size the number of weights
 */
Weights::Weights(int size) : size(size) {
    weights = new float[size];
    gradients = new float[size];
}

/**
 * Returns the number of weights
 * 
 * @return the number of weights
 */
int Weights::getSize() {
    return size;
}

// TODO