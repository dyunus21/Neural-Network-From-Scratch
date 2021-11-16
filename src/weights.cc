#include "weights.hpp"

#include <cstdlib>

/**
 * Initializes size, weights, and gradients
 *
 * @param size the number of weights
 */
Weights::Weights(int size): size(size) {
  weights = new float[size];
  gradients = new float[size];
}

Weights::~Weights() {
  delete[] weights;
  delete[] gradients;
}

/**
 * Returns the number of weights
 *
 * @return the number of weights
 */
int Weights::getSize() { return size; }

/**
 * Applies a weight in the forward direction
 *
 * @param n1
 * @param n2
 * @param idx
 */
void Weights::forward_apply(const Node* n1, Node* n2, int idx) {
  n2->value += n1->value * weights[idx];
}

/**
 * Propagates gradient backward
 *
 * @param n1
 * @param n2
 * @param idx
 */
void Weights::reverse_apply(Node* n1, Node* n2, int idx) {
  gradients[idx] += n2->gradient * n1->value;
  n1->gradient += n2->gradient * weights[idx];
}

// TODO

void Weights::randomize() {
  for (int i = 0; i < size; i++) {
    weights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // TODO decide random range for biases
  }
}

void Weights::clearGradients() {
  for (int i = 0; i < size; i++) {
    gradients[i] = 0;
  }
}