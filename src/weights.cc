#include "weights.hpp"

#include <cstdlib>
#include <cstring>


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
void Weights::forward_apply(Node* n1, Node* n2, int idx) {
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

void Weights::initialize(Util::Initializer initializer,
                         int fan_in,
                         int fan_out) {
  for (int i = 0; i < size; i++) {
    weights[i] = Util::initialize(initializer, fan_in, fan_out);
  }
}

void Weights::clearGradients() { std::memset(gradients, 0, size); }

void Weights::applyGradients() {
  for (int i = 0; i < size; i++) {
    weights[i] += gradients[i];
  }
}