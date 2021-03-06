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
  std::memset(weights, 0, size * sizeof(float));
  std::memset(gradients, 0, size * sizeof(float));
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
 * Returns the array storing the weights
 * 
 * @return pointer to the weight array 
 */
float* Weights::getWeights() {
  return weights;
}

/**
 * Returns the array storing the gradients
 * 
 * @return pointer to the gradient array 
 */
float* Weights::getGradients() {
  return gradients;
}

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

void Weights::initialize(Util::Initializer initializer,
                         int fan_in,
                         int fan_out) {
  for (int i = 0; i < size; i++) {
    weights[i] = Util::initialize(initializer, fan_in, fan_out);
  }
}

void Weights::update(Optimizer* optimizer) {
  optimizer->optimize(weights, gradients, size);
}

void Weights::clearGradients() {
  std::memset(gradients, 0, size * sizeof(float));
}