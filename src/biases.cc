#include "biases.hpp"

#include <cstdlib>
#include <cstring>

/**
 * Initializes size, biases, and gradients
 *
 * @param size the number of biases
 */
Biases::Biases(int size): size(size) {
  biases = new float[size];
  gradients = new float[size];
  std::memset(biases, 0, size * sizeof(float));
  std::memset(gradients, 0, size * sizeof(float));
}

Biases::~Biases() {
  delete[] biases;
  delete[] gradients;
}

/**
 * Returns the number of biases
 *
 * @return the number of biases
 */
int Biases::getSize() { return size; }

/**
 * Applies a bias in the forward direction
 *
 * @param n1
 * @param n2
 * @param idx
 */
void Biases::forward_apply(Node* n, int idx) { n->value += biases[idx]; }

/**
 * Propagates gradient backward
 *
 * @param n1
 * @param n2
 * @param idx
 */
void Biases::reverse_apply(Node* n, int idx) { gradients[idx] += n->gradient; }

void Biases::update(Optimizer* optimizer) {
  optimizer->optimize(biases, gradients, size);
}

void Biases::clearGradients() {
  std::memset(gradients, 0, size * sizeof(float));
}