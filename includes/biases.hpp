#ifndef BIASES_HPP
#define BIASES_HPP

#include "node.hpp"

/**
 * Handles biases and gradients
 * Biases and gradients are stored as one-dimensional arrays
 */
class Biases {
public:
  Biases() = delete;
  Biases(int size);
  ~Biases();
  int getSize();

  void forward_apply(Node* n, int idx);
  void reverse_apply(Node* n, int idx);

  void applyGradients();
  void clearGradients();

private:
  int size;
  float* biases;
  float* gradients;
};

#endif