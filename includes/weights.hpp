#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include "node.hpp"

/**
 * Handles weights and gradients
 * Weights and gradients are stored as one-dimensional arrays
 */
class Weights {
public:
  Weights() = delete;
  Weights(int size);
  ~Weights();
  int getSize();

  void forward_apply(Node* n1, Node* n2, int idx);
  void reverse_apply(Node* n1, Node* n2, int idx);

  void randomize();
  void applyGradients();
  void clearGradients();

private:
  int size;
  float* weights;
  float* gradients;
};

#endif