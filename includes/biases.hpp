#ifndef BIASES_HPP
#define BIASES_HPP

#include "node.hpp"
#include "optimizer.hpp"

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
  float* getBiases();
  float* getGradients();

  void forward_apply(Node* n, int idx);
  void reverse_apply(Node* n, int idx);

  void update(Optimizer* optimizer);
  void clearGradients();

private:
  int size;
  float* biases;
  float* gradients;
};

#endif