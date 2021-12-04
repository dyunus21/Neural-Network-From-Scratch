#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include "node.hpp"
#include "util.hpp"
#include "optimizer.hpp"

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

  void initialize(Util::Initializer initializer, int fan_in, int fan_out);
  void update(Optimizer* optimizer);
  void clearGradients();

private:
  int size;
  float* weights;
  float* gradients;
};

#endif