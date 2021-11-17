#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include <vector>

#include "layer.hpp"

/**
 * Represents a dense (hidden) layer of a neural network
 */
class DenseLayer : public Layer {
public:
  DenseLayer() = delete;
  DenseLayer(int size, Layer* previous);
  void forward_propagate();
  void backward_propagate();
  std::vector<Layer*>& getDependencies();

private:
  std::vector<Layer*> dependencies;
  Weights weights;
  Biases biases;
};

#endif