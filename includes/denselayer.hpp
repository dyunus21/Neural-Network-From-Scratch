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
  DenseLayer(int size);
  void forward_propagate();
  void backward_propagate();
};

#endif