#ifndef INPUTLAYER_HPP
#define INPUTLAYER_HPP

#include <vector>

#include "layer.hpp"

/**
 * Represents a 1D input layer of a neural network
 */
class InputLayer : public Layer {
public:
  InputLayer() = delete;
  InputLayer(int size);
  
  void forward_propagate();
  void backward_propagate();
};

#endif