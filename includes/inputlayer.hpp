#ifndef INPUTLAYER_HPP
#define INPUTLAYER_HPP

#include <vector>

#include "layer.hpp"
#include <vector>

/**
 * Represents a 1D input layer of a neural network
 */
class InputLayer : public Layer {
public:
  InputLayer() = delete;
  InputLayer(int size);
  
  void initialize();
  void forward_propagate();
  void backward_propagate();
  std::vector<Layer*> getDependencies();

  void setValues(float* values);
};

#endif