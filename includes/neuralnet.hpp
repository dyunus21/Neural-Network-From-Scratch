#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <vector>

#include "layer.hpp"

/**
 *
 *
 */
class NeuralNet {
public:
  NeuralNet() = delete;
  NeuralNet(Layer* input, Layer* output);
  void train();

private:
  Layer* input;
  Layer* output;
  std::vector<Layer*> layer_order;

  std::vector<Layer*> gather_layers(Layer* output_layer);
};

#endif