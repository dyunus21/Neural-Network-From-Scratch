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
  std::vector<Layer*> layers;
};

#endif