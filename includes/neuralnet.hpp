#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <vector>

#include "layer.hpp"
#include "optimizer.hpp"

/**
 *
 *
 */
class NeuralNet {
public:
  NeuralNet() = delete;
  NeuralNet(Layer* input, Layer* output, Optimizer* optimizer);
  void propagate(float* input, float* output);
  void update();
  void deep_clear();

private:
  Layer* input;
  Layer* output;
  float loss;
  Optimizer* optimizer;
  std::vector<Layer*> layerOrder;

  std::vector<Layer*> gather_layers(Layer* output_layer);
};

#endif