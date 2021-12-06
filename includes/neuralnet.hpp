#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include "inputlayer.hpp"

/**
 * Trains and optimizes neural network using input and output layers
 *
 */
class NeuralNet {
public:
  NeuralNet() = delete;
  NeuralNet(InputLayer* input, Layer* output, Optimizer* optimizer);
  void initialize();
  std::vector<float> predict(float* inputs);
  void propagate(float* input, float* expected);
  void update();
  void deepClear();
  float getLoss();

private:
  InputLayer* input;
  Layer* output;
  float loss;
  Optimizer* optimizer;
  std::vector<Layer*> layerOrder;
  std::vector<Layer*> gatherLayers(Layer* output_layer);
};

#endif