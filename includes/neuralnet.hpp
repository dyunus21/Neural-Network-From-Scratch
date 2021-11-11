#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <layer.hpp>
#include <vector>

/**
 *
 *
 */
class NeuralNet {
public:
  NeuralNet() = delete;
  void train();

private:
  std::vector<Layer*> layers;
};

#endif