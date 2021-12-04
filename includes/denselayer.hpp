#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include <vector>

#include "layer.hpp"
#include "optimizer.hpp"
#include "util.hpp"

/**
 * Represents a dense (hidden) layer of a neural network
 */
class DenseLayer : public Layer {
public:
  DenseLayer() = delete;
  DenseLayer(int size, Layer* previous);
  DenseLayer(int size, Layer* previous, Util::ActivationFunction act);
  DenseLayer(int size, Layer* previous, Util::ActivationFunction act, Util::Initializer init);

  void initialize();
  void forward_propagate();
  void backward_propagate();
  void deep_clear();
  void update(Optimizer& optimizer);

  std::vector<Layer*> getDependencies();

private:
  Util::ActivationFunction activation = Util::ActivationFunction::none;
  Util::Initializer initializer = Util::Initializer::xavier;
  std::vector<Layer*> dependencies;
  Weights weights;
  Biases biases;
};

#endif