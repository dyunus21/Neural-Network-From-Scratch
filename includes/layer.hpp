#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "biases.hpp"
#include "node.hpp"
#include "weights.hpp"

/**
 * Represents a layer of a neural network
 */
class Layer {
public:
  virtual ~Layer();
  virtual void forward_propagate() = 0;
  virtual void backward_propagate() = 0;

  const std::vector<Layer*>& getDependencies() const;
  std::vector<int> getShape() const;
  const Node* getNodes() const;

protected:
  std::vector<Layer*> dependencies;
  std::vector<int> shape;
  Weights weights;
  Biases biases;
  Node* nodes;
};

#endif