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
  Layer() = delete;
  Layer(int size);
  virtual ~Layer();
  virtual void forward_propagate() = 0;
  virtual void backward_propagate() = 0;

  const Node* getNodes() const;
  std::vector<int> getShape() const;

protected:
  Node* nodes;
  std::vector<int> shape;
};

#endif