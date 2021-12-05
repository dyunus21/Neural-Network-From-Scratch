#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "biases.hpp"
#include "node.hpp"
#include "optimizer.hpp"
#include "weights.hpp"

/**
 * Represents a layer of a neural network
 */
class Layer {
public:
  Layer() = delete;
  Layer(int size);
  virtual ~Layer();
  virtual void initialize() = 0;
  virtual void forward_propagate() = 0;
  virtual void backward_propagate() = 0;
  virtual void clear();
  virtual void deepClear();
  virtual void update(Optimizer* optimizer);

  Node* getPreActivationNodes();
  Node* getPostActivationNodes();
  std::vector<int> getShape() const;
  int getTotalSize() const;
  virtual std::vector<Layer*> getDependencies() = 0;

  void printValues() const;

protected:
  Node* preActivationNodes;
  Node* postActivationNodes;
  std::vector<int> shape;
};

#endif