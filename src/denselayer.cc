#include "denselayer.hpp"

// TODO
DenseLayer::DenseLayer(int size, Layer* previous):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)} {
  dependencies.push_back(previous);
}

std::vector<Layer*>& DenseLayer::getDependencies() { return dependencies; }

void DenseLayer::forward_propagate() {
  // weights
  for (int i = 0; i < shape.at(0); i++) {  // loop thru current layer
    for (int j = 0; j < dependencies.at(0)->getShape().at(0);
         j++) {  // loop thru prev layer
      weights.forward_apply(&(dependencies.at(0)->getPostActivationNodes()[j]),
                            &preActivationNodes[i],
                            i * dependencies.at(0)->getTotalSize() + j);
    }
  }

  // biases
  for (int i = 0; i < shape.at(0); i++) {
    biases.forward_apply(&preActivationNodes[i], i);
  }
}

void DenseLayer::backward_propagate() {
  // weights
  for (int i = 0; i < shape.at(0); i++) {  // loop thru current layer
    for (int j = 0; j < dependencies.at(0)->getShape().at(0);
         j++) {  // loop thru prev layer
      weights.reverse_apply(&(dependencies.at(0)->getPostActivationNodes()[j]),
                            &preActivationNodes[i],
                            i * dependencies.at(0)->getTotalSize() + j);
    }
  }

  // biases
  for (int i = 0; i < shape.at(0); i++) {
    biases.reverse_apply(&preActivationNodes[i], i);
  }
}

void DenseLayer::update(Optimizer& optimizer) {
  weights.update(optimizer);
  biases.update(optimizer);
}