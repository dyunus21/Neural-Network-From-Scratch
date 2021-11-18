#include "denselayer.hpp"

// TODO
DenseLayer::DenseLayer(int size, Layer* previous):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)} {
  dependencies.push_back(previous);
}

DenseLayer::DenseLayer(int size, Layer* previous, Util::ActivationFunction a):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)},
    activation{a} {
  dependencies.push_back(previous);
}

std::vector<Layer*>& DenseLayer::getDependencies() { return dependencies; }

void DenseLayer::forward_propagate() {
  // weights
  for (int i = 0; i < getTotalSize(); i++) {  // loop thru current layer
    for (int j = 0; j < dependencies.at(0)->getTotalSize();
         j++) {  // loop thru prev layer
      weights.forward_apply(&(dependencies.at(0)->getPostActivationNodes()[j]),
                            &preActivationNodes[i],
                            i * dependencies.at(0)->getTotalSize() + j);
    }
  }

  // biases
  for (int i = 0; i < getTotalSize(); i++) {
    biases.forward_apply(&preActivationNodes[i], i);
  }

  // apply activation function
  std::vector<Node*> to_activate;
  for (int i = 0; i < getTotalSize(); i++) {
    postActivationNodes[i] = preActivationNodes[i];
    to_activate.push_back(&(postActivationNodes[i]));
  }
  Util::activate(activation, to_activate);
}

void DenseLayer::backward_propagate() {
  // weights
  for (int i = 0; i < getTotalSize(); i++) {  // loop thru current layer
    for (int j = 0; j < dependencies.at(0)->getTotalSize();
         j++) {  // loop thru prev layer
      weights.reverse_apply(&(dependencies.at(0)->getPostActivationNodes()[j]),
                            &preActivationNodes[i],
                            i * dependencies.at(0)->getTotalSize() + j);
    }
  }

  // biases
  for (int i = 0; i < getTotalSize(); i++) {
    biases.reverse_apply(&preActivationNodes[i], i);
  }
}

void DenseLayer::update(Optimizer& optimizer) {
  weights.update(optimizer);
  biases.update(optimizer);
}