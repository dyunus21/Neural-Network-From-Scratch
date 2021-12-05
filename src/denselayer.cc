#include "denselayer.hpp"

// TODO
DenseLayer::DenseLayer(int size, Layer* previous):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)} {
  dependencies.push_back(previous);
}

DenseLayer::DenseLayer(int size, Layer* previous, Util::ActivationFunction act):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)},
    activation{act} {
  dependencies.push_back(previous);
}

DenseLayer::DenseLayer(int size, Layer* previous, Util::ActivationFunction act, Util::Initializer init):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)},
    activation{act},
    initializer{init} {
  dependencies.push_back(previous);
}

std::vector<Layer*> DenseLayer::getDependencies() { return dependencies; }

void DenseLayer::initialize() {
  weights.initialize(initializer, dependencies[0]->getTotalSize(), getTotalSize());
}

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

  // activations
  Util::forward_activate(activation, preActivationNodes, postActivationNodes, shape);
}

void DenseLayer::backward_propagate() {
  // activations
  Util::backward_activate(activation, preActivationNodes, postActivationNodes, shape);

  // biases
  for (int i = 0; i < getTotalSize(); i++) {
    biases.reverse_apply(&preActivationNodes[i], i);
  }

  // weights
  for (int i = 0; i < getTotalSize(); i++) {  // loop thru current layer
    for (int j = 0; j < dependencies.at(0)->getTotalSize();
         j++) {  // loop thru prev layer
      weights.reverse_apply(&(dependencies.at(0)->getPostActivationNodes()[j]),
                            &preActivationNodes[i],
                            i * dependencies.at(0)->getTotalSize() + j);
    }
  }
}

void DenseLayer::deepClear() {
  clear();
  weights.clearGradients();
  biases.clearGradients();
}

void DenseLayer::update(Optimizer* optimizer) {
  weights.update(optimizer);
  biases.update(optimizer);
}