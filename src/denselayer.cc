#include "denselayer.hpp"

// TODO
DenseLayer::DenseLayer(int size, Layer* previous):
    Layer(size),
    weights{Weights(previous->getShape()[0] * size)},
    biases{Biases(size)} {
  dependencies.push_back(previous);
}

const std::vector<Layer*>& DenseLayer::getDependencies() const {
  return dependencies;
}