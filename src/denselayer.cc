#include "denselayer.hpp"

// TODO
DenseLayer::DenseLayer(int size): weights{Weights(0)}, biases{Biases(0)} {
  shape.push_back(size);
  nodes = new Node[size];
  for (int i = 0; i < size; i++) {
    nodes[i] = {0, 0};
  }
}