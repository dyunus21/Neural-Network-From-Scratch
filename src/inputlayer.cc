#include "inputlayer.hpp"

InputLayer::InputLayer(int size): Layer(size) {}

void InputLayer::initialize(){};

void InputLayer::forward_propagate() {}  // THESE ARE SUPPOSED TO DO NOTHING

void InputLayer::backward_propagate() {}

std::vector<Layer*> InputLayer::getDependencies() { return std::vector<Layer*>{}; };

void InputLayer::setValues(float* values) {
  for (int i = 0; i < getTotalSize(); i++) {
    postActivationNodes[i].value = values[i];
  }
}

std::vector<Layer*> InputLayer::getDependencies() {
  return std::vector<Layer*>();
}