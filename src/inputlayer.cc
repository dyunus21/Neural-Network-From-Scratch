#include "inputlayer.hpp"

InputLayer::InputLayer(int size): Layer(size) {}

void InputLayer::initialize() {};

void InputLayer::forward_propagate() {}  // THESE ARE SUPPOSED TO DO NOTHING

void InputLayer::backward_propagate() {}

void InputLayer::setValues(float* values) {
  for (int i=0; i<getTotalSize(); i++) {
    postActivationNodes[i].value = values[i];
  }
}