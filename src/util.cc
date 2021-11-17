#include "util.hpp"

#include <cmath>
#include <stdexcept>

// utility function that calculates loss using least squared
float Util::loss(std::vector<Node*> expected, std::vector<Node*> actual) {
  if (expected.size() != actual.size()) {
    throw std::runtime_error(
        "Loss function needs identical dimensional layers");
  }
  float loss = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    loss += pow(expected.at(i)->value - actual.at(i)->value, 2);
  }
  return loss;
}
void Util::activate(ActivationFunction activationFunction,
                    std::vector<Node*> nodes,
                    int length) {
  switch (activationFunction) {
  case ActivationFunction::relu: {
    for (size_t i = 0; i < nodes.size(); i++) {
      nodes[i]->value = relu(nodes[i]->value);
    }
    break;
  }
  case ActivationFunction::softmax: {
    float sumOfExponentials = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
      sumOfExponentials += exp(nodes[i]->value);
    }

    for (size_t i = 0; i < nodes.size(); i++) {
      nodes[i]->value = exp(nodes[i]->value) / sumOfExponentials;
    }
    break;
  }
  }
}
float Util::relu(float x) {
  if (x > 0.0) {
    return x;
  }
  return 0.0;
}