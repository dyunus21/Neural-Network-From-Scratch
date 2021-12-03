#include "util.hpp"

#include <cmath>
#include <stdexcept>
#include <stdlib.h>

void Util::forward_activate(ActivationFunction activationFunction, Node* preActivationNodes, Node* postActivationNodes, std::vector<int>& shape) {
  int num_nodes = shape[0];
  for (int i=1; i<shape.size(); i++) num_nodes *= shape[i];
  switch (activationFunction) {
  case ActivationFunction::relu: {
    for (size_t i = 0; i < num_nodes; i++) {
      postActivationNodes[i].value = preActivationNodes[i].value > 0 ? preActivationNodes[i].value : 0;
    }
    break;
  }
  case ActivationFunction::softmax: {
    float sumOfExponentials = 0;
    for (size_t i = 0; i < num_nodes; i++) {
      sumOfExponentials += exp(preActivationNodes[i].value);
    }

    for (size_t i = 0; i < num_nodes; i++) {
      postActivationNodes[i].value = exp(preActivationNodes[i].value) / sumOfExponentials;
    }
    break;
  }
  }
}

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

float Util::relu(float x) {
  if (x > 0.0) {
    return x;
  }
  return 0.0;
}

float Util::randomFloat() {
  return float(rand()) / float((RAND_MAX)) * 2. - 1.;
}

/**
 * Creates an random weight value based on a certain initialization scheme
 *
 * @param initializer the type of initialization
 * @param fan_in The number of notes going into an output node
 * @param fan_out The number of output nodes that an input node feeds into
 * @return float
 */
float Util::initialize(Util::Initializer initializer, int fan_in, int fan_out) {
  if (initializer == Util::Initializer::xavier)
    return randomFloat() * std::sqrt(6. / float(fan_in + fan_out));
  else if (initializer == Util::Initializer::he)
    return randomFloat() * std::sqrt(6. / float(fan_in));
}