#include "util.hpp"

#include <cmath>
#include <stdexcept>
#include <stdlib.h>

void Util::forward_activate(ActivationFunction activationFunction, Node* preActivationNodes, Node* postActivationNodes, std::vector<int>& shape) {
  int numNodes = shape[0];
  for (int i=1; i<shape.size(); i++) numNodes *= shape[i];
  switch (activationFunction) {
  case ActivationFunction::none: {
    for (size_t i = 0; i < numNodes; i++) {
      postActivationNodes[i].value = preActivationNodes[i].value;
    }
    break;
  }
  case ActivationFunction::relu: {
    for (size_t i = 0; i < numNodes; i++) {
      postActivationNodes[i].value = preActivationNodes[i].value > 0 ? preActivationNodes[i].value : 0;
    }
    break;
  }
  case ActivationFunction::softmax: {
    float sumOfExponentials = 0;
    for (size_t i = 0; i < numNodes; i++) {
      postActivationNodes[i].value = exp(preActivationNodes[i].value);
      sumOfExponentials += postActivationNodes[i].value;
    }

    for (size_t i = 0; i < numNodes; i++) {
      postActivationNodes[i].value /= sumOfExponentials;
    }
    break;
  }
  }
}

void Util::backward_activate(ActivationFunction activationFunction, Node* preActivationNodes, Node* postActivationNodes, std::vector<int>& shape) {
  int numNodes = shape[0];
  for (int i=1; i<shape.size(); i++) numNodes *= shape[i];
  switch (activationFunction) {
  case ActivationFunction::none: {
    for (size_t i = 0; i < numNodes; i++) {
      postActivationNodes[i].gradient = postActivationNodes[i].gradient;
    }
    break;
  }
  case ActivationFunction::relu: {
    for (size_t i = 0; i < numNodes; i++) {
      preActivationNodes[i].gradient = preActivationNodes[i].value > 0 ? postActivationNodes[i].gradient : 0;
    }
    break;
  }
  case ActivationFunction::softmax: { // It took me a while to work out the math for this
    for (size_t i = 0; i < numNodes; i++) {
      for (size_t j = 0; j < numNodes; i++) {
        if (i == j) {
          preActivationNodes[i].gradient += postActivationNodes[j].gradient * postActivationNodes[j].value * (1. - postActivationNodes[i].value);
        }
        else {
          preActivationNodes[i].gradient -= postActivationNodes[j].gradient * postActivationNodes[i].value * postActivationNodes[j].value;
        }
      }
    }

    break;
  }
  }
}

// utility function that calculates loss using least squared
float Util::loss(float* expected, Node* output, int numNodes) {
  float loss = 0;
  for (size_t i = 0; i < numNodes; i++) {
    float d = output[i].value - expected[i];
    output[i].gradient = -2.*d;
    loss += d*d;
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