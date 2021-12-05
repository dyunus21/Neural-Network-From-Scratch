#ifndef UTIL_HPP
#define UTIL_HPP

#include <math.h>
#include <stdlib.h>

#include <vector>
#include <iostream>

#include "node.hpp"
/**
  * Handles the enum class for Activation Function and Initializer
  *
  */

namespace Util {
  enum class ActivationFunction { none, relu, softmax };
  void forward_activate(ActivationFunction activationFunction, Node* preActivationNodes, Node* postActivationNodes, std::vector<int>& shape);
  void backward_activate(ActivationFunction activationFunction, Node* preActivationNodes, Node* postActivationNodes, std::vector<int>& shape);
  float loss(float* expected, Node* output, int numNodes);
  float relu(float value);

  enum class Initializer { xavier, he };
  float randomFloat();
  float initialize(Initializer initializer, int fan_in, int fan_out);
}  // namespace Util

std::ostream& operator<<(std::ostream& os, std::vector<float> vec);

#endif