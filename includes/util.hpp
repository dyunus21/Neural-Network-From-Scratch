#ifndef UTIL_HPP
#define UTIL_HPP

#include <math.h>
#include <stdlib.h>

#include <vector>

#include "node.hpp"


namespace Util {
  enum class ActivationFunction { none, relu, softmax };
  void activate(ActivationFunction activationFunction,
                std::vector<Node*> nodes);
  float loss(std::vector<Node*> expected, std::vector<Node*> actual);
  float relu(float value);

  enum class Initializer { xavier, he };
  float randomFloat();
  float initialize(Initializer initializer, int fan_in, int fan_out);
}  // namespace Util

#endif