#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <math.h>
#include <stdlib.h>

#include "node.hpp"

namespace Util {
  enum class ActivationFunction { none, relu };
  void activate(ActivationFunction activationFunction,
                std::vector<Node*> nodes,
                int length);
  float loss(std::vector<Node*> expected, std::vector<Node*> actual);

  enum class Initializer { xavier, he };
  float randomFloat();
  float initialize(Initializer initializer, int fan_in, int fan_out);
}  // namespace Util

#endif