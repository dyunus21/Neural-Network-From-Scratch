#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>

#include "node.hpp"

namespace Util {
  enum class ActivationFunction { none, relu };
  void activate(ActivationFunction activationFunction,
                std::vector<Node*> nodes,
                int length);
  float loss(std::vector<Node*> expected, std::vector<Node*> actual);
  float relu(float value);

  enum class Initializer { xavier, he };
}  // namespace Util

#endif