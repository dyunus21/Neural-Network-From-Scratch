#ifndef UTIL_HPP
#define UTIL_HPP

#include "node.hpp"
#include <vector>

namespace Util {
    enum class ActivationFunction{none, relu};
    void activate(ActivationFunction activationFunction, std::vector<Node*> nodes, int length);

    enum class Initializer{xavier, he};
}

#endif