#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "node.hpp"
#include <vector>

namespace Activation {
    enum class ActivationFunction{relu};
    void activate(ActivationFunction activationFunction, std::vector<Node*> nodes, int length);
}

#endif