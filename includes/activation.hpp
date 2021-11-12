#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "node.hpp"
#include "weights.hpp"
#include "biases.hpp"
#include <vector>
enum class activationFunction{Relu};


//other functions

void activate(enum activationFunction, std::vector<Node*> nodes, int length);

#endif