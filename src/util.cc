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
  for (int i = 0; i < expected.size(); i++) {
    loss += pow(expected.at(i)->value - actual.at(i)->value, 2);
  }
  return loss;
}

float Util::randomFloat() {
  return float(rand())/float((RAND_MAX)) * 2. - 1.;
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
    return randomFloat() * 2. - 1. * std::sqrt(6. / float(fan_in));
}