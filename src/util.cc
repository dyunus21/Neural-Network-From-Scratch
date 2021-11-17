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
  for(size_t i = 0;i<expected.size();i++)
  {
    loss+= pow(expected.at(i)->value - actual.at(i)->value, 2);
  }
  return loss;
}