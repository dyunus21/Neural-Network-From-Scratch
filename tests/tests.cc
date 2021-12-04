#ifndef CATCH_CONFIG_MAIN
#  define CATCH_CONFIG_MAIN
#endif
#include "catch.hpp"
#include "denselayer.hpp"
#include "neuralnet.hpp"
#include "reader.hpp"
#include "util.hpp"

// equals operator for node, allows some level of imprecision to account for
// floating point approximation
bool operator==(Node a, Node b) {
  return (std::fabs(a.value - b.value) <=
              std::numeric_limits<float>::epsilon() *
                  std::fabs(a.value + b.value) * 2 ||
          std::fabs(a.value - b.value) < std::numeric_limits<float>::min()) &&
         (std::fabs(a.gradient - b.gradient) <=
              std::numeric_limits<float>::epsilon() *
                  std::fabs(a.gradient + b.gradient) * 2 ||
          std::fabs(a.gradient - b.gradient) <
              std::numeric_limits<float>::min());
}

TEST_CASE("forward Activate relu") {
  Node* input = new Node[3];
  Node* expectedPost = new Node[3];
  Node* actualPost = new Node[3];

  std::vector<int> shape;
  shape.push_back(3);

  input[0] = {-5.4, 0};
  input[1] = {0, 0};
  input[2] = {3.14, 0};

  expectedPost[0] = {0, 0};
  expectedPost[1] = {0, 0};
  expectedPost[2] = {3.14, 0};

  Util::forward_activate(
      Util::ActivationFunction::relu, input, actualPost, shape);

  REQUIRE(expectedPost[0] == actualPost[0]);
  REQUIRE(expectedPost[1] == actualPost[1]);
  REQUIRE(expectedPost[2] == actualPost[2]);
}
TEST_CASE("forward activate softmax") {
  Node* input = new Node[3];
  Node* expectedPost = new Node[3];
  Node* actualPost = new Node[3];

  std::vector<int> shape;
  shape.push_back(3);

  input[0] = {-5.4, 0};
  input[1] = {0, 0};
  input[2] = {3.14, 0};

  expectedPost[0] = {0.00018734482774078, 0};
  expectedPost[1] = {0.041479346904477, 0};
  expectedPost[2] = {0.95833330826778, 0};

  Util::forward_activate(
      Util::ActivationFunction::softmax, input, actualPost, shape);

  REQUIRE(expectedPost[0] == actualPost[0]);
  REQUIRE(expectedPost[1] == actualPost[1]);
  REQUIRE(expectedPost[2] == actualPost[2]);
}
