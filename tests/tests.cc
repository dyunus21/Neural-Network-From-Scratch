#ifndef CATCH_CONFIG_MAIN
#  define CATCH_CONFIG_MAIN
#endif
#include "catch.hpp"
#include "denselayer.hpp"
#include "neuralnet.hpp"
#include "reader.hpp"
#include "util.hpp"

bool operator==(Node a, Node b) {
  return (a.value == b.value) && (a.gradient == b.gradient);
}

TEST_CASE("Test case goes here") {}
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
