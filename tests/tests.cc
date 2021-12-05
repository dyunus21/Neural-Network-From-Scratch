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

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(expectedPost[i] == actualPost[i]);
  }
}

TEST_CASE("backward Activate relu") {
  Node* post = new Node[3];
  Node* expectedPre = new Node[3];
  Node* actualPre = new Node[3];

  std::vector<int> shape;
  shape.push_back(3);

  post[0] = {0, 3};
  post[1] = {0, 6};
  post[2] = {3.14, 9};

  expectedPre[0] = {-5.4, 0};
  expectedPre[1] = {0, 0};
  expectedPre[2] = {3.14, 9};

  actualPre[0] = {-5.4, 0};
  actualPre[1] = {0, 0};
  actualPre[2] = {3.14, 0};

  Util::backward_activate(
      Util::ActivationFunction::relu, actualPre, post, shape);

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(expectedPre[i] == actualPre[i]);
  }
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

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(expectedPost[i] == actualPost[i]);
  }
}
TEST_CASE("forward activate none") {
  Node* input = new Node[3];
  Node* expectedPost = new Node[3];
  Node* actualPost = new Node[3];
  std::vector<int> shape;
  shape.push_back(3);
  input[0] = {1, 0};
  input[1] = {0, 0};
  input[2] = {3.14, 0};
  expectedPost[0] = {1, 0};
  expectedPost[1] = {0, 0};
  expectedPost[2] = {3.14, 0};
  Util::forward_activate(
      Util::ActivationFunction::none, input, actualPost, shape);
  for (size_t i = 0; i < 3; i++) {
    REQUIRE(expectedPost[i] == actualPost[i]);
  }
}
TEST_CASE("backwards activate softmax")
{
  Node* input = new Node[3];
  Node* expectedPost = new Node[3];
  Node* actualPost = new Node[3];
  std::vector<int> shape;
  shape.push_back(3);
  input[0] = {0.5, 0.25};
  input[1] = {0.75, 0.67};
  input[2] = {0.8, 1};
  actualPost[0] = {0, 0};
  actualPost[1] = {0, 0};
  actualPost[2] = {0, 0};

  expectedPost[0] = {0, -0.58875};
  expectedPost[1] = {0, -0.568125};
  expectedPost[2] = {0, -0.342};
  Util::backward_activate(Util::ActivationFunction::softmax, actualPost, input, shape);
  for(size_t i = 0;i<3;i++)
  {
    REQUIRE(expectedPost[i] == actualPost[i]);
  }
}

