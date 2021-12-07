#ifndef CATCH_CONFIG_MAIN
#define CATCH_CONFIG_MAIN
#endif
#include "catch.hpp"
#include "denselayer.hpp"
#include "iostream"
#include "neuralnet.hpp"
#include "optimizer.hpp"
#include "reader.hpp"
#include "sgdoptimizer.hpp"
#include "util.hpp"

//*********************************Helper
// Functions**********************************************
// equals operator for node, allows some level of imprecision to account for
// floating point approximation
bool fuzzy_equals(float a, float b) { return std::fabs(a - b) < 0.0001; }
bool operator==(Node a, Node b) {
  return fuzzy_equals(a.value, b.value) && fuzzy_equals(a.gradient, b.gradient);
}
bool value_equal_to_float(float a, Node b) { return fuzzy_equals(a, b.value); }

//********************************* Test Cases
//**********************************************
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
TEST_CASE("backwards activate softmax") {
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
  Util::backward_activate(
      Util::ActivationFunction::softmax, actualPost, input, shape);
  for (size_t i = 0; i < 3; i++) {
    REQUIRE(expectedPost[i] == actualPost[i]);
  }
}
TEST_CASE("loss function") {
  Node* input = new Node[3];
  input[0] = {1, 0};
  input[1] = {0, 0};
  input[2] = {3, 0};
  float* expected = new float[3];
  expected[0] = 0.9;
  expected[1] = 0.2;
  expected[2] = 3.1;
  float correct_loss = 0.1 * 0.1 + 0.2 * 0.2 + 0.1 * 0.1;
  float loss = Util::loss(expected, input, 3);
  REQUIRE(fuzzy_equals(correct_loss, loss));
  Node* e = new Node[3];
  e[0] = {1, 2. * 0.1};
  e[1] = {0, 2. * -0.2};
  e[2] = {3, 2. * -0.1};
  for (int i = 0; i < 3; i++) {
    REQUIRE(input[i] == e[i]);
  }
}

TEST_CASE("Forward apply weights") {
  Node* inputs = new Node[2]();
  Node* outputs = new Node[2]();
  inputs[0].value = 1.;
  inputs[1].value = 2.;
  Weights weights(4);
  weights.getWeights()[0] = 1;
  weights.getWeights()[1] = 2;
  weights.getWeights()[2] = 3;
  weights.getWeights()[3] = 4;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      weights.forward_apply(&inputs[j], &outputs[i], i * 2 + j);
    }
  REQUIRE(outputs[0].value == 5.);
  REQUIRE(outputs[1].value == 11.);

  delete[] inputs;
  delete[] outputs;
}

TEST_CASE("Reverse apply weights") {
  Node* inputs = new Node[2]();
  Node* outputs = new Node[2]();
  inputs[0].value = 1.;
  inputs[1].value = 2.;
  outputs[0].gradient = 1.;
  outputs[1].gradient = 2.;
  Weights weights(4);
  weights.getWeights()[0] = 1;
  weights.getWeights()[1] = 2;
  weights.getWeights()[2] = 3;
  weights.getWeights()[3] = 4;
  weights.clearGradients();
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      weights.reverse_apply(&inputs[j], &outputs[i], i * 2 + j);
    }
  REQUIRE(inputs[0].gradient == 7.);
  REQUIRE(inputs[1].gradient == 10.);

  delete[] inputs;
  delete[] outputs;
}

TEST_CASE("Forward apply biases") {
  Node* nodes = new Node[2]();
  nodes[0].value = 1.;
  nodes[1].value = 2.;
  Biases biases(2);
  biases.getBiases()[0] = 2;
  biases.getBiases()[1] = 3;
  for (int i = 0; i < 2; i++) {
    biases.forward_apply(&nodes[i], i);
  }
  REQUIRE(nodes[0].value == 3.);
  REQUIRE(nodes[1].value == 5.);

  delete[] nodes;
}

TEST_CASE("Reverse apply biases") {
  Node* nodes = new Node[2]();
  nodes[0].gradient = 1.;
  nodes[1].gradient = 2.;
  Biases biases(2);
  biases.clearGradients();
  for (int i = 0; i < 2; i++) {
    biases.reverse_apply(&nodes[i], i);
  }
  REQUIRE(biases.getGradients()[0] == 1.);
  REQUIRE(biases.getGradients()[1] == 2.);

  delete[] nodes;
}

TEST_CASE("Random Float test", "[RandomFloat]") {
  for (int i = 0; i < 30; i++) {
    float r = Util::randomFloat();
    REQUIRE(r <= 1.0);
    REQUIRE(r >= -1.0);
  }
}

TEST_CASE("SGDOptimizer") {
  Optimizer* optimizer = new SGDOptimizer(0.1);
  float weights[] = {1.0, 2.0, 3.0};
  float gradients[] = {1.0, 2.0, 3.0};
  optimizer->optimize(weights, gradients, 3);
  REQUIRE(fuzzy_equals(weights[0], 0.9));
  REQUIRE(fuzzy_equals(weights[1], 1.8));
  REQUIRE(fuzzy_equals(weights[2], 2.7));
  delete optimizer;
}