#include "layer.hpp"

Layer::Layer(int size) {
  nodes = new Node[size];
  shape.push_back(size);
}

Layer::~Layer() { delete[] nodes; }

const Node* Layer::getNodes() const { return nodes; }

std::vector<int> Layer::getShape() const { return shape; }