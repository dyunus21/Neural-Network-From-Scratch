#include "layer.hpp"

#include <cstring>

Layer::Layer(int size) {
  nodes = new Node[size];
  shape.push_back(size);
}

Layer::~Layer() { delete[] nodes; }

void Layer::clear() {
  std::memset(nodes, 0, getTotalSize()*sizeof(Node));
}

void Layer::update(Optimizer& optimizer) {}

Node* Layer::getNodes() { return nodes; }

std::vector<int> Layer::getShape() const { return shape; }

int Layer::getTotalSize() const {
  int tot = 1;
  for (int dim : shape) tot *= dim;
  return tot;
}