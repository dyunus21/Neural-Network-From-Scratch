#include "neuralnet.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <stack>
#include <stdexcept>
#include <unordered_map>

#include "layer.hpp"


NeuralNet::NeuralNet(InputLayer* input, Layer* output, Optimizer* optimizer):
    input(input), output(output), optimizer(optimizer), loss(0) {
  // performs a topological sort of the layers

  std::vector<Layer*> layers = gatherLayers(output);
  for (Layer* l : layers) {
    if (l->getDependencies().size() == 0 && l != input) {
      throw std::runtime_error{"Dangling layer"};
    }
  }

  std::unordered_map<Layer*, int> degree;
  for (Layer* l : layers) {
    for (Layer* d : l->getDependencies()) {
      degree[d]++;
    }
  }

  std::queue<Layer*> queue;
  for (Layer* l : layers) {
    if (degree[l] == 0) {
      queue.push(l);
    }
  }

  while (!queue.empty()) {
    Layer* l = queue.front();
    queue.pop();
    layerOrder.push_back(l);
    for (Layer* d : l->getDependencies()) {
      degree[d]--;
      if (degree[d] == 0) {
        if (d != input) queue.push(d);
      } else if (degree[d] < 0) {
        throw std::runtime_error{"Neural network has a cycle"};
      }
    }
  }

  std::reverse(layerOrder.begin(), layerOrder.end());
}

void NeuralNet::initialize() {
  for (Layer* l : layerOrder) {
    l->initialize();
  }
}

std::vector<Layer*> NeuralNet::gatherLayers(Layer* output_layer) {
  std::set<Layer*> layer_set;
  std::stack<Layer*> stack;
  stack.push(output_layer);
  while (!stack.empty()) {
    Layer* l = stack.top();
    stack.pop();
    if (layer_set.count(l)) {
      continue;
    }
    layer_set.insert(l);
    for (Layer* d : l->getDependencies()) {
      stack.push(d);
    }
  }

  return std::vector<Layer*>{layer_set.begin(), layer_set.end()};
}

std::vector<float> NeuralNet::predict(float* inputs) {
  // clear layer nodes
  for (Layer* l : layerOrder) {
    l->clear();
  }

  // feed inputs into input layer
  input->setValues(inputs);

  for (auto itr = layerOrder.begin(); itr != layerOrder.end(); itr++) {
    (*itr)->forward_propagate();
  }

  std::vector<float> ret;
  for (int i = 0; i < output->getTotalSize(); i++) {
    ret.push_back(output->getPostActivationNodes()[i].value);
  }
  return ret;
}

void NeuralNet::propagate(float* inputs, float* expected) {
  // clear layer nodes
  for (Layer* l : layerOrder) {
    l->clear();
  }

  // feed inputs into input layer
  input->setValues(inputs);

  for (auto itr = layerOrder.begin(); itr != layerOrder.end(); itr++) {
    (*itr)->forward_propagate();
  }
  loss += Util::loss(
      expected, output->getPostActivationNodes(), output->getTotalSize());
  for (auto itr = layerOrder.rbegin(); itr != layerOrder.rend(); itr++) {
    (*itr)->backward_propagate();
  }
}

void NeuralNet::update() {
  for (Layer* l : layerOrder) {
    l->update(optimizer);
  }
}

void NeuralNet::deepClear() {
  for (Layer* l : layerOrder) {
    loss = 0;
    l->deepClear();
  }
}

float NeuralNet::getLoss() { return loss; }