#include "neuralnet.hpp"

#include <set>
#include <unordered_map>
#include <stack>
#include <queue>
#include <stdexcept>
#include <algorithm>

NeuralNet::NeuralNet(Layer* input, Layer* output, Optimizer* optimizer) : input(input), output(output), optimizer(optimizer) {
    // performs a topological sort of the layers

    std::vector<Layer*> layers = gather_layers(output);
    for (Layer* l : layers) {
        if (l->getDependencies().size() == 0 && l != input) {
            throw std::runtime_error{"Dangling layer"};
        }
    }

    std::unordered_map<Layer*,int> degree;
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

std::vector<Layer*> NeuralNet::gather_layers(Layer* output_layer) {
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

void NeuralNet::propagate(float* inputs, float* expected) {
    for (Layer* l : layerOrder) {
        l->clear();
    }

    // feed inputs into input layer
    for (int i=0; i<input->getTotalSize(); i++) {
        input->getPostActivationNodes()[i].value = inputs[i];
    }

    for (auto itr = layerOrder.begin(); itr != layerOrder.end(); itr++) {
        (*itr)->forward_propagate();
    }
    loss = Util::loss(expected, output->getPostActivationNodes(), output->getTotalSize());
    for (auto itr = layerOrder.rbegin(); itr != layerOrder.rend(); itr++) {
        (*itr)->backward_propagate();
    }
}

void NeuralNet::update() {
    for (Layer* l : layerOrder) {
        l->update(optimizer);
    }
}

void NeuralNet::deep_clear() {
    for (Layer* l : layerOrder) {
        l->deep_clear();
    }
}