#include "neuralnet.hpp"
#include <set>
#include <unordered_map>
#include <stack>
#include <queue>
#include <stdexcept>

NeuralNet::NeuralNet(Layer* input, Layer* output) : input(input), output(output) {
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
        layer_order.push_back(l);
        for (Layer* d : l->getDependencies()) {
            degree[d]--;
            if (degree[d] == 0) {
                queue.push(d);
            } else if (degree[d] < 0) {
                throw std::runtime_error{"Neural network has a cycle"};
            }
        }
    }
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