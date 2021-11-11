#ifndef LAYER_HPP
#define LAYER_HPP

#include "node.hpp"
#include "weights.hpp"
#include <vector>

/**
 * Represents a layer of a neural network
 */
class Layer {
    public:
        Layer() = delete;
        virtual void forward_propagate() = 0;
        virtual void backward_propagate() = 0;
        
        const std::vector<Layer*>& getDependencies() const;
        std::vector<int> getShape() const;
        const Node* getNodes() const;

    private:
        std::vector<Layer*> dependencies;
        std::vector<int> shape;
        Weights weights;
        Node* nodes;
};

#endif