#include "layer.hpp"

Layer::~Layer(){
    delete[] nodes;
}

const std::vector<Layer*>& Layer::getDependencies() const {
    return dependencies;
}

std::vector<int> Layer::getShape() const {
    return shape;
}

const Node* Layer::getNodes() const {
    return nodes;
}