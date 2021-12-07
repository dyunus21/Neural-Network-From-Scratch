#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include "optimizer.hpp"

/*
 * Stochastic Gradient Descent Optimizer utilizing provided learning rate
 */

class SGDOptimizer : public Optimizer {
    public:
        SGDOptimizer();
        SGDOptimizer(float learningRate);
        void optimize(float* weights, float* gradients, int size);

    private:
        float learningRate;
};

#endif