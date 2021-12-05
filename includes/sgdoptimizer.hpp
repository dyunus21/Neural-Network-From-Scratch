#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include "optimizer.hpp"

/*
 * Stochastic Gradient Descent Optimizer using provided learning rate
 */

class SGDOptimizer : public Optimizer {
    public:
        SGDOptimizer(float learning_rate);
        void optimize(float* weights, float* gradients, int size);

    private:
        float learning_rate;
};

#endif