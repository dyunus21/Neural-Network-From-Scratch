#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer {
    public:
        virtual void optimize(float* weights, float* gradients, int size) = 0;
};

#endif