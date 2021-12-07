#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include "optimizer.hpp"
#include <unordered_map>

class AdamOptimizer : public Optimizer {
    public:
        AdamOptimizer();
        AdamOptimizer(float learningRate);
        AdamOptimizer(float learningRate, float beta1, float beta2);
        ~AdamOptimizer();
        void optimize(float* weights, float* gradients, int size);

    private:
        float learningRate;
        float beta1;
        float beta2;
        int t;
        std::unordered_map<float*, float*> parameterCache;
};

#endif