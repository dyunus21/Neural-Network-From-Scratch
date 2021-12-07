#include "adamoptimizer.hpp"
#include <cmath>

AdamOptimizer::AdamOptimizer(): learningRate(0.01), beta1(0.9), beta2(0.999), t(0) {}
AdamOptimizer::AdamOptimizer(float learningRate): learningRate(learningRate), beta1(0.9), beta2(0.999), t(0){}
AdamOptimizer::AdamOptimizer(float learningRate, float beta1, float beta2): learningRate(learningRate), beta1(beta1), beta2(beta2), t(0) {}
AdamOptimizer::~AdamOptimizer() {
    for (auto entry : parameterCache) {
        delete[] entry.second;
    }
}

void AdamOptimizer::optimize(float* weights, float* gradients, int size) {
    if (!parameterCache.count(gradients)) {
        parameterCache[gradients] = new float[size*2]();
    }

    float* parameters = parameterCache[gradients];

    for (int i=0; i<size; i++) {
        t++;
        float g = gradients[i] / batch_size;
        float& m = parameters[i];
        float& v = parameters[size+i];
        m = beta1*m + (1.0 - beta1)*g;
        v = beta2*v + (1.0 - beta2)*g*g;
        float mHat = m / (1. - std::pow(beta1, t));
        float vHat = v / (1. - std::pow(beta2, t));
        float& w = weights[i];
        w -= learningRate * mHat / (std::sqrt(vHat) + 1e-8);
    }
}