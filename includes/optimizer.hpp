#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer {
    public:
        virtual void optimize(float* weights, float* gradients, int size) = 0;
        void set_batch_size(int size) { batch_size = size; };
    protected:
        int batch_size = 1;
};

#endif