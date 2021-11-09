class Weights {
    public:
        Weights() = delete;
        Weights(int size);
        int getSize();
    
    private:
        int size;
        float* weights;
        float* gradients;
};