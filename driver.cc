#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>
#include "neuralnet.hpp"
#include "reader.hpp"
#include "inputlayer.hpp"
#include "denselayer.hpp"
#include "util.hpp"

/*  Test Dataset: bin/exec tests/datasets/t10k-labels-idx1-ubyte tests/datasets/t10k-images-idx3-ubyte 10000
*   Training Dataset: bin/exec tests/datasets/train-labels-idx1-ubyte tests/datasets/train-images-idx3-ubyte 60000
*/

void test_dataset(std::string label_path, std::string image_path, int number_of_images) {
    int num = number_of_images; //number of labels/images
    
    Reader r(num,num,784);  // 784 = 28*28 (size of an image)
    int* labels_dataset = r.read_mnist_labels(label_path);
    typedef unsigned char uchar;
    uchar** image_dataset = r.read_mnist_images(image_path);
    // std::cout<<labels_dataset[19];

    //for visualization of number
    for(int i = 0; i<784;i++){
        if(i%28==0)
            std::cout<<std::endl;
        std::cout<<int(image_dataset[45][i])<< " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    test_dataset("tests/datasets/t10k-labels-idx1-ubyte", "tests/datasets/t10k-images-idx3-ubyte", 10000);

    srand(time(NULL));
    InputLayer input(2);
    DenseLayer layer(2, &input, Util::ActivationFunction::relu);
    float test_data[2] = {1.0, -2.0};
    layer.initialize();
    input.clear();
    layer.clear();
    input.setValues(test_data);
    layer.forward_propagate();
    
    std::cout << "Input Values:" << std::endl;
    input.printValues();
    std::cout << "Resulting values:" << std::endl;
    layer.printValues();
}

