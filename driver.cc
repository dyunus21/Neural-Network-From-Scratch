#include <iostream>
#include <string>
#include<fstream>
#include <stdexcept>
#include "neuralnet.hpp"
#include "reader.hpp"


//setup driver file to have arguments
int main() {

    //sample for test dataset, later can modify to get data from command line
    std::string label_path = "tests/datasets/t10k-labels-idx1-ubyte";
    std::string image_path = "tests/datasets/t10k-images-idx3-ubyte";
    Reader r(10000,10000,784);
    int* labels_dataset = r.read_mnist_labels(label_path);
    int** image_dataset = r.read_mnist_images(image_path);
    std::cout<<labels_dataset[1];
    
}

