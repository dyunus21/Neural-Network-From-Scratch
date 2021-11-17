#include <iostream>
#include <string>
#include<fstream>
#include<sstream>
#include <stdexcept>
#include "neuralnet.hpp"

typedef unsigned char uchar;
uchar* read_mnist_labels(std::string full_path, int& number_of_labels);
int main() {
    // std::cout << "Test" << std::endl;
    typedef unsigned char uchar;
    int numLabels = 10000;
    uchar* dataset = read_mnist_labels("/home/vagrant/src/final-project-labgroup41/tests/datasets/t10k-labels-idx1-ubyte", numLabels);
    std::cout<<dataset[0];
}

typedef unsigned char uchar;

uchar* read_mnist_labels(std::string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
            // std::cout<<_dataset[i];
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}