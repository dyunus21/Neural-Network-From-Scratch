#include <iostream>
#include <string>
#include<fstream>
#include <stdexcept>
#include "neuralnet.hpp"
#include "reader.hpp"

/*  Test Dataset: bin/exec tests/datasets/t10k-labels-idx1-ubyte tests/datasets/t10k-images-idx3-ubyte 10000
*   Training Dataset: bin/exec tests/datasets/train-labels-idx1-ubyte tests/datasets/train-images-idx3-ubyte 60000
*/

int main(int argc, char* argv[]) {
    if(argc!=4) {
        std::cout<<"Usage: "<<argv[0]<< " label_path image_path number_of_images"<<std::endl;
        return 1;
    }
    std::string label_path = argv[1];
    std::string image_path = argv[2];
    int num = *argv[3]; //number of labels/images
    
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
}

