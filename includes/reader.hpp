#ifndef READER_HPP
#define READER_HPP

#include <string>
#include <fstream>
#include <stdexcept>

class Reader {
public:
    Reader(int numlabels, int numImages,int imageSize);
    int* read_mnist_labels(std::string full_path);
    int** read_mnist_images(std::string full_path);
private:
    int number_of_labels;
    int number_of_images;
    int image_size;
};

#endif