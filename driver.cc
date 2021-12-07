#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>

#include "denselayer.hpp"
#include "inputlayer.hpp"
#include "neuralnet.hpp"
#include "reader.hpp"
#include "inputlayer.hpp"
#include "denselayer.hpp"
#include "optimizer.hpp"
#include "sgdoptimizer.hpp"
#include "adamoptimizer.hpp"
#include "util.hpp"

typedef unsigned char uchar;

void print_image(uchar* image) {
    for (int i = 0; i < 784; i++) {
        if (i % 28 == 0 && i > 0) std::cout << std::endl;
        std::cout << (image[i] > 100 ? "#" : ".");
    }
    std::cout << std::endl;
}

int argmax(std::vector<float> output) {
    int argmax = 0;
    float max = 0.0f;
    for (int i=0; i<output.size(); i++) {
        if (output[i] > max) {
            max = output[i];
            argmax = i;
        }
    }
    return argmax;
}

/*  Test Dataset: bin/exec tests/datasets/t10k-labels-idx1-ubyte
 * tests/datasets/t10k-images-idx3-ubyte 10000 Training Dataset: bin/exec
 * tests/datasets/train-labels-idx1-ubyte tests/datasets/train-images-idx3-ubyte
 * 60000
 */

void test_dataset(std::string label_path,
                  std::string image_path,
                  int number_of_images) {
  int num = number_of_images;  // number of labels/images

  Reader r(num, num, 784);  // 784 = 28*28 (size of an image)
  int* labels_dataset = r.read_mnist_labels(label_path);
  uchar** image_dataset = r.read_mnist_images(image_path);
  // std::cout<<labels_dataset[19];

  // for visualization of number
  print_image(image_dataset[45]);
}

void test_neuralnet() {
    srand(time(NULL));

    InputLayer inputLayer(2);
    DenseLayer midLayer1(20, &inputLayer, Util::ActivationFunction::relu);
    DenseLayer midLayer2(20, &midLayer1, Util::ActivationFunction::relu);
    DenseLayer outputLayer(4, &midLayer2, Util::ActivationFunction::softmax);

    Optimizer* optimizer = new AdamOptimizer(0.01);
    NeuralNet neuralNet(&inputLayer, &outputLayer, optimizer);
    neuralNet.initialize();
    std::cout << "Neural network initialized" << std::endl;

    float input1[] = {0, 0};
    float input2[] = {1, 0};
    float input3[] = {0, 1};
    float input4[] = {1, 1};
    float target1[] = {1, 0, 0, 0};
    float target2[] = {0, 1, 0, 0};
    float target3[] = {0, 0, 1, 0};
    float target4[] = {0, 0, 0, 1};

    std::cout << "output for [ 0, 0 ]: " << neuralNet.predict(input1) << std::endl;
    std::cout << "output for [ 1, 0 ]: " << neuralNet.predict(input2) << std::endl;
    std::cout << "output for [ 0, 1 ]: " << neuralNet.predict(input3) << std::endl;
    std::cout << "output for [ 1, 1 ]: " << neuralNet.predict(input4) << std::endl;

    optimizer->set_batch_size(4);
    for (int i=0; i<100; i++) {
        neuralNet.propagate(input1, target1);
        neuralNet.propagate(input2, target2);
        neuralNet.propagate(input3, target3);
        neuralNet.propagate(input4, target4);
        if ((i+1) % 5 == 0) std::cout << "Loss after batch " << i+1 << ": " << neuralNet.getLoss() << std::endl;
        neuralNet.update();
        neuralNet.deepClear();
    }

    std::cout << "output for [ 0, 0 ]: " << neuralNet.predict(input1) << std::endl;
    std::cout << "output for [ 1, 0 ]: " << neuralNet.predict(input2) << std::endl;
    std::cout << "output for [ 0, 1 ]: " << neuralNet.predict(input3) << std::endl;
    std::cout << "output for [ 1, 1 ]: " << neuralNet.predict(input4) << std::endl;
}

void test_image_recognition() {
    srand(time(NULL));

    int num_train_images = 60000;
    Reader r(num_train_images, num_train_images, 784);  // 784 = 28*28 (size of an image)
    int* labels_dataset = r.read_mnist_labels("tests/datasets/train-labels-idx1-ubyte");
    typedef unsigned char uchar;
    uchar** image_dataset = r.read_mnist_images("tests/datasets/train-images-idx3-ubyte");

    InputLayer inputLayer(784);
    DenseLayer midLayer1(500, &inputLayer, Util::ActivationFunction::relu, Util::Initializer::he);
    DenseLayer midLayer2(200, &midLayer1, Util::ActivationFunction::none, Util::Initializer::he);
    DenseLayer outputLayer(10, &midLayer2, Util::ActivationFunction::softmax, Util::Initializer::he);

    Optimizer* optimizer = new AdamOptimizer(0.01);
    NeuralNet neuralNet(&inputLayer, &outputLayer, optimizer);
    neuralNet.initialize();
    std::cout << "Neural network initialized" << std::endl;

    float input[784];
    float target[10];

    optimizer->set_batch_size(64);
    for (int i=1; i<=20*64; i++) {
        for (int j=0; j<784; j++) {
            input[j] = image_dataset[i-1][j] / 255.;
        }
        std::memset(target, 0, sizeof(float)*10);
        target[labels_dataset[i-1]] = 1;

        //std::cout << neuralNet.predict(input) << std::endl;;
        neuralNet.propagate(input, target);

        if (i % 64 == 0) {
            std::cout << "Loss after batch " << i/64 << ": " << neuralNet.getLoss() / 64. << std::endl;
            neuralNet.update();
            neuralNet.deepClear();
            //if (i == 8) return;
        } 
    }

    for (int i=0; i<10; i++) {
        print_image(image_dataset[10000 + i]);
        for (int j=0; j<784; j++) {
            input[j] = image_dataset[10000 + i][j] / 255.;
        }
        auto res = neuralNet.predict(input);
        std::cout << "prediction: " << argmax(res) << "\n" << res << std::endl;
    }
}

int main(int argc, char* argv[]) {
    //test_dataset("tests/datasets/t10k-labels-idx1-ubyte", "tests/datasets/t10k-images-idx3-ubyte", 10000);
    test_image_recognition();
    //test_neuralnet();

    return 0;
}