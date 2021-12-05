#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "denselayer.hpp"
#include "inputlayer.hpp"
#include "neuralnet.hpp"
#include "reader.hpp"
#include "inputlayer.hpp"
#include "denselayer.hpp"
#include "optimizer.hpp"
#include "sgdoptimizer.hpp"
#include "util.hpp"


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
  typedef unsigned char uchar;
  uchar** image_dataset = r.read_mnist_images(image_path);
  // std::cout<<labels_dataset[19];

  // for visualization of number
  for (int i = 0; i < 784; i++) {
    if (i % 28 == 0) std::cout << std::endl;
    std::cout << int(image_dataset[45][i]) << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // test_dataset("tests/datasets/t10k-labels-idx1-ubyte", "tests/datasets/t10k-images-idx3-ubyte", 10000);

    srand(time(NULL));

    InputLayer inputLayer(2);
    DenseLayer midLayer1(20, &inputLayer, Util::ActivationFunction::relu);
    DenseLayer midLayer2(20, &midLayer1, Util::ActivationFunction::relu);
    DenseLayer outputLayer(4, &inputLayer, Util::ActivationFunction::softmax);

    Optimizer* optimizer = new SGDOptimizer(0.01);
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
    for (int i=0; i<10000; i++) {
        neuralNet.propagate(input1, target1);
        neuralNet.propagate(input2, target2);
        neuralNet.propagate(input3, target3);
        neuralNet.propagate(input4, target4);
        neuralNet.update();
        neuralNet.deepClear();
        if ((i+1) % 100 == 0) std::cout << "Loss after batch " << i+1 << ": " << neuralNet.getLoss() << std::endl;
    }

    std::cout << "output for [ 0, 0 ]: " << neuralNet.predict(input1) << std::endl;
    std::cout << "output for [ 1, 0 ]: " << neuralNet.predict(input2) << std::endl;
    std::cout << "output for [ 0, 1 ]: " << neuralNet.predict(input3) << std::endl;
    std::cout << "output for [ 1, 1 ]: " << neuralNet.predict(input4) << std::endl;

    return 0;
}