### Building a Neural Network from Scratch in C++

We will implement a basic neural network without using any existing ML libraries. We will create a layer class, a weights class, and integrate that into a neural network class which can take in data to be trained on. We want our neural network class to be trainable on both image and numerical data to produce a desired output. We will also write a driver class that demonstrates both neural network training and classification in action.

Many of the aformentioned parts will be worked on by multiple people at once. We hope to get a working version by November 19th, and will adjust accordingly if need be.

We currently do not plan to use any third-party libraries, though this is subject to change.

#####How to build the project:

We made a Makefile which makes this process simple. Simply run `make exec` in this repo's directory in the command line to build the executable.

#####How to use the resultant application:

Now that you have the executable, simply run `./bin/exec` in this repo's directory to run the driver code.

#####How to run the tests:

The tests are created and ran using a similar process described above. Run `make tests` to create the test suite executable and use `./bin/tests` to run it.