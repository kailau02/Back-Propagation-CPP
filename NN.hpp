#ifndef NN_HPP
#define NN_HPP

#include "Matrix.hpp"

class NN{
private:
    NN() = delete;

    const int hiddenSize;

    // Holds the values which are calculated during 'ForwardPropagate'
    struct NNOutputs{
        std::vector<Matrix> preActivation;
        std::vector<Matrix> postActivation;
    };
    // Holds weights and biases for network or for delta values
    struct NNValues{
        std::vector<Matrix> weights;
        std::vector<Matrix> biases;
    };

    float (*activation)(float); // Activation function
    float (*d_activation)(float); // Derivative of activation function

    void CreateNN(int, std::vector<int>&, int); // Initializes weights and biases for the network
    NNOutputs ForwardPropagate(const std::vector<float>&); // Feeds through inputs, returns each layer's outputs of before and after activation function
    NNValues BackPropagate(const std::vector<float>&, const std::vector<float>&); // Takes inputs and targets and performs back propagation algorithm to determine delta weights and biases

public:
    enum ActivationType{
        Activation_sigmoid,
        Activation_lrelu,
        Activation_tanh
    };

    // Holds a data structure for inputs and targets
    struct BatchItem{
        BatchItem(std::vector<float>&inputs, std::vector<float>&targets) : inputs(inputs), targets(targets){}
        std::vector<float> inputs;
        std::vector<float> targets;
    };

    NN(int, std::vector<int>&, int); // Construct network with input nodes, a vector of hidden layer nodes, and output nodes
    NN(int, std::vector<int>&, int, ActivationType); // Construct network with node#'s and provide an activation function (sigmoid is default)
    ~NN();

    float learningRate;
    struct NNValues values; // Holds the network's current weights and biases

    void Randomize(); // Randomizes all weights and biases in network
    void SetActivation(ActivationType); // Set activation & derivative functions from the following: sigmoid, leaky relu, and tanh
    std::vector<float> FeedForward(const std::vector<float>&); // Feeds inputs into network and returns final outputs
    void Train(const std::vector<float>&, const std::vector<float>&); // Gets delta weights and biases from 'BackPropagate' and applies these delta values to the current network
    void TrainBatch(const std::vector<BatchItem>&); // Applies sum of delta weights and biases after back propagating multiple sets of inputs & outputs
};

#endif /* NN_HPP */
