#include "NN.hpp"
#include <iostream>

// Training data XOR
std::vector<float> in0 = {1.0, 0.0};
std::vector<float> t0 = {0.0};
std::vector<float> in1 = {0.0, 1.0};
std::vector<float> t1 = {0.0};
std::vector<float> in2 = {1.0, 1.0};
std::vector<float> t2 = {1.0};
std::vector<float> in3 = {0.0, 0.0};
std::vector<float> t3 = {1.0};

int main(){
    srand(time(NULL));

    // Form batch set
    struct std::vector<NN::BatchItem> batch;
    batch.push_back(NN::BatchItem(in0, t0));
    batch.push_back(NN::BatchItem(in1, t1));
    batch.push_back(NN::BatchItem(in2, t2));
    batch.push_back(NN::BatchItem(in3, t3));


    // Setup neural network
    std::vector<int> hiddenNodes {10, 10, 10};    
    NN network(2, hiddenNodes, 1, NN::Activation_lrelu);
    network.Randomize();
    network.learningRate = 0.000001;

    // Run same batch at 10,000 epoch's
    for(int i = 0; i < 10000; i++){
        network.TrainBatch(batch);
    }

    // Print results of network after training
    std::cout << network.FeedForward(in0).at(0) << std::endl;
    std::cout << network.FeedForward(in1).at(0) << std::endl;
    std::cout << network.FeedForward(in2).at(0) << std::endl;
    std::cout << network.FeedForward(in3).at(0) << std::endl;


    return 0;
}