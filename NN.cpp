#include "NN.hpp"
#include "NN_math.hpp"
#include <iostream>

NN::NN(int inputNodes, std::vector<int> &hiddenNodes, int outputNodes) : hiddenSize(hiddenNodes.size()), learningRate(0.01){
    CreateNN(inputNodes, hiddenNodes, outputNodes);
    SetActivation(Activation_sigmoid);
}
NN::NN(int inputNodes, std::vector<int> &hiddenNodes, int outputNodes, ActivationType type) : hiddenSize(hiddenNodes.size()), learningRate(0.01){
    CreateNN(inputNodes, hiddenNodes, outputNodes);
    SetActivation(type);
}

NN::~NN(){}

void NN::CreateNN(int inputNodes, std::vector<int> &hiddenNodes, int outputNodes){
    // Initialize values from input to first hidden layer
    values.weights.push_back(Matrix(hiddenNodes.front(), inputNodes));
    values.biases.push_back(Matrix(hiddenNodes.front()));

    // Loop through and initialize values from first hidden layer to last hidden layer
    for(int i = 0; i + 1 < hiddenSize; i++){
        values.weights.push_back(Matrix(hiddenNodes.at(i + 1), hiddenNodes.at(i)));
        values.biases.push_back(Matrix(hiddenNodes.at(i + 1)));
    }

    // Initialize values from last hidden layer to output
    values.weights.push_back(Matrix(outputNodes, hiddenNodes.back()));
    values.biases.push_back(Matrix(outputNodes));
}

void NN::Randomize(){
    for(int i = 0; i < values.weights.size(); i++){
        values.weights.at(i).Randomize();
        values.biases.at(i).Randomize();
    }
}

void NN::SetActivation(ActivationType type){
    switch (type){
    case Activation_lrelu: // Set activation functions to leaky relu
        activation = &NN_math::lrelu;
        d_activation = &NN_math::d_lrelu;
        break;
    
    case Activation_tanh: // Set activation functions to tanh
        activation = &NN_math::tanh;
        d_activation = &NN_math::d_tanh;
        break;

    case Activation_sigmoid:
    default: // Set activation functions to sigmoid
        activation = &NN_math::sigmoid;
        d_activation = &NN_math::d_sigmoid;
        break;
    }
}

NN::NNOutputs NN::ForwardPropagate(const std::vector<float> &inputs){
    struct NNOutputs outputs;

    // Get first layer outputs
    Matrix inputMatrix(inputs);
    outputs.preActivation.push_back(values.weights.front().dot(inputMatrix) + values.biases.front());
    outputs.postActivation.push_back(outputs.preActivation.back().Map(activation));

    // Get hidden layer outputs
    for(int i = 0; i + 1 < hiddenSize; i++){
        outputs.preActivation.push_back(values.weights.at(i + 1).dot(outputs.postActivation.back()) * values.biases.at(i + 1));
        outputs.postActivation.push_back(outputs.preActivation.back().Map(activation));
    }

    // Get output layer outputs
    outputs.preActivation.push_back(values.weights.back().dot(outputs.postActivation.back()) + values.biases.back());
    outputs.postActivation.push_back(outputs.preActivation.back().Map(activation));

    return outputs;
}

// vvvvvvvvv THE MAGIC... calculus vvvvvvvvv
NN::NNValues NN::BackPropagate(const std::vector<float> &inputs, const std::vector<float> &targets){
    struct NNOutputs outputs = ForwardPropagate(inputs);
    struct NNValues deltas;

    std::vector<Matrix> errors;

    // Push back output errors
    Matrix targetsMatrix(targets);
    errors.push_back(outputs.postActivation.back() - targetsMatrix);

    // Push back hidden errors starting with the highest index
    for(int i = hiddenSize; i > 0; i--){
        errors.push_back(values.weights.at(i).T().dot(errors.back()));
    }

    // Find deltas for each value layer
    for(int i = 0; i <= hiddenSize; i++){
        int errorIndex = hiddenSize - i; // Since errors were pushed back from right to left, index needs to be reversed
        
        Matrix d_outputs = outputs.preActivation.at(i).Map(d_activation); // da/dz

        Matrix negativeGradient = errors.at(errorIndex) * d_outputs * -learningRate; // dc/da * da/dz * -learningRate

        Matrix outputs_T;
        if(i == 0){
            outputs_T = Matrix(inputs).T();
        }
        else{
            outputs_T = outputs.postActivation.at(i - 1).T();
        }
        deltas.weights.push_back(negativeGradient.dot(outputs_T));
        deltas.biases.push_back(negativeGradient);
    }
    return deltas;
}

std::vector<float> NN::FeedForward(const std::vector<float> &inputs){
    struct NNOutputs outputs = ForwardPropagate(inputs);
    return outputs.postActivation.back().toVector();
}

void NN::Train(const std::vector<float> &inputs, const std::vector<float> &targets){
    NNValues deltas = BackPropagate(inputs, targets);
    for(int i = 0; i < deltas.weights.size(); i++){
        values.weights.at(i) = values.weights.at(i) + deltas.weights.at(i);
        values.biases.at(i) = values.biases.at(i) + deltas.biases.at(i);
    }
}

void NN::TrainBatch(const std::vector<BatchItem> &batchItem){
    const int batchSize = batchItem.size();
    NNValues deltas;
    for(int i = 0; i < batchSize; i++){
        NNValues setDeltas = BackPropagate(batchItem.at(i).inputs, batchItem.at(i).targets);
        if(deltas.weights.size() == 0){
            deltas = setDeltas;
        }
        else{
            for(int j = 0; j < deltas.weights.size(); j++){
                deltas.weights.at(i) = deltas.weights.at(i) + setDeltas.weights.at(i);
                deltas.biases.at(i) = deltas.biases.at(i) + setDeltas.biases.at(i);
            }
            
        }
    }

    for(int i = 0; i < deltas.weights.size(); i++){
        values.weights.at(i) = values.weights.at(i) + deltas.weights.at(i);
        values.biases.at(i) = values.biases.at(i) + deltas.biases.at(i);
    }
}
