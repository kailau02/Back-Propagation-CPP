# Back-Propagation-CPP

## Instructions
1. Download repository and locate it on terminal
2. Type: `g++ -std=c++11 *.cpp -o back`
3. Type: `./back`

After pressing enter, the program will take a few seconds to train the network. Once training is complete, inputs will be fed forward and the outputs will appear on the terminal.
## Files
- `main.cpp` holds the XOR test training data.
- `Matrix.hpp` and `Matrix.cpp` hold the `Matrix` class. This class has matrix operations and other functions.
- `NN.hpp` and `NN.cpp` hold the `NN` class. This class holds weights and biases, and it performs forward propagation, back propagation.
- `NN_math.hpp` holds the functions required for randomized weights and biases, and it has activation and derivative funcitons.
