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

## About
For about a year now, I have been working with neural networks and algorithms to train them. I have mastered the genetic algorithm, and I have previously followed along with [The Coding Train's back propagation series](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh), for back propagation in Python, but I could never personally, fully grasp the concept and calculus behind back propagation. Now I have completed first-year calculus at a college level, and I was able to understand [3blue1brown's back propagation calculus video](https://www.youtube.com/watch?v=tIeHLnjs5U8). Now I am able to create a fully working back propagation algorithm from pure understanding of the topic.
