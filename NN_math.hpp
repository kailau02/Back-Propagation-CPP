#ifndef MATH_HPP
#define MATH_HPP

#include <vector>
#include <cmath>
#include <cstdlib>
#include <time.h>

namespace NN_math{

    static const float e = 2.71828;

    static int randDigit(){
        return rand() % 10;
    }

    static float rand_float(){
        float sign = rand() % 2 == 0 ? -1 : 1;
        float tmpFloat = static_cast<float>(randDigit());
        tmpFloat += static_cast<float>(randDigit()) * 0.1;
        tmpFloat += static_cast<float>(randDigit()) * 0.01;
        tmpFloat += static_cast<float>(randDigit()) * 0.001;
        tmpFloat *= sign;
        return tmpFloat * 0.25;
    }


    // Activation functions and derivatives
    static float sigmoid(float f){
        return 1.0 / (1.0 + pow(e, -f));
    }
    static float d_sigmoid(float f){
        return sigmoid(f) * (1 - sigmoid(f));
    }

    static float lrelu(float f){
        return f > 0 ? f : f * 0.5;
    }
    static float d_lrelu(float f){
        return f > 0 ? 1 : 0.5;
    }

    static float tanh(float f){
        return std::tanh(f);
    }
    static float d_tanh(float f) {
        float sh = 1.0 / std::cosh(f);
        return sh*sh;                     
    }
};

#endif /* MATH_HPP */
