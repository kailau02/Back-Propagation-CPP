#include "Matrix.hpp"
#include "NN_math.hpp"
#include <iostream>

Matrix::Matrix(){
    array = std::vector<std::vector<float> >();
}
Matrix::Matrix(const int rows){
    array = std::vector<std::vector<float> >(rows, std::vector<float>(1));
}
Matrix::Matrix(const int rows, const int cols){
    array = std::vector<std::vector<float> >(rows, std::vector<float>(cols));
}
Matrix::Matrix(const std::vector<float> &vec){
    const int rows = vec.size();
    array = std::vector<std::vector<float> >(rows, std::vector<float>(1));
    for(int i = 0; i < rows; i++){
        array.at(i).at(0) = vec.at(i);
    }
}
Matrix::~Matrix(){}

int* Matrix::Shape(){
    int *shape = new int[2];
    shape[0] = array.size();
    shape[1] = array.at(0).size();
    return shape;
}

void Matrix::PrintMatrix(){
    int *shape = Shape();
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            std::cout << at(i, j) << ' ';
        }
        std::cout << std::endl;
    }
}

void Matrix::PrintShape(){
    int *shape = Shape();
    std::cout << '[' << shape[0] << 'x' << shape[1] << ']' << std::endl;
}

float& Matrix::at(const int row){
    return array.at(row).at(0);
}
float& Matrix::at(int row, int col){
    return array.at(row).at(col);
}

Matrix Matrix::T(){
    int *shape = Shape();
    Matrix transpose(shape[1], shape[0]);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            transpose.at(j, i) = at(i, j);
        }
    }
    return transpose;
}

void Matrix::Randomize(){
    int *shape = Shape();
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            at(i, j) = NN_math::rand_float();
        }
    }
}

Matrix Matrix::Map(float(*func)(float)){
    int *shape = Shape();
    Matrix mapMatrix(shape[0], shape[1]);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            mapMatrix.at(i, j) = func(at(i, j));
        }
    }
    return mapMatrix;
}

Matrix Matrix::operator+(Matrix &m2){
    int* shape = Shape();
    Matrix sumMatrix(shape[0], shape[1]);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            sumMatrix.at(i, j) = at(i, j) + m2.at(i, j);
        }
    }
    return sumMatrix;
}
Matrix Matrix::operator-(Matrix &m2){
    int* shape = Shape();
    Matrix diffMatrix(shape[0], shape[1]);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            diffMatrix.at(i, j) = at(i, j) - m2.at(i, j);
        }
    }
    return diffMatrix;
}
Matrix Matrix::operator*(Matrix &m2){
    int* shape = Shape();
    Matrix prodMatrix(shape[0], shape[1]);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            prodMatrix.at(i, j) = at(i, j) * m2.at(i, j);
        }
    }
    return prodMatrix;
}
Matrix Matrix::operator*(float f){
    int *shape = Shape();
    Matrix prodMatrix(shape[0], shape[1]);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < shape[1]; j++){
            prodMatrix.at(i, j) = at(i, j) * f;
        }
    }
    return prodMatrix;
}
Matrix Matrix::dot(Matrix &m2){
    int *m1Shape = Shape();
    int *m2Shape = m2.Shape();
    Matrix dotMatrix(m1Shape[0], m2Shape[1]);
    for(int i = 0; i < m1Shape[0]; i++){
        for(int j = 0; j < m2Shape[1]; j++){
            float sum = 0;
            for(int k = 0; k < m1Shape[1]; k++){
                sum += at(i, k) * m2.at(k, j);
            }
            dotMatrix.at(i, j) = sum;
        }
    }
    return dotMatrix;
}

std::vector<float> Matrix::toVector(){
    int size = Shape()[0];
    std::vector<float> vec(size);
    for(int i = 0; i < size; i++){
        vec.at(i) = at(i);
    }
    return vec;
}