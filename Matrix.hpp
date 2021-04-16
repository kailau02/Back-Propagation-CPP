#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix{
private:
    std::vector<std::vector<float> > array;
public:
    Matrix();
    Matrix(const int); // Construct a [i x 1] matrix (basically a vector with matrix functionality)
    Matrix(const int, const int); // Construct a [i x j] matrix
    Matrix(const std::vector<float>&); // Construct a vector shaped matrix with values from a pre-defined std::vector
    ~Matrix();

    int* Shape();

    void PrintMatrix();
    void PrintShape();

    float& at(const int); // Returns a reference to the value at index (typically used in a vector style matrix [i, 0])
    float& at(const int, const int); // Returns reference to value at [i, j]

    Matrix T(); // Transpose matrix
    void Randomize(); // Randomize all values in matrix
    Matrix Map(float(*func)(float)); // Maps a function to all values in matrix


    Matrix operator+(Matrix&); // Add two matrices
    Matrix operator-(Matrix&); // Subtract two matrices
    Matrix operator*(Matrix&); // Multiply two matrices at cooresponding values
    Matrix operator*(float); // Multiply a matrix by a scalar value
    Matrix dot(Matrix&); // Dot product of two matrices

    std::vector<float> toVector(); // Return this matrix as a vector (typically used for an [i x 1] shaped matrix)
};

#endif /* MATRIX_HPP */
