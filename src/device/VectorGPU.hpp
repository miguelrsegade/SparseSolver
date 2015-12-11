#pragma once

#include "../host/Vector.hpp"
#include <cuda.h>

class VectorGPU
{
friend class CsrMatrixGPU;
private:
    double* d_mData;
    int mSize;

public:
    VectorGPU( const Vector& hostVector);
    VectorGPU( int size, double value);
    VectorGPU( int size);
    ~VectorGPU();
    int GetSize() const;

    VectorGPU& operator=(const VectorGPU& otherVector);
    VectorGPU operator+(const VectorGPU& v1) const;
    VectorGPU operator-(const VectorGPU& v1) const;
    VectorGPU operator*(double a) const;
    // Norm
    double Norm(int p=2) const ;
    // Dot-product
    double operator*(const VectorGPU& v1) const;
    void copyToHost(Vector& hostVector);
};
