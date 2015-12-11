#pragma once
#include <vector>
#include <iostream>

#include "VectorGPU.hpp"
#include "../Matrix.hpp"
#include "../host/CsrMatrix.hpp"

class CsrMatrixGPU: public Matrix
{

private:
   double *d_mData; 
   int *d_mColInd;
   int *d_mRowPtr;

public:

    CsrMatrixGPU(const CsrMatrix& otherMatrix);
    CsrMatrixGPU(int n, int nnz);
    ~CsrMatrixGPU();

    void copyToHost(CsrMatrix& hostMatrix);

    CsrMatrixGPU& operator=(const CsrMatrixGPU& otherMatrix);
    CsrMatrixGPU operator+(const CsrMatrixGPU& m1) const;
    CsrMatrixGPU operator-(const CsrMatrixGPU& m1) const;
    CsrMatrixGPU operator*(const double a) const;

    VectorGPU operator*(const VectorGPU& v);


};
