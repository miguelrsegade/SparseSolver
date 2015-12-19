#pragma once
#include "../Matrix.hpp"
#include "Vector.hpp"

class CsrMatrix: public Matrix
{

friend class CsrMatrixGPU;

private: 
    double *mData;
    int *mColInd;
    int *mRowPtr;

public:
    CsrMatrix(int n, int nnz);
    CsrMatrix(const string& filename);
    CsrMatrix(const CsrMatrix& otherMatrix);
    ~CsrMatrix();

    double operator()(int i, int j); //1-base indexing 
    CsrMatrix& operator=(const CsrMatrix& otherMatrix); //Assingment
	CsrMatrix operator+(const CsrMatrix& m1) const; 
    CsrMatrix operator-(const CsrMatrix& m1) const; // binary -
    CsrMatrix operator*(double a) const;

    // Output operator
    friend Vector operator*(const CsrMatrix& m, const Vector& v);
    friend ostream& operator<<(ostream& os, const CsrMatrix& Mat) ;

};

//Prototype signatures for friend operators
ostream& operator<<(ostream& os, const CsrMatrix& Mat); //Output operator
Vector operator*(const CsrMatrix& m, const Vector& v);
