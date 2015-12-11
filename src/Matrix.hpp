#pragma once

#include <fstream>
#include <string>

using namespace std;

class Matrix
{
protected:
    int mNrows;
    int mNcols;
    int mNnz;

    ifstream mFile;

    bool mIsSymmetric;
    bool mIsReal;

public:
    Matrix( int n, int nnz);
    Matrix( const Matrix& otherMatrix);
    Matrix( const string& filename);
    int GetNrows() const; 
    int GetNcols() const;
    int GetNnz() const;

    bool isSymmetric();
    bool isReal();

    void getProperties();

    // Virtual functions
    //virtual Matrix& operator=(const Matrix& otherMatrix) = 0;
    //virtual Matrix& operator+(const Matrix& m1) const = 0;
    //virtual Matrix& operator-(const Matrix& m1) const = 0;
    //virtual Matrix& operator*(double) const = 0;
    //virtual Vector operator*(const Vector v) = 0;
    
};




