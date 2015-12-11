#pragma once

#include "../pssolver.hpp"
#include <cassert>

template <typename Mat, typename Vec>
class LinearSystem
{
private:
    int mSize;
    // This have to be pointers because they don't have a constructor with 0 arguments
    Mat* mpA;
    Vec* mpb;

    // Don't allow the use of copy constructor -> private
    //LinearSystem(const LinearSystem& otherLinearSystem){};

public:
    LinearSystem(Mat& A, Vec& b);
    // No objects allocated with new
//    ~LinearSystem();

    Vec SolveCG(int maxiter, double tol);

};

// Constructor - Matrix and vector remain unchanged
template <typename Mat, typename Vec>
LinearSystem<Mat, Vec>::LinearSystem (Mat& A, Vec& b) 
{
    // Check size compatibility
    assert(A.GetNrows() == b.GetSize());
    mSize = A.GetNrows();
    //TODO Using double the memory??
    //mpA = new Mat(A);
    //mpb = new Vec(b);
    mpA = &A;
    mpb = &b;
}

// Destructor
//template <typename Mat, typename Vec>
//LinearSystem<Mat, Vec>::~LinearSystem()
//{
//}

template <typename Mat, typename Vec>
Vec LinearSystem<Mat, Vec>::SolveCG(int maxiter, double tol)
{
    // References to make the code easier
    Mat& rA = *mpA;
    Vec& rb = *mpb;

    // Residue Vector and old residue
    Vec res(mSize);
    Vec resold(mSize);
    // Conjugated direction vector
    Vec dir(mSize);
    // Initial guess
    Vec x(mSize, 1.0);

    res = rb - rA*x;
    dir = res;

    double alpha, beta;
    
    for (int i=1; i<maxiter; i++)
    {
        alpha = res*res / (dir*(rA*dir));
        x = x + dir*alpha;
        resold = res;
        res = res - (rA*dir)*alpha;
        beta = res*res / (resold*resold);
        dir = res + dir*beta;
        if ( ( res.Norm() /  resold.Norm() ) < tol)
            break;
    }

    return x;
    
    
}
