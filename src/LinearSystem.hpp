#pragma once

#include "host/Vector.hpp"
#include "Matrix.hpp"

class LinearSystem
{
private:
    int mSize;
    // This have to be pointers because they don't have a constructor with 0 arguments
    CsrMatrix* mpA;
    Vector* mpb;

    // Don't allow the use of copy constructor -> private
    //LinearSystem(const LinearSystem& otherLinearSystem){};

public:
    LinearSystem(const CsrMatrix& A, const Vector& b);
    ~LinearSystem();

    Vector SolveCG(int maxiter, double tol);

};
