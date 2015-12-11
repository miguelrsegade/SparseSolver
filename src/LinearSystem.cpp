#include "host/Vector.hpp"
#include "host/CsrMatrix.hpp"
#include "LinearSystem.hpp"
#include <cassert>

// Constructor - Matrix and vector remain unchanged
LinearSystem::LinearSystem(const CsrMatrix& A, const Vector& b)
{
    // Check size compatibility
    assert(A.GetNrows() == b.GetSize());
    mSize = A.GetNrows();
    //TODO Using double the memory??
    mpA = new CsrMatrix(A);
    mpb = new Vector(b);
}

// Destructor
LinearSystem::~LinearSystem()
{
    delete mpA;
    delete mpb;
}

Vector LinearSystem::SolveCG(int maxiter, double tol)
{
    // References to make the code easier
    CsrMatrix& rA = *mpA;
    Vector& rb = *mpb;

    // Residue Vector and old residue
    Vector res(mSize);
    Vector resold(mSize);
    // Conjugated direction vector
    Vector dir(mSize);
    // Initial guess
    Vector x(mSize, 1.0);

    res = rb - rA*x;
    dir = res;

    double alpha, beta;
    
    for (int i=1; i<maxiter; i++)
    {
        cout << dir << endl;
        alpha = res*res / (dir*(rA*dir));
        x = x + alpha*dir;
        resold = res;
        res = res - alpha*(rA*dir);
        beta = res*res / (resold*resold);
        dir = res + beta*dir;
        if ( ( res.Norm() /  resold.Norm() ) < tol)
            break;
    }

    return x;
    
    
}
