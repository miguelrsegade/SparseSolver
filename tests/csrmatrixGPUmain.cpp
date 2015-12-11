#include "pssolver.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main (int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage csrmatrixGPUmain [matrixfile]" << endl;
        exit(EXIT_FAILURE);
    }
    
    string filename = string(argv[1]);

    CsrMatrix A(filename);
    CsrMatrix B(5, 13);
    CsrMatrix C(5, 13);
    CsrMatrixGPU dA(A);

    dA.copyToHost(A);
    cout << A;

    CsrMatrixGPU dB(5, 13);
    dB = dA;

    dB.copyToHost(B);
    cout << B;

    CsrMatrixGPU dC(5, 13);
    dC =  dA;
    cout << "Sum of vectors " << endl;
    dC = dB + dA;
    dC.copyToHost(C);
    cout << C;

    cout << "Diference of vectors " << endl;
    dC = dB - dA;
    dC.copyToHost(C);
    cout << C;

    cout << "Multiply by a scalar" << endl;
    dC = dB*3;
    dC.copyToHost(C);
    cout << C;

    cout << "Dot product" << endl;
    VectorGPU db(5, 1.0);
    VectorGPU dresult(5, 1.0);

    dresult = dC*db;
    Vector result(5, 1.0);
    dresult.copyToHost(result);
    cout << result;


}
