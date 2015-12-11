#include <iostream>
#include "pssolver.hpp"

int main(int argc, char *argv[])
{
    
    if (argc != 2)
    {
        cerr << "Usage csrmatrixGPUmain [matrixfile]" << endl;
        exit(EXIT_FAILURE);
    }
    
    string filename = string(argv[1]);

    int n = 5;

    CsrMatrixGPU SparseA(filename);
    VectorGPU b(n, 1.0);

    LinearSystem<CsrMatrixGPU, VectorGPU> LinSys(SparseA, b);

    VectorGPU result(n);

    result = LinSys.SolveCG(10, 0.01);

    Vector hostb(n);
    result.copyToHost(hostb);
    
    cout << "Result: " << endl;
    cout << hostb << endl;


}
