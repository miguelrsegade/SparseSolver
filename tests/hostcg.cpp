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

    CsrMatrix SparseA(filename);
    Vector b(n, 1.0);

    LinearSystem LinSys(SparseA, b);

    Vector result(n);

    result = LinSys.SolveCG(10, 0.01);
    
    cout << "Result: " << endl;
    cout << result << endl;


}
