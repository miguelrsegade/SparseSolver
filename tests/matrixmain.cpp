#include "pssolver.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main (int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage matrixmain [matrixfile]" << endl;
        exit(EXIT_FAILURE);
    }
    
    string filename = string(argv[1]);

    Matrix A(filename);
    cout << "Test methods " << endl;
    cout << "==========================" << endl;
    cout  << A.GetNrows() << endl;
    cout  << A.GetNcols() << endl;
    cout  << A.GetNnz() << endl;

    if (A.isSymmetric())
        cout << "Symmetric" << endl;
    else
        cout << "Non-Symmetric" << endl;
    if (A.isReal())
        cout << "Real " << endl;
    else 
        cout << "Complex" << endl;

    cout << "==========================" << endl;

    A.getProperties();

    return 0;

}
