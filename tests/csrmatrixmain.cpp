#include "pssolver.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main (int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage csrmatrixmain [matrixfile]" << endl;
        exit(EXIT_FAILURE);
    }
    
    string filename = string(argv[1]);

    CsrMatrix A(filename);
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

    cout << "==========================" << endl;
    cout << "Output operator " << endl;
    cout << A;
    cout << "==========================" << endl;
    cout << "Copy Constructor" << endl;
    CsrMatrix C(A);
    cout << C;

    cout << "1 base indexing " << endl;
    for (int i =1; i < C.GetNrows() ; i++)
    {
        for (int j=1; j < C.GetNcols(); j++)
        {
            cout << C(i,j) << "\t";        
        }
        cout << endl;
    }

    CsrMatrix D(5, 13);

    cout << "D Matrix " << endl;
    cout << "Assingment operator " << endl;
    cout << "=========================" << endl;
    D = C;
    cout << D;

    cout << "Binary sum " << endl;
    cout << "=========================" << endl;
    cout << D+A << endl;

    cout << "Binary substraction " << endl;
    cout << "=========================" << endl;
    cout << D-A << endl;

    cout << "Scalar multiplication" << endl;
    cout << "=========================" << endl;
    cout << D*3 << endl;

    cout << "Matrix vector multiplication " << endl;
    cout << "=========================" << endl;
    Vector b(5, 1);
    cout << D*b << endl;



    


    return 0;

}
