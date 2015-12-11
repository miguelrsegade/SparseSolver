#include "pssolver.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main (int argc, char *argv[])
{
    if (argc != 1)
    {
        cerr << "Usage vectormain " << endl;
        exit(EXIT_FAILURE);
    }

    // Constructor
    Vector b(10, 2.5);
    cout << b;

    //Scalar multiplication and sum
    Vector c(b);
    c = c + 2*b;
    cout << c;

    // Binary -
    c = c - b;
    cout << c;

    // Unary +
    c = -c;
    cout << c;

    // Norm
    cout << c.Norm() << endl;

    //Dot Product
    cout << b*c << endl;


    return 0;
}
