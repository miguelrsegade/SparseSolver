#include "pssolver.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main (int argc, char *argv[])
{
    if (argc != 1)
    {
        cerr << "Usage vectorGPUmain " << endl;
        exit(EXIT_FAILURE);
    }

    // Constructor fro another vector
    Vector b(10, 2.5);

    VectorGPU db(b);
    cout << db.GetSize();

    db = db*2;

    db.copyToHost(b);
    cout << b;

    Vector c(10, 7);
    VectorGPU dc(c);

    dc = dc+db;
    dc.copyToHost(c);
    cout << c;

    dc = dc*3;
    dc.copyToHost(c);
    cout << c;

    cout << dc.Norm() << endl;

    cout << dc*db << endl;

    return 0;
}
