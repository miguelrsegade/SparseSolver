#include "pssolver.hpp"


int main(int argc, char *argv[])
{
    Vector b(1000, 3.0);
    CsrMatrix A(1000, 2300);
    LinearSystem<CsrMatrix, Vector> LinSys(A, b);

    Vector result(1000);
    result = LinSys.SolveCG(1, 0.001);


    return 0;
}
