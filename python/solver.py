import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spl
import scipy.io as sio

inputfile =  (sys.argv[1])
# Read file into a scipy.sparse.coo.coo_matrix object
A = sio.mmread(inputfile)
# The solver only reads the matrix in csr
A = A.tocsr()
b = np.ones(5);

result = spl.spsolve(A,b)
print(result)


