#include "CsrMatrix.hpp"
#include "Vector.hpp"
#include "auxFuncs.hpp"

#include <cassert>
#include <iostream>
#include <regex>
#include <vector>
#include <algorithm>

//Constructor with int
CsrMatrix::CsrMatrix(int n, int nnz) : Matrix(n,nnz),
    mData(new double[nnz]), mColInd(new int[nnz]),
    mRowPtr(new int[n]) {}


//Constructor with another CsrMatrix
CsrMatrix::CsrMatrix(const CsrMatrix& otherMatrix) : Matrix(otherMatrix),
    mData(new double[mNnz]), mColInd(new int[mNnz]), 
    mRowPtr(new int[mNrows+1])
{

   for (int i=0; i < mNnz; i++)
   {
        mData[i] = otherMatrix.mData[i];
        mColInd[i] = otherMatrix.mColInd[i];
   }
   for (int i=0; i < (mNrows+1); i++)
   {
        mRowPtr[i] = otherMatrix.mRowPtr[i];
   }
}

CsrMatrix::CsrMatrix(const string& filename) :
    Matrix(filename)
{

    // Allocate the arrays
    mData = new double[mNnz];
    mColInd = new int[mNnz];
    mRowPtr = new int[mNrows+1];

    // Read the file
    string line; // Where each line will be stored
    istringstream linestream; // To convert line to stream
    int linenumber = 3; // We begin in line 3
    int index; // linenumber - 3
    int row, col;// Keeping track of the rows and cols
    int rowold = 0;
    // The file is in column order so we need to rearrange the vectors
    // after reading with an order vector
    vector<int> order;
    try
    {
        while (true)
        {
            getline(mFile, line);
            // Check syntax
            if (!regex_match (line, regex("^((?:(?:[1-9][0-9]*\\s+?){2})"
                                    "-?[0-9.]+(?:\\+|\\-)[0-9]+)")))
            {
                throw linenumber;
            }
            // Convert to stream
            linestream.str(line);

            // Set the three arrays of the matrix
            index = linenumber - 3;
            // 0-index values from 1-index file
            linestream >> row; row--;
            // we store the rows in the order vector to get then the 
            // neccesary order in row-major order
            order.push_back(row);
            linestream >> col; col--;
            mColInd[index] = col;

            linestream >> mData[index];

            linenumber++;
            // We can't exit with while(!mFile.eof) because that raises 
            // an exception
            if(mFile.peek() == EOF) break;
        }

        // First order the order vector (without saving it) and then
        // store the original indexes in iorder
        vector<int> iorder;
        iorder = sort_indexes(order);
        // Reorder mColInd and mData with iorder
        for (int i=0; i< iorder.size(); i++)
        {
            if ( iorder[i] == i)
            {
                continue;
            }
            else
            {
                swap(mColInd[i], mColInd[iorder[i]]);
                swap(mData[i], mData[iorder[i]]);
                swap(iorder[i], iorder[iorder[i]]);
            }
        
        }
        
        // RowPtr object
        // =============================
        // Keep track of the rowptrindex
        int rowptrindex = 0;
        mRowPtr[rowptrindex]=0; 
        sort(order.begin(), order.end());
        for (int i = 1; i < order.size(); i++)
        {
            if ( order[i] > order[i-1] )
            {
                rowptrindex++;
                mRowPtr[rowptrindex] = i;
            }
        }
         

        mRowPtr[mNrows] = mNnz; // By convention

    }
    catch (ifstream::failure e)
    {
        cout << "Exception reading file" << endl;
    }
    catch (int linenumber)
    {
        cout << "Bad syntax on line " << linenumber << endl;
    }

    mFile.close();
}

// Destructor
CsrMatrix::~CsrMatrix()
{
    delete[] mData;
    delete[] mColInd;
    delete[] mRowPtr;
}

// 1-base indexing for debugging
double CsrMatrix::operator()(int i, int j) 
{                                                            

    i = i-1; j = j-1;                                        
    //Assert we access a non zero element                    
    assert ( i >= 0);                                        
    assert ( j >= 0);                                        
    assert ( i < mNrows);                                     
    assert ( j < mNcols);                                     
    int aux = mRowPtr[i];                                   
    int rowWidth = mRowPtr[i+1] - mRowPtr[i];              
    int k;                                                   
    for (k = 0; k < rowWidth; k++)                           
    {                                                        
        if(mColInd[aux+k] == j)                             
            break;                                           
    }                                                        
    if (mColInd[aux+k] == j)                                
        return mData[aux+k];                                  
    else                                                     
        return 0.0;                                          
}                                                            

CsrMatrix& CsrMatrix::operator=(const CsrMatrix& otherMatrix)
{
    assert (mNrows == otherMatrix.mNrows &&
            mNcols == otherMatrix.mNcols );
    for (int i=0; i < mNnz; i++)               
    {                                          
             mData[i] = otherMatrix.mData[i];        
                  mColInd[i] = otherMatrix.mColInd[i];
    }                                          
    for ( int i=0; i<mNrows+1; i++)             
    {                                          
            mRowPtr[i] = otherMatrix.mRowPtr[i]; 
    }                                          
    return *this;                              
            

}
// Binary + operator                                      
// Only works for same matrix 
CsrMatrix CsrMatrix::operator+(const CsrMatrix& m1) const 
{                                                         
    assert(mNrows == m1.mNcols && mNrows == m1.mNrows);
    CsrMatrix mat(mNrows, mNnz);
    for ( int i=0; i<mNnz; i++)                           
    {                                                     
       mat.mData[i] = mData[i] + m1.mData[i];                
       mat.mColInd[i] = mColInd[i];                     
    }                                                     
    for ( int i=0; i<mNrows+1; i++)                        
    {                                                     
        mat.mRowPtr[i] = m1.mRowPtr[i];                 
    }                                                     
                                                          
    return mat;                                           
}                                                         

// Binary - operator
// Solo funciona para matrices con el mismo perfil
CsrMatrix CsrMatrix::operator-(const CsrMatrix& m1) const
{
    assert(mNrows == m1.mNrows);
    CsrMatrix mat(mNrows, mNnz);
    for ( int i=0; i<mNnz; i++)
    {
       mat.mData[i] = mData[i] - m1.mData[i];
       mat.mColInd[i] = mColInd[i]; 
    }
    for ( int i=0; i<mNrows+1; i++)
    {
        mat.mRowPtr[i] = m1.mRowPtr[i];
    }

    return mat;
}

//Scalar Multiplication
CsrMatrix CsrMatrix::operator*(double a) const
{
    CsrMatrix mat(mNrows, mNnz); 
    for ( int i=0; i <mNnz; i++)
    {
        mat.mData[i] = mData[i]*a;
        mat.mColInd[i] = mColInd[i];
    }
    for ( int i=0; i<mNrows+1; i++)
    {
        mat.mRowPtr[i] = mRowPtr[i];
    }
    return mat;
   
}

// =====================================================
// Friend functions
// =====================================================

// Friend Functions
// Matrix Vector
Vector operator*(const CsrMatrix& m, const Vector& v)
{
    int size = v.GetSize();
    assert(size == m.GetNrows());
    Vector result_vector(size);

    for ( int i = 0; i < size; i++)
    {
        for ( int k = m.mRowPtr[i]; k < m.mRowPtr[i+1]; k++)
        {
            result_vector[i] = result_vector[i] + m.mData[k]*v.Read(m.mColInd[k]);
        }
    }
    // Last row
    return result_vector;


}

// Print matrix for debuggin                                 
ostream& operator<<(ostream& os, const CsrMatrix& Mat)       
{                                                            
    os << "Number of rows" << endl;                   
    os << Mat.mNrows << endl;                                
    os << "Number of cols" << endl;                   
    os << Mat.mNcols << endl;                                
    os << "Number of nonzeros" << endl;                   
    os << Mat.mNnz << endl;                                
    os << "Data" << "\t" << "ColInd" << "\t" << "RowPtr" << endl;
    for (int i = 0; i < Mat.mNnz; i++)                       
    {                                                        
    
        if (i <= Mat.mNrows)
            os << Mat.mData[i] << "\t" << Mat.mColInd[i] << "\t" 
                               << Mat.mRowPtr[i] << endl;
        else
            os << Mat.mData[i] << "\t" << Mat.mColInd[i] << endl;
    }                                                        
    return os;                                               
}
                                                             
