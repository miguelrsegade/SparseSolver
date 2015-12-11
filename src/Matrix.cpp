#include "Matrix.hpp"

#include <fstream>
#include <regex>
#include <string>
#include <iostream>

using namespace std;

Matrix::Matrix( int n, int nnz) : mNrows(n), mNcols(n), mNnz(nnz) {}

Matrix::Matrix( const Matrix& otherMatrix ) : 
    mNrows(otherMatrix.mNrows), mNcols(otherMatrix.mNcols), 
    mNnz(otherMatrix.mNnz), mIsSymmetric(otherMatrix.mIsSymmetric),
    mIsReal(otherMatrix.mIsReal)  {} 

Matrix::Matrix( const string& filename)
{
    // Enable exceptions when 
    // failbit -> logical error on i/o operation
    // badbit -> read/writing error on i/o operation
    mFile.exceptions ( ifstream::failbit | ifstream::badbit );
    try
    {
        mFile.open(filename); 
        string line;
        // Start a linenumber count 
        int linenumber = 1; 

        // Read first line
        getline(mFile, line);
        // regex_search looks for substrings matchs
        // regex_match matchs whole string
        // Check syntax of the line
        if ( !regex_match ( line, regex("%%MatrixMarket matrix"
                                        " coordinate (real|complex)"
                                        " (symmetric|non-symmetric)")))
        {
            throw linenumber;
        }
        if ( regex_search (line, regex("symmetric")) )
            mIsSymmetric = true;
        else
            mIsSymmetric = false;
        if ( regex_search (line, regex("real")) )
            mIsReal = true;
        else 
            mIsReal = false;

        // Next line
        linenumber++;
        getline(mFile, line);
        // Check syntax 
        // ?: make the inner group a non-capturing group
        // \\s double escape to get the escape character to the engine
        // Accepts any number of blanks between numbers
        if ( !regex_match (line, regex("^((?:[1-9][0-9]*\\s*?){3})")))
        {
            throw linenumber;
        }
        istringstream linestream(line);
        linestream >> mNrows >> mNcols >> mNnz;
        // The stream is in position for reading the data (line 3)
        // That is done in the subclasses
        // getline(mFile,line);

    }
    catch (ifstream::failure e)
    {
        cout << "Exception opening file" << endl;
    }
    catch (int linenumber)
    {
        cerr << "Bad syntax on line " << linenumber << endl;
    }

    }


int Matrix::GetNrows() const
{
    return mNrows;
}
int Matrix::GetNcols() const
{
    return mNcols;
}
int Matrix::GetNnz() const
{
    return mNnz;
}

bool Matrix::isSymmetric()
{
    return mIsSymmetric;
}

bool Matrix::isReal()
{
    return mIsReal;
}

void Matrix::getProperties()
{
    cout << "Matrix Properties " << endl;
    if (mIsSymmetric)                  
        cout << "Symmetric " ;
    else                                  
        cout << "Non-Symmetric " ;
    if (mIsReal)                       
        cout << "Real " << endl;          
    else                                  
        cout << "Complex " << endl;

    cout << "Number of Rows = " << mNrows << endl;
    cout << "Number of Columns = " << mNcols << endl;
    cout << "Number of Non zero elements = " << mNnz << endl;

}


