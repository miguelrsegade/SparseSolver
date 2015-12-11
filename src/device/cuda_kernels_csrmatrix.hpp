#pragma once

__global__ void kernel_csrmatrix_add( const int nnz, 
                const double* mData, const double* Data, double* rData)
{

    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < nnz)
    {
        rData[ind] = mData[ind] + Data[ind];
    }

}

__global__ void kernel_csrmatrix_substract( const int nnz, 
                const double* mData, const double* Data, double* rData)
{

    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < nnz)
    {
        rData[ind] = mData[ind] - Data[ind];
    }
}

__global__ void kernel_csrmatrix_scalarmultiply( const int nnz,
        const double a, const double* mData,  double* rData)
{

    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < nnz)
    {
        rData[ind] = mData[ind]*a;
    }
}


__global__ void kernel_csrmatrix_matrixvector ( const int nrows, 
        const int nnz, const double* mData, const int* mColInd, 
        const int* mRowPtr, const double* v, double* result)
{
    
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if ( ind < nrows)
    {
        result[ind] = 0.0;
        for (int i=mRowPtr[ind]; i<mRowPtr[ind+1]; i++)
        {
           result[ind] = result[ind]  + mData[i]*v[mColInd[i]];
        }
        
    }
    

}
