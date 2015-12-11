#include "CsrMatrixGPU.hpp"
#include "utils.h"
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernels_csrmatrix.hpp"

const int BLOCKSIZE = 4;

CsrMatrixGPU::CsrMatrixGPU ( const CsrMatrix& hostMatrix) 
    : Matrix(hostMatrix)
{
    checkCudaErrors(cudaMalloc(&d_mData, mNnz*sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mColInd, mNnz*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_mRowPtr, (mNrows+1)*sizeof(int)));

    checkCudaErrors(cudaMemcpy( d_mData, hostMatrix.mData,
                            mNnz*sizeof(double),
                            cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_mColInd, hostMatrix.mColInd,
                            mNnz*sizeof(int),
                            cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_mRowPtr, hostMatrix.mRowPtr,
                            (mNrows+1)*sizeof(int),
                            cudaMemcpyHostToDevice));


}

CsrMatrixGPU::CsrMatrixGPU (int n, int nnz) : Matrix(n, nnz)
{

    checkCudaErrors(cudaMalloc(&d_mData, mNnz*sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mColInd, mNnz*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_mRowPtr, (mNrows+1)*sizeof(int)));

}

CsrMatrixGPU::~CsrMatrixGPU()
{
   checkCudaErrors(cudaFree(d_mData));
}
void CsrMatrixGPU::copyToHost(CsrMatrix& hostMatrix)
{
    checkCudaErrors(cudaMemcpy ( hostMatrix.mData, d_mData, 
                            mNnz*sizeof(double),
                            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy ( hostMatrix.mColInd, d_mColInd,
                            mNnz*sizeof(int),
                            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy ( hostMatrix.mRowPtr, d_mRowPtr,
                            (mNrows+1)*sizeof(int),
                            cudaMemcpyDeviceToHost));

}

CsrMatrixGPU& CsrMatrixGPU::operator=(const CsrMatrixGPU& otherMatrix)
{
    assert(mNrows == otherMatrix.mNrows &&
            mNcols == otherMatrix.mNrows);

    checkCudaErrors(cudaMemcpy ( d_mData, otherMatrix.d_mData,
                            mNnz*sizeof(double),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy ( d_mColInd, otherMatrix.d_mColInd,
                            mNnz*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy ( d_mRowPtr, otherMatrix.d_mRowPtr,
                            (mNrows+1)*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    return *this;
}

// Sum of two matrices
// Only works for same profile matrices
CsrMatrixGPU CsrMatrixGPU::operator+(const CsrMatrixGPU& m1) const
{
    CsrMatrixGPU result(mNrows, mNnz);
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize(mNnz / BLOCKSIZE +1 );
    kernel_csrmatrix_add <<<GridSize, BlockSize>>> (mNnz,
                        d_mData, m1.d_mData, result.d_mData);

    // ColInd and RowPtr are equal 
    checkCudaErrors(cudaMemcpy ( result.d_mColInd, d_mColInd, 
                            mNnz*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy ( result.d_mRowPtr, d_mRowPtr, 
                            (mNrows+1)*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    return result;
    
}

// Diferences of two matrices
// Only works for same profile matrices
CsrMatrixGPU CsrMatrixGPU::operator-(const CsrMatrixGPU& m1) const
{
    CsrMatrixGPU result(mNrows, mNnz);
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize(mNnz / BLOCKSIZE +1 );
    kernel_csrmatrix_substract <<<GridSize, BlockSize>>> (mNnz,
                        d_mData, m1.d_mData, result.d_mData);

    // ColInd and RowPtr are equal 
    checkCudaErrors(cudaMemcpy ( result.d_mColInd, d_mColInd, 
                            mNnz*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy ( result.d_mRowPtr, d_mRowPtr, 
                            (mNrows+1)*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    return result;
    
}

// Only works for same profile matrices
CsrMatrixGPU CsrMatrixGPU::operator*(const double a) const
{
    CsrMatrixGPU result(mNrows, mNnz);
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize(mNnz / BLOCKSIZE +1 );
    kernel_csrmatrix_scalarmultiply <<<GridSize, BlockSize>>> (mNnz, a,
                        d_mData, result.d_mData);

    // ColInd and RowPtr are equal 
    checkCudaErrors(cudaMemcpy ( result.d_mColInd, d_mColInd, 
                            mNnz*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy ( result.d_mRowPtr, d_mRowPtr,
                            (mNrows+1)*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    return result;
    
}
VectorGPU CsrMatrixGPU::operator*(const VectorGPU& v)
{
    VectorGPU result(mNrows);
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize(mNnz / BLOCKSIZE +1 );
    kernel_csrmatrix_matrixvector <<< GridSize, BlockSize >>> (mNrows, mNnz,
            d_mData, d_mColInd, d_mRowPtr, v.d_mData, result.d_mData);

    return result;
}
