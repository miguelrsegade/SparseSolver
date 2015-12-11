#include "utils.h"
#include "VectorGPU.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_kernels_vector.hpp"


//Remove Later
#include "../host/Vector.hpp"

const int BLOCKSIZE = 4;

VectorGPU::VectorGPU(const Vector& hostVector)
{
    mSize = hostVector.mSize;

    checkCudaErrors(cudaMalloc(&d_mData, mSize*sizeof(double)));

    checkCudaErrors(cudaMemcpy( d_mData,
                hostVector.mData, 
                hostVector.mSize*sizeof(double), 
                cudaMemcpyHostToDevice));
}

VectorGPU::VectorGPU(int size, double value)
{
    mSize = size;

    checkCudaErrors(cudaMalloc(&d_mData, mSize*sizeof(double)));
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_fill_vector <<<GridSize, BlockSize>>>(mSize, d_mData, value);
}

VectorGPU::VectorGPU(int size)
{
    mSize = size;

    checkCudaErrors(cudaMalloc(&d_mData, mSize*sizeof(double)));
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_fill_vector <<<GridSize, BlockSize>>>(mSize, d_mData, 0.0);


}

//Asigment operator
VectorGPU& VectorGPU::operator=(const VectorGPU& otherVector)
{
    assert(mSize == otherVector.mSize);
    checkCudaErrors(cudaMemcpy ( d_mData,
                                otherVector.d_mData,
                                mSize*sizeof(double),
                                cudaMemcpyDeviceToDevice));
    return *this;

}



// Destructor
VectorGPU::~VectorGPU()
{
    checkCudaErrors(cudaFree(d_mData));
}

// Return Size method
int VectorGPU::GetSize() const
{
    return mSize;
}

// Binary vector addition
VectorGPU VectorGPU::operator+(const VectorGPU& v1) const
{
    VectorGPU result(mSize);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_add <<<GridSize, BlockSize>>> (mSize, d_mData, v1.d_mData, result.d_mData);

    return result;
}

// Binary vector substraction
VectorGPU VectorGPU::operator-(const VectorGPU& v1) const
{
    VectorGPU result(mSize);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_substract <<<GridSize, BlockSize>>> (mSize, d_mData, v1.d_mData, result.d_mData);

    return result;
}

// Scalar multiplication
VectorGPU VectorGPU::operator*(double a) const
{
    VectorGPU result(mSize);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_scalarmul <<<GridSize, BlockSize>>> (mSize, d_mData, result.d_mData, a);

    return result;
}

// p-norm method
double VectorGPU::Norm(int p) const
{
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    double result = 0.0;

    //double* d_result;
    //checkCudaErrors(cudaMalloc(&d_result, mSize*sizeof(double)));
    VectorGPU aux(mSize);
    Vector    aux_host(mSize);
    kernel_vector_power <<<GridSize, BlockSize>>> (mSize, d_mData, aux.d_mData, p);
    
    kernel_sum_reduce_onevector <<<GridSize, BlockSize>>> (mSize, aux.d_mData);

    //aux.copyToHost(aux_host);
    //std::cout << aux_host << std::endl;

    //If the grid size is odd, we get at an odd number of elements at the beginning of the
    // array that we need to sum, and the algorithm kernel_sum_reduce_onevector
    // only reduces an even number of elements
    if ( GridSize.x % 2 != 0)
    {
        // We change the element to 0 and increase the size of the grid by one
        // to get an even number of elements
        double zero = 0.0;
        checkCudaErrors(cudaMemcpy(&aux.d_mData[GridSize.x], &zero, sizeof(double), cudaMemcpyHostToDevice));
        BlockSize = GridSize.x + 1;
        GridSize = 1;
        kernel_sum_reduce_onevector <<<GridSize, BlockSize>>> (mSize, aux.d_mData);
    }
    BlockSize = GridSize;
    GridSize = 1;
    kernel_sum_reduce_onevector <<<GridSize, BlockSize>>> (mSize, aux.d_mData);

    //aux.copyToHost(aux_host);
    //std::cout << aux_host << std::endl;

    checkCudaErrors(cudaMemcpy( &result, aux.d_mData, sizeof(double), cudaMemcpyDeviceToHost));

    //cudaFree(d_aux);
    //cudaFree(d_result);

    // 1/p power
    result = pow(result, 1/(double)p);
    return result;
}

double VectorGPU::operator*(const VectorGPU& v1) const
{
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    double result = 0.0;

    VectorGPU aux(mSize);
    Vector    aux_host(mSize);
    kernel_vector_elementwise_product<<<GridSize, BlockSize>>> ( mSize, d_mData,
                                                  v1.d_mData, aux.d_mData);

    kernel_sum_reduce_onevector<<<GridSize, BlockSize>>> (mSize, aux.d_mData);
    
    //aux.copyToHost(aux_host);
    //std::cout << aux_host << std::endl;

    //If the grid size is odd, we get at an odd number of elements at the beginning of the
    // array that we need to sum, and the algorithm kernel_sum_reduce_onevector
    // only reduces an even number of elements
    if ( GridSize.x % 2 != 0)
    {
        // We change the element to 0 and increase the size of the grid by one
        // to get an even number of elements
        double zero = 0.0;
        checkCudaErrors(cudaMemcpy(&aux.d_mData[GridSize.x], &zero, sizeof(double), cudaMemcpyHostToDevice));
        BlockSize = GridSize.x + 1;
        GridSize = 1;
        kernel_sum_reduce_onevector <<<GridSize, BlockSize>>> (mSize, aux.d_mData);
    }
    BlockSize = GridSize;
    GridSize = 1;
    kernel_sum_reduce_onevector <<<GridSize, BlockSize>>> (mSize, aux.d_mData);

    checkCudaErrors(cudaMemcpy( &result, aux.d_mData, sizeof(double), cudaMemcpyDeviceToHost));

    return result;
}

void VectorGPU::copyToHost(Vector& hostVector)
{
    checkCudaErrors(cudaMemcpy( hostVector.mData, 
                d_mData,
                hostVector.mSize*sizeof(double), 
                cudaMemcpyDeviceToHost));
                                    
}

