#pragma once


__global__ void kernel_fill_vector(const int n, double* v, double value)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < n)
        v[ind] = value;
}

__global__ void kernel_vector_add(const int n, const double* v1, const double* v2, double* out)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        out[ind] = v1[ind] + v2[ind];
}

__global__ void kernel_vector_substract(const int n, const double* v1, const double* v2, double* out)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        out[ind] = v1[ind] - v2[ind];
}

__global__ void kernel_vector_scalarmul(const int n, const double* v, double* result,  double a)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        result[ind] = v[ind]*a;
}

__global__ void kernel_vector_power(const int n, const double* v, double* aux, double p)
{

    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind <n)
    {
        double tmp = v[ind];
        for (int i=1; i<p; i++)
        {
           tmp *= v[ind]; 
        }
        aux[ind] = tmp;
    }
        
}

__global__ void kernel_vector_elementwise_product(const int n, const double* v1,
                                             const double* v2, double* res)
{
    
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
    {
        res[ind] = v1[ind]*v2[ind];
    }

}

__global__ void kernel_sum_reduce_onevector(const int n,  double* in)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    int tind = threadIdx.x;

        for (unsigned int s = blockDim.x / 2; s > 0; s>>=1 )
        {
            // If ind+s is greater than n It takes garbage
            if (tind < s && (ind+s) < n)
            {
                //in[ind] = 1.0;
                in[ind] += in[ind + s];
            }
        }

        if (tind == 0)
        {
            in[blockIdx.x] = in[ind];
        }
}

