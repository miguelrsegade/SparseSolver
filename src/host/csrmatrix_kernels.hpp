namespace
{

 void nonSymmetric_matVecMul(double* matData, int *matColInd, 
        int *matRowPtr, const Vector& v, Vector& res , int size)
{
    for ( int i = 0; i < size; i++)
    {
        for ( int k = matRowPtr[i]; k < matRowPtr[i+1]; k++)
        {
            res[i] = res[i] + matData[k]*v.Read(matColInd[k]);
        }
    }
}


}


namespace
{

// Not working
 void Symmetric_matVecMul(double* matData, int *matColInd, 
        int *matRowPtr, const Vector& v, Vector& res , int size)
{
    for ( int i = 0; i < size; i++)
    {
        
        for ( int k = matRowPtr[i]; k < matRowPtr[i+1]; k++)
        {
            res[i] = res[i] + matData[k]*v.Read(matColInd[k]);
        }
    }
}


}
