#include "mult_mv.h"

__global__ void multMatrixVector(float *b, float *A, float *x, unsigned int nrows, unsigned int ncols)
{
}

Matrix multMatrixVectorOnDevice(const Matrix &A, const Matrix &x)
{
    Matrix outMatrix(A.getRows(), x.getCols());

    // allocate input and output in the device
    float *d_A;
    float *d_x;
    float *d_outMatrix;

    cudaMalloc((void **)&d_A, A.getRows() * A.getCols() * sizeof(float));
    cudaMalloc((void **)&d_x, x.getRows() * x.getCols() * sizeof(float));
    cudaMalloc((void **)&d_outMatrix, outMatrix.getRows() * outMatrix.getCols() * sizeof(float));

    // copy to the device
    cudaMemcpy(d_A, A.getDataConstPtr(), A.getRows() * A.getCols() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.getDataConstPtr(), x.getRows() * x.getCols() * sizeof(float), cudaMemcpyHostToDevice);

    float blockSize = 1024.0f;
    int gridSize = ceil(A.getRows() / blockSize)

    blurKernel<<<dimGrid, dimBlock>>>(d_outputImage, d_inputImage, outputImage.getCols(), outputImage.getRows(), blurSize);

    cudaMemcpy(outputImage.getDataPtr(), d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return outputImage;
}

Matrix multMatrixVectorOnHost(const Matrix &A, const Matrix &x)
{
    if (A.getCols() != x.getRows())
    {
        throw std::runtime_error("Matrix and vector dimensions do not match for multiplication.");
    }

    Matrix b(A.getRows(), 1);
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        float sum = 0.0f;
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            sum += A.getDataConstPtr()[i * A.getCols() + j] * x.getDataConstPtr()[j];
        }
        b.getDataPtr()[i] = sum;
    }
    return b;
}
