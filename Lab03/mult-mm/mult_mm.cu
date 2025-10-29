#include "mult_mm.h"


__global__ void matrixMulKernel(const float *A, const float *B, float *C,
                                int A_rows, int A_cols, int B_cols)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (( Row < A_rows ) && ( Col < B_cols )) {
        float Pvalue = 0;
        // Each thread computes one element of the block sub - matrix
        for ( int k = 0; k < A_cols ; ++ k ) {
            Pvalue += A[Row * A_cols + k] * B[k * B_cols + Col];
        }
        C[Row * B_cols + Col] = Pvalue;
    }
}

__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C,
                                     int A_rows, int A_cols, int B_cols)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    float dot_acc = 0.0f;
    for (int ph = 0; ph < (A_cols + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        if (row < A_rows && ph * TILE_SIZE + tile_col < A_cols)
            tileA[tile_row][tile_col] = A[row * A_cols + ph * TILE_SIZE + tile_col];
        else
            tileA[tile_row][tile_col] = 0.0f;

        if (ph * TILE_SIZE + tile_row < A_cols && col < B_cols)
            tileB[tile_row][tile_col] = B[(ph * TILE_SIZE + tile_row) * B_cols + col];
        else
            tileB[tile_row][tile_col] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE ; ++k) {
            dot_acc += tileA[tile_row][k] * tileB[k][tile_col];
        }

        __syncthreads();
    }

    if ((row < A_rows) && (col < B_cols))
        C[row * B_cols + col] = dot_acc;

}

__global__ void matrixMulGranularKernel(const float *A, const float *B, float *C,
                                        int A_rows, int A_cols, int B_cols)
{
}

Matrix multMatrixMatrixOnDevice(const Matrix &A, const Matrix &B, MultMethod method)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Matrixes dimensions do not match for multiplication.");
    }

    Matrix C(A.getRows(), B.getCols());

    // allocate input and output in the device
    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, A.getRows() * A.getCols() * sizeof(float));
    cudaMalloc((void **)&d_B, B.getRows() * B.getCols() * sizeof(float));
    cudaMalloc((void **)&d_C, C.getRows() * C.getCols() * sizeof(float));

    // copy to the device
    cudaMemcpy(d_A, A.getDataConstPtr(), A.getRows() * A.getCols() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.getDataConstPtr(), B.getRows() * B.getCols() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);  // Maksimum block size on device
    dim3 gridSize((C.getCols() + TILE_SIZE - 1) / TILE_SIZE, (C.getRows() + TILE_SIZE - 1) / TILE_SIZE);

    if (method == MultMethod::Standard) {
        matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, A.getRows(), A.getCols(), B.getCols());
    } else if (method == MultMethod::Tiled) {
        matrixMulTiledKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, A.getRows(), A.getCols(), B.getCols());
    }

    cudaMemcpy(C.getDataPtr(), d_C, C.getRows() * C.getCols() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

Matrix multMatrixMatrixOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < B.getCols(); ++j)
        {
            for (unsigned int k = 0; k < A.getCols(); ++k)
            {
                C.getDataPtr()[i * C.getCols() + j] +=
                    A.getDataConstPtr()[i * A.getCols() + k] *
                    B.getDataConstPtr()[k * B.getCols() + j];
            }
        }
    }
    return C;
}
