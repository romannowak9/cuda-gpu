#include "histogram.h"

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{   
    // TODO: Inaczej pętle napisać 
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        const int binWidth = (N_LETTERS + nBins - 1) / nBins;

        for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
            if (i >= size)
                break;
            unsigned char letter = buffer[i];
            int alphabetPosition = letter - 'a';
            if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
            {
                atomicAdd(histogram + (alphabetPosition / binWidth), 1);
            }
        }
    }
}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        const int binWidth = (N_LETTERS + nBins - 1) / nBins;

        for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
            if (i >= size)
                break;
            unsigned char letter = buffer[i];
            int alphabetPosition = letter - 'a';
            if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
            {
                atomicAdd(histogram + (alphabetPosition / binWidth), 1);
            }
        }
    }
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}


std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{
    unsigned char *d_buffer;
    std::vector<unsigned int> hist(nBins);
    unsigned int *d_hist;

    const size_t buffer_size = data.size() * sizeof(unsigned char);
    const size_t hist_size = nBins * sizeof(unsigned int);
    
    // allocate input and output in the device
    cudaMalloc((void **)&d_buffer, buffer_size);
    cudaMalloc((void **)&d_hist, hist_size);
    
    // copy to the device
    cudaMemcpy(d_buffer, data.data(), buffer_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, hist_size);
    
    int blockSize = 64;  // Maksimum block size on device
    int gridSize = (data.size() + blockSize - 1) / blockSize;
    
    if (method == HistMethod::Block) {
        histogram_1<<<gridSize, blockSize>>>(d_buffer, data.size(), d_hist, nBins);
    } else if (method == HistMethod::Interleaved) {
        histogram_2<<<gridSize, blockSize>>>(d_buffer, data.size(), d_hist, nBins);
    } else if (method == HistMethod::Privatised) {
        histogram_3<<<gridSize, blockSize>>>(d_buffer, data.size(), d_hist, nBins);
    }
    
    cudaMemcpy(hist.data(), d_hist, hist_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_hist);
    cudaFree(d_buffer);
    
    return hist;
}

std::vector<unsigned int> computeHistogramOnHost(const std::vector<unsigned char> &data, int nBins)
{
    std::vector<unsigned int> histogram(nBins, 0);
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division

    for (const auto &ch : data)
    {
        int alphabetPosition = ch - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
        {
            histogram[alphabetPosition / binWidth]++;
        }
    }

    return histogram;
}
