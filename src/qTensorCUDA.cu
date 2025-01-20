#include "qTensor.cuh"
#include "qTensorCUDA.cuh"
#include "bitsetCU.cuh"
#include "Contraction.hpp"

#include <cuComplex.h>
#include <unordered_map>
#include <cublas_v2.h>

// using namespace cuda_classes;
#ifdef USE_FLOAT
using dtype = float;
using cpx = cuFloatComplex;
#else
using dtype = double;
using cpx = cuDoubleComplex;
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

auto findCommonValues = [](std::vector<unsigned char> set1, std::vector<unsigned char> set2) -> std::vector<unsigned char> {
    std::vector<unsigned char> commonValues;
    for (auto value : set1) {
        if (std::find(set2.begin(), set2.end(), value) != set2.end()) {
            commonValues.push_back((unsigned char)value);
        }
    }
    return commonValues;
};

unsigned char getIndexInSet(unsigned char* set, unsigned char element, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (set[i] == element) {
            return i;
        }
    }
    return 255; // Element not found in the set
}

__device__ void keepNtoMbits(cuda_classes::bitset& bits, size_t n, size_t m) 
{ 
    for (size_t i = 0; i < n; i++) 
    { 
        bits.set(i, 0);
    }  
    for (size_t i = m; i < 64; i++) 
    { 
        bits.set(i, 0);
    }  
}

__device__ void print_bitset(cuda_classes::bitset& bits) {
    for (size_t i = 0; i < 64; i++) {
        printf("%d", bits.get(i));
    }
    printf("\n");
}

__global__ void contractionKernel(cuda_classes::bitset* bit_addressesA, cuda_classes::bitset* bit_addressesB, cpx* d_valuesA, cpx* d_valuesB, cpx* d_resultValues, size_t rankA, size_t rankB, size_t rankResult, size_t connectionsSize, unsigned char* indexesA_connections, unsigned char* indexesB_connections)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= (1 << (rankResult*2))) return;

    #ifdef USE_FLOAT
    cpx value = cuCmulf(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
    d_resultValues[i] = cuCaddf(d_resultValues[i], value);
    #else
    cpx value = cuCmul(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
    d_resultValues[i] = cuCadd(d_resultValues[i], value);
    #endif
    
    size_t old_gray = 0;
    for (size_t m = 1; m < (1 << connectionsSize); m++)
    {
        size_t gray_code = m ^ (m >> 1);

        unsigned int position_vacant =  __ffsll(gray_code ^ old_gray) - 1;

        unsigned char indexA = indexesA_connections[position_vacant];
        unsigned char indexB = indexesB_connections[position_vacant];

        bit_addressesA[i].xor_op(1 << (rankA + indexA));
        bit_addressesB[i].xor_op(1 << (indexB));

        #ifdef USE_FLOAT
        cpx value = cuCmulf(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
        d_resultValues[i] = cuCaddf(d_resultValues[i], value);
        #else
        cpx value = cuCmul(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
        d_resultValues[i] = cuCadd(d_resultValues[i], value);
        #endif

        old_gray = gray_code;
    }
}

__global__ void compute_bit_address_map(cuda_classes::bitset* bit_addressesA, cuda_classes::bitset* bit_addressesB, size_t rankA, size_t rankB, size_t rankResult,  unsigned char* indexesA, unsigned char* indexesB){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= (1 << (rankResult*2))) return;

    cuda_classes::bitset bits(i);


    for (size_t k = 0 ; k < rankResult; k++)
    {
        if (indexesB[k] != 255) bit_addressesB[i].set(rankB + indexesB[k], bits.get(rankResult + k));
        else                    bit_addressesA[i].set(rankA + indexesA[k], bits.get(rankResult + k));

        if (indexesA[k] != 255) bit_addressesA[i].set(indexesA[k], bits.get(k));
        else                    bit_addressesB[i].set(indexesB[k], bits.get(k));
    
    }
}

size_t round_div_up (size_t a, size_t b){
    return (a + b - 1)/b;
}

cublasHandle_t handle;

// python binding code for a single contraction
extern "C" {
    void single_contraction (cpx* A, cpx* B, cpx* C, size_t rankA, size_t rankB, size_t rankC, unsigned char* spanA, unsigned char* spanB, unsigned char* spanC, size_t spanA_size, size_t spanB_size, size_t spanC_size){
        // initialize cublas
        cublasStatus_t status;
        status = cublasCreate(&handle); 
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %s\n", _cudaGetErrorEnum(status));
            exit(EXIT_FAILURE);
        }


        std::vector<unsigned char> connections = findCommonValues(std::vector<unsigned char>(spanA, spanA + rankA), std::vector<unsigned char>(spanB, spanB + rankB));
        
        cudaError_t err;

        // start transfering data to the GPU
        cpx *d_resultValues;

        // use cudaMallocHost to allocate pinned memory
        unsigned char* indexesA, *indexesB, *indexes_connectionsA, *indexes_connectionsB;
        err = cudaMallocHost(&indexesA, spanC_size * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocHost(&indexesB, spanC_size * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
        
        err = cudaMallocHost(&indexes_connectionsA, connections.size() * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocHost(&indexes_connectionsB, connections.size() * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);

        cpx *d_valuesA, *d_valuesB;

        // memcopies
        {
            err = cudaMalloc(&d_valuesA, (1 << (spanA_size*2)) * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaMemcpy(d_valuesA, A, (1 << (spanA_size*2)) * sizeof(cpx), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaMalloc(&d_valuesB, (1 << (spanB_size*2)) * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaMemcpy(d_valuesB, B, (1 << (spanB_size*2)) * sizeof(cpx), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaMalloc(&d_resultValues,   (1 << (spanC_size*2)) * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaMemset(d_resultValues, 0, (1 << (spanC_size*2)) * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
        }

        // kernel call
        {
            size_t nels = 1 << (spanC_size*2);
            size_t blocksize = 256;
            size_t numBlocks = round_div_up(nels, blocksize);

            // std::cout << "numBlocks: " << numBlocks << " blocksize: " << blocksize << std::endl;

            // if the span are the same use gemm
            if (rankA == rankB && std::equal(spanA, spanA + rankA, spanB)) {
                cublasSetStream(handle, 0);
                size_t nels = 1 << (spanC_size);
                cpx alpha = {1.0, 0.0};
                cpx beta = {0.0, 0.0};
                #ifdef USE_FLOAT
                // cublasStatus_t status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, gpuQtensorMap[root->left].values, nels, gpuQtensorMap[root->right].values, nels, &beta, gpuQtensorMap[root].values, nels);
                cublasStatus_t status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, d_valuesA, nels, d_valuesB, nels, &beta, d_resultValues, nels);
                #else
                // cublasStatus_t status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, gpuQtensorMap[root->left].values, nels, gpuQtensorMap[root->right].values, nels, &beta, gpuQtensorMap[root].values, nels);
                cublasStatus_t status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, d_valuesA, nels, d_valuesB, nels, &beta, d_resultValues, nels);
                #endif
                if (status != CUBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "cublasCgemm failed: %s\n", _cudaGetErrorEnum(status));
                    exit(EXIT_FAILURE);
                }
            }
            else {
                cuda_classes::bitset* bit_addressesA, *bit_addressesB;

                err = cudaMalloc(&bit_addressesA, nels * sizeof(cuda_classes::bitset)); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMalloc(&bit_addressesB, nels * sizeof(cuda_classes::bitset)); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaMemset(bit_addressesA, 0, nels * sizeof(cuda_classes::bitset)); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMemset(bit_addressesB, 0, nels * sizeof(cuda_classes::bitset)); cuda_err_check(err, __FILE__, __LINE__);

                #pragma omp parallel for
                for (size_t i = 0; i < spanC_size; i++) {
                    indexesA[i] = getIndexInSet(spanA, spanC[i], spanA_size);
                    indexesB[i] = getIndexInSet(spanB, spanC[i], spanB_size);
                }

                #pragma omp parallel for
                for (size_t i = 0; i < connections.size(); i++) {
                    indexes_connectionsA[i] = getIndexInSet(spanA, connections[i], spanA_size);
                    indexes_connectionsB[i] = getIndexInSet(spanB,  connections[i], spanB_size);
                }

                unsigned char* d_indexesA, *d_indexesB;
                err = cudaMalloc(&d_indexesA, spanC_size * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMalloc(&d_indexesB, spanC_size * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaMemcpy(d_indexesA, indexesA, spanC_size * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMemcpy(d_indexesB, indexesB, spanC_size * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

                unsigned char* d_indexes_connectionsA, *d_indexes_connectionsB;
                err = cudaMalloc(&d_indexes_connectionsA, connections.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMalloc(&d_indexes_connectionsB, connections.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaMemcpy(d_indexes_connectionsA, indexes_connectionsA, connections.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMemcpy(d_indexes_connectionsB, indexes_connectionsB, connections.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

                double gb_used = (double)(sizeof(cuda_classes::bitset) * nels * 2) / (1024 * 1024 * 1024);

                if (gb_used > 1)
                    std::cout << "Memory allocation: " << (double)(sizeof(cuda_classes::bitset) * nels * 2) / (1024 * 1024 * 1024) << " GB" << std::endl;

                // compute_bit_address_map<<<numBlocks, blocksize, 0>>>(bit_addressesA, bit_addressesB, root->left->span.size(), root->right->span.size(), spanC_size, d_indexesA, d_indexesB);
                compute_bit_address_map<<<numBlocks, blocksize>>>(bit_addressesA, bit_addressesB, rankA, rankB, rankC, d_indexesA, d_indexesB);

                // contractionKernel<<<numBlocks, blocksize, 0>>>(bit_addressesA, bit_addressesB, gpuQtensorMap[root->left].values, gpuQtensorMap[root->right].values, gpuQtensorMap[root].values, root->left->span.size(), root->right->span.size(), spanC_size, connections.size(), d_indexes_connectionsA, d_indexes_connectionsB);
                contractionKernel<<<numBlocks, blocksize>>>(bit_addressesA, bit_addressesB, d_valuesA, d_valuesB, d_resultValues, rankA, rankB, rankC, connections.size(), d_indexes_connectionsA, d_indexes_connectionsB);

                err = cudaFree(d_indexesA); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaFree(d_indexesB); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaFree(d_indexes_connectionsA); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaFree(d_indexes_connectionsB); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaFree(bit_addressesA); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaFree(bit_addressesB); cuda_err_check(err, __FILE__, __LINE__);
            }

            err = cudaMemcpy(C, d_resultValues, (1 << (spanC_size*2)) * sizeof(cpx), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
            
            err = cudaFree(d_valuesA); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaFree(d_valuesB); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaFree(d_resultValues); cuda_err_check(err, __FILE__, __LINE__);
        }
    }
}