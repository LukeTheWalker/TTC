#include <sycl/sycl.hpp>
#include <complex>
#include <vector>
#include <algorithm>
#include <iostream>
#include <bitsetCU.hpp>
// #include <oneapi/mkl.hpp>

#ifdef USE_FLOAT
using dtype = float;
using cpx = std::complex<float>;
#else
using dtype = double;
using cpx = std::complex<double>;
#endif

// Previous helper functions remain the same
SYCL_EXTERNAL inline int sycl_ffsll(long long x) {
    if (x == 0) return 0;
    
    #if defined(__GNUC__) || defined(__clang__)
        return __builtin_ffsll(x);
    #else
        unsigned long long ux = static_cast<unsigned long long>(x);
        int pos = 1;
        
        while ((ux & 0xFF) == 0) {
            ux >>= 8;
            pos += 8;
        }
        
        while ((ux & 1) == 0) {
            ux >>= 1;
            pos++;
        }
        
        return pos;
    #endif
}

std::vector<unsigned char> findCommonValues(const std::vector<unsigned char>& set1, 
                                          const std::vector<unsigned char>& set2) {
    std::vector<unsigned char> commonValues;
    for (auto value : set1) {
        if (std::find(set2.begin(), set2.end(), value) != set2.end()) {
            commonValues.push_back(value);
        }
    }
    return commonValues;
}

unsigned char getIndexInSet(const unsigned char* set, unsigned char element, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (set[i] == element) {
            return i;
        }
    }
    return 255;
}

class TensorContractor {
public:
    TensorContractor() : queue_(sycl::gpu_selector_v) {
        auto device = queue_.get_device();
        if (!device.is_gpu()) {
            throw std::runtime_error("No GPU device found");
        }
    }

void single_contraction(cpx* A, cpx* B, cpx* C,
                          size_t rankA, size_t rankB, size_t rankC,
                          unsigned char* spanA, unsigned char* spanB, unsigned char* spanC,
                          size_t spanA_size, size_t spanB_size, size_t spanC_size) {
        
        std::vector<unsigned char> connections = 
            findCommonValues(std::vector<unsigned char>(spanA, spanA + rankA),
                           std::vector<unsigned char>(spanB, spanB + rankB));

        const size_t result_size = 1 << (spanC_size * 2);
        sycl::buffer<cpx> buf_A((1 << (spanA_size * 2)));
        sycl::buffer<cpx> buf_B((1 << (spanB_size * 2)));
        sycl::buffer<cpx> buf_result(result_size);

        // Copy input data
        {
            auto acc_A = buf_A.get_access<sycl::access::mode::write>();
            auto acc_B = buf_B.get_access<sycl::access::mode::write>();
            std::copy(A, A + (1 << (spanA_size * 2)), acc_A.begin());
            std::copy(B, B + (1 << (spanB_size * 2)), acc_B.begin());
        }

        if (rankA == rankB && std::equal(spanA, spanA + rankA, spanB)) {
            mkl_matrix_multiply(buf_A, buf_B, buf_result, spanC_size);
        } else {
            general_contraction(buf_A, buf_B, buf_result,
                              rankA, rankB, rankC,
                              spanA, spanB, spanC,
                              spanA_size, spanB_size, spanC_size,
                              connections);
        }

        // Copy result back
        {
            auto acc_result = buf_result.get_access<sycl::access::mode::read>();
            std::copy(acc_result.begin(), acc_result.end(), C);
        }
    }

private:
    sycl::queue queue_;

    void mkl_matrix_multiply(sycl::buffer<cpx>& A, sycl::buffer<cpx>& B,
                           sycl::buffer<cpx>& C, size_t size) {
        const size_t WORK_GROUP_SIZE = 256;
        const size_t result_size = 1 << (size * 2);

        queue_.submit([&](sycl::handler& h) {
            auto acc_A = A.get_access<sycl::access::mode::read>(h);
            auto acc_B = B.get_access<sycl::access::mode::read>(h);
            auto acc_C = C.get_access<sycl::access::mode::write>(h);

            sycl::range<1> global{((result_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE};
            sycl::range<1> local{WORK_GROUP_SIZE};

            h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i < result_size) {
                    size_t row = i >> size;
                    size_t col = i & ((1 << size) - 1);
                    cpx sum = 0;
                    for (size_t k = 0; k < (1 << size); k++) {
                        sum += acc_A[row * (1 << size) + k] * acc_B[k * (1 << size) + col];
                    }
                    acc_C[i] = sum;
                }
            });
        });
    }

    // Previous general_contraction implementation remains the same
    void general_contraction(sycl::buffer<cpx>& A, sycl::buffer<cpx>& B,
                           sycl::buffer<cpx>& result,
                           size_t rankA, size_t rankB, size_t rankC,
                           unsigned char* spanA, unsigned char* spanB,
                           unsigned char* spanC,
                           size_t spanA_size, size_t spanB_size,
                           size_t spanC_size,
                           const std::vector<unsigned char>& connections) {
        
        std::vector<unsigned char> indexesA(spanC_size);
        std::vector<unsigned char> indexesB(spanC_size);
        std::vector<unsigned char> indexes_connectionsA(connections.size());
        std::vector<unsigned char> indexes_connectionsB(connections.size());

        #pragma omp parallel for
        for (size_t i = 0; i < spanC_size; i++) {
            indexesA[i] = getIndexInSet(spanA, spanC[i], spanA_size);
            indexesB[i] = getIndexInSet(spanB, spanC[i], spanB_size);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < connections.size(); i++) {
            indexes_connectionsA[i] = getIndexInSet(spanA, connections[i], spanA_size);
            indexes_connectionsB[i] = getIndexInSet(spanB, connections[i], spanB_size);
        }

        sycl::buffer<unsigned char> buf_indexesA(indexesA);
        sycl::buffer<unsigned char> buf_indexesB(indexesB);
        sycl::buffer<unsigned char> buf_connectionsA(indexes_connectionsA);
        sycl::buffer<unsigned char> buf_connectionsB(indexes_connectionsB);

        const size_t num_connections = connections.size();
        const size_t result_size = 1 << (spanC_size * 2);
        const size_t WORK_GROUP_SIZE = 256;

        queue_.submit([&](sycl::handler& h) {
            auto acc_A = A.get_access<sycl::access::mode::read>(h);
            auto acc_B = B.get_access<sycl::access::mode::read>(h);
            auto acc_result = result.get_access<sycl::access::mode::read_write>(h);
            auto acc_indexesA = buf_indexesA.get_access<sycl::access::mode::read>(h);
            auto acc_indexesB = buf_indexesB.get_access<sycl::access::mode::read>(h);
            auto acc_connectionsA = buf_connectionsA.get_access<sycl::access::mode::read>(h);
            auto acc_connectionsB = buf_connectionsB.get_access<sycl::access::mode::read>(h);

            sycl::range<1> global{((result_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE};
            sycl::range<1> local{WORK_GROUP_SIZE};

            h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i < result_size) {
                    sycl_classes::bitset bitsA, bitsB;

                    for (size_t k = 0; k < rankC; k++) {
                        bool bit_high = ((i >> (rankC + k)) & 1) != 0;
                        bool bit_low = ((i >> k) & 1) != 0;

                        if (acc_indexesB[k] != 255) {
                            bitsB.set(rankB + acc_indexesB[k], bit_high);
                        } else {
                            bitsA.set(rankA + acc_indexesA[k], bit_high);
                        }

                        if (acc_indexesA[k] != 255) {
                            bitsA.set(acc_indexesA[k], bit_low);
                        } else {
                            bitsB.set(acc_indexesB[k], bit_low);
                        }
                    }

                    cpx sum = 0;
                    sum += acc_A[bitsA.to_ulong()] * acc_B[bitsB.to_ulong()];
                    size_t old_gray = 0;
                    
                    for (size_t m = 1; m < (1 << num_connections); m++) {
                        size_t gray_code = m ^ (m >> 1);
                        unsigned int position_vacant = sycl_ffsll(gray_code ^ old_gray) - 1;
                        
                        unsigned char indexA = acc_connectionsA[position_vacant];
                        unsigned char indexB = acc_connectionsB[position_vacant];
                        
                        bitsA.xor_op(1ULL << (rankA + indexA));
                        bitsB.xor_op(1ULL << indexB);
                        
                        sum += acc_A[bitsA.to_ulong()] * acc_B[bitsB.to_ulong()];
                        
                        old_gray = gray_code;
                    }
                    
                    acc_result[i] = sum;
                }
            });
        });
    }
};

extern "C" {
    void single_contraction(cpx* A, cpx* B, cpx* C,
                          size_t rankA, size_t rankB, size_t rankC,
                          unsigned char* spanA, unsigned char* spanB, unsigned char* spanC,
                          size_t spanA_size, size_t spanB_size, size_t spanC_size) {
        static TensorContractor contractor;
        contractor.single_contraction(A, B, C, rankA, rankB, rankC,
                                    spanA, spanB, spanC,
                                    spanA_size, spanB_size, spanC_size);
    }
}