#include <sycl/sycl.hpp>
#include <complex>
#include <vector>
#include <algorithm>
#include <iostream>
#include <bitsetCU.hpp>

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
        sycl::buffer<cpx> buf_A(A, sycl::range<1>(1 << (spanA_size * 2)));
        sycl::buffer<cpx> buf_B(B, sycl::range<1>(1 << (spanB_size * 2)));
        sycl::buffer<cpx> buf_result(C, sycl::range<1>(result_size));

        if (rankA == rankB && std::equal(spanA, spanA + rankA, spanB)) {
            mkl_matrix_multiply(buf_A, buf_B, buf_result, spanC_size);
        } else {
            general_contraction(buf_A, buf_B, buf_result,
                              rankA, rankB, rankC,
                              spanA, spanB, spanC,
                              spanA_size, spanB_size, spanC_size,
                              connections);
        }
    }

private:
    sycl::queue queue_;

    void mkl_matrix_multiply(sycl::buffer<cpx>& A, sycl::buffer<cpx>& B,
                            sycl::buffer<cpx>& C, size_t size) {

        // Calculate matrix dimension
        const int n = 1 << size;  // Matrix dimension is 2^size

        try {
            queue_.submit([&](sycl::handler& h) {
                auto a = A.get_access<sycl::access::mode::read>(h);
                auto b = B.get_access<sycl::access::mode::read>(h);
                auto c = C.get_access<sycl::access::mode::write>(h);

                const int TILE_SIZE = 32;  // Optimal for V100's architecture
                
                h.parallel_for(
                    sycl::range<2>(n, n),
                    [=](sycl::id<2> idx) {
                        const int row = idx[0];
                        const int col = idx[1];
                        
                        cpx sum(0.0, 0.0);
                        for (int k = 0; k < n; k += TILE_SIZE) {
                            for (int t = 0; t < TILE_SIZE && k + t < n; t++) {
                                sum += a[row * n + (k + t)] * b[(k + t) * n + col];
                            }
                        }
                        c[row * n + col] = sum;
                    }
                );
            }).wait();
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            throw;
        }
        catch (std::exception const& e) {
            std::cerr << "Standard exception caught: " << e.what() << std::endl;
            throw;
        }
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