#include <algorithm>
#include <bitsetCU.hpp>
#include <complex>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#ifdef USE_FLOAT
using dtype = float;
using cpx = std::complex<float>;
#else
using dtype = double;
using cpx = std::complex<double>;
#endif

struct gate
{
    unsigned char *qubits;
    cpx *unitary;
    size_t rank;
};

// Previous helper functions remain the same
SYCL_EXTERNAL inline int sycl_ffsll(long long x)
{
    if (x == 0)
        return 0;

#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ffsll(x);
#else
    unsigned long long ux = static_cast<unsigned long long>(x);
    int pos = 1;

    while ((ux & 0xFF) == 0)
    {
        ux >>= 8;
        pos += 8;
    }

    while ((ux & 1) == 0)
    {
        ux >>= 1;
        pos++;
    }

    return pos;
#endif
}

std::vector<unsigned char> findCommonValues(const std::vector<unsigned char> &set1,
                                            const std::vector<unsigned char> &set2)
{
    std::vector<unsigned char> commonValues;
    for (auto value : set1)
    {
        if (std::find(set2.begin(), set2.end(), value) != set2.end())
        {
            commonValues.push_back(value);
        }
    }
    return commonValues;
}

unsigned char getIndexInSet(const unsigned char *set, unsigned char element, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        if (set[i] == element)
        {
            return i;
        }
    }
    return 255;
}

class TensorContractor
{
public:
    TensorContractor() : queue_(sycl::gpu_selector_v)
    {
        auto device = queue_.get_device();
        if (!device.is_gpu())
        {
            throw std::runtime_error("No GPU device found");
        }
    }

    void optimal_contraction(sycl::buffer<cpx> &buf_A,
                             sycl::buffer<cpx> &buf_B,
                             sycl::buffer<cpx> &buf_result,
                             const size_t rankA,
                             const size_t rankB,
                             const size_t rankC,
                             const unsigned char *spanA,
                             const unsigned char *spanB,
                             const unsigned char *spanC,
                             const std::vector<unsigned char> &connections)
    {
        if (rankA == rankB && std::equal(spanA, spanA + rankA, spanB))
        {
            mkl_matrix_multiply(buf_A, buf_B, buf_result, rankC);
        }
        else
        {
            general_contraction(buf_A, buf_B, buf_result,
                                rankA, rankB, rankC,
                                spanA, spanB, spanC,
                                connections);
        }
    }

    void single_contraction(const cpx *A, const cpx *B, cpx *C,
                            const size_t rankA, const size_t rankB, size_t rankC,
                            const unsigned char *spanA, const unsigned char *spanB, const unsigned char *spanC)
    {

        std::vector<unsigned char> connections =
            findCommonValues(std::vector<unsigned char>(spanA, spanA + rankA),
                             std::vector<unsigned char>(spanB, spanB + rankB));

        const size_t result_size = 1 << (rankC * 2);
        sycl::buffer<cpx> buf_A(A, sycl::range<1>(1 << (rankA * 2)));
        sycl::buffer<cpx> buf_B(B, sycl::range<1>(1 << (rankB * 2)));
        sycl::buffer<cpx> buf_result(C, sycl::range<1>(result_size));

        optimal_contraction(buf_A, buf_B, buf_result,
                            rankA, rankB, rankC,
                            spanA, spanB, spanC,
                            connections);
    }

private:
    sycl::queue queue_;

    void mkl_matrix_multiply(sycl::buffer<cpx> &A, sycl::buffer<cpx> &B,
                             sycl::buffer<cpx> &C, size_t size)
    {

        // Calculate matrix dimension
        const int n = 1 << size; // Matrix dimension is 2^size

        try
        {
            queue_.submit([&](sycl::handler &h)
                          {
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
                ); })
                .wait_and_throw();
        }
        catch (sycl::exception const &e)
        {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            throw;
        }
        catch (std::exception const &e)
        {
            std::cerr << "Standard exception caught: " << e.what() << std::endl;
            throw;
        }
    }
    // Previous general_contraction implementation remains the same
    void general_contraction(sycl::buffer<cpx> &A, sycl::buffer<cpx> &B,
                             sycl::buffer<cpx> &result,
                             size_t rankA, size_t rankB, size_t rankC,
                             const unsigned char *spanA, const unsigned char *spanB,
                             const unsigned char *spanC,
                             const std::vector<unsigned char> &connections)
    {

        std::vector<unsigned char> indexesA(rankC);
        std::vector<unsigned char> indexesB(rankC);
        std::vector<unsigned char> indexes_connectionsA(connections.size());
        std::vector<unsigned char> indexes_connectionsB(connections.size());

#pragma omp parallel for
        for (size_t i = 0; i < rankC; i++)
        {
            indexesA[i] = getIndexInSet(spanA, spanC[i], rankA);
            indexesB[i] = getIndexInSet(spanB, spanC[i], rankB);
        }

#pragma omp parallel for
        for (size_t i = 0; i < connections.size(); i++)
        {
            indexes_connectionsA[i] = getIndexInSet(spanA, connections[i], rankA);
            indexes_connectionsB[i] = getIndexInSet(spanB, connections[i], rankB);
        }

        sycl::buffer<unsigned char> buf_indexesA(indexesA);
        sycl::buffer<unsigned char> buf_indexesB(indexesB);
        sycl::buffer<unsigned char> buf_connectionsA(indexes_connectionsA);
        sycl::buffer<unsigned char> buf_connectionsB(indexes_connectionsB);

        const size_t num_connections = connections.size();
        const size_t result_size = 1 << (rankC * 2);
        const size_t WORK_GROUP_SIZE = 256;

        queue_.submit([&](sycl::handler &h)
                      {
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
            }); });
    }
};

struct sycl_gate_wrapper
{
    sycl::buffer<cpx> unitary;
    std::vector<unsigned char> span;
};

extern "C"
{
    void single_contraction(const cpx *A, const cpx *B, cpx *C,
                            const size_t rankA, const size_t rankB, const size_t rankC,
                            const unsigned char *spanA, const unsigned char *spanB, const unsigned char *spanC)
    {
        static TensorContractor contractor;
        contractor.single_contraction(A, B, C, rankA, rankB, rankC,
                                      spanA, spanB, spanC);
    }

    void contract_circuit(gate *gates, size_t num_gates, cpx *result_gate, size_t num_qubits)
    {
        static TensorContractor contractor;

        // make a vector holding the pointer to the gates
        std::vector<std::unique_ptr<sycl_gate_wrapper>> gate_vector;
        for (size_t i = 0; i < num_gates; i++)
        {
            auto wrapper = std::make_unique<sycl_gate_wrapper>(sycl_gate_wrapper{
                sycl::buffer<cpx>(gates[i].unitary, sycl::range<1>(1 << (2 * gates[i].rank))),
                std::vector<unsigned char>(gates[i].qubits, gates[i].qubits + gates[i].rank)
            });
            gate_vector.push_back(std::move(wrapper));
        }

        // go through the list of gate, when two consecutive gates perfectly match, contract them, put the result in place of the first gate and remove the second gate
        while (gate_vector.size() > 1)
        {
            for (size_t i = 0; i < gate_vector.size() - 1; i++)
            {
                // get the connections between the two gates 
                std::vector<unsigned char> connections = findCommonValues(
                    gate_vector[i]->span,
                    gate_vector[i + 1]->span
                );

                // the qubits of the result are the union of the qubits of the two gates
                std::vector<unsigned char> result_qubits;
                for (size_t j = 0; j < gate_vector[i]->span.size(); j++) result_qubits.push_back(gate_vector[i]->span[j]);
                for (size_t j = 0; j < gate_vector[i + 1]->span.size(); j++)
                {
                    if (std::find(result_qubits.begin(), result_qubits.end(), gate_vector[i + 1]->span[j]) == result_qubits.end())
                    {
                    result_qubits.push_back(gate_vector[i + 1]->span[j]);
                    }
                }

                // create a new gate with the result qubits
                auto result_gate = std::make_unique<sycl_gate_wrapper>(sycl_gate_wrapper{
                    sycl::buffer<cpx>(sycl::range<1>(1 << (2 * result_qubits.size()))),
                    result_qubits
                });

                // contract the two gates
                contractor.optimal_contraction(gate_vector[i]->unitary, gate_vector[i + 1]->unitary, result_gate->unitary,
                                gate_vector[i]->span.size(), gate_vector[i + 1]->span.size(), result_qubits.size(),
                                gate_vector[i]->span.data(), gate_vector[i + 1]->span.data(), result_qubits.data(),
                                connections);

                // put the result in place of the first gate
                gate_vector[i] = std::move(result_gate);

                // remove the second gate
                gate_vector.erase(gate_vector.begin() + i + 1);

                break;
            }
        }

        // copy the result to the output gate
        auto acc = gate_vector[0]->unitary.get_host_access(sycl::read_only);
        std::copy(acc.get_pointer(), acc.get_pointer() + (1 << (2 * num_qubits)), result_gate);

    }
}