#include <algorithm>
#include <bitsetCU.hpp>
#include <chrono>
#include <complex>
#include <iostream>
#include <queue>
#include <sycl/sycl.hpp>
#include <vector>
#include <type_traits>
#include <oneapi/math.hpp>


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

struct contraction_t
{
    size_t g1; ///< Index of the first gate.
    size_t g2; ///< Index of the second gate.
    size_t g1i;
    size_t g2i;
    std::vector<unsigned char> connections; ///< Common qubits shared between the two gates.
};

struct sycl_gate_wrapper
{
    cpx* unitary;
    std::vector<unsigned char> span;
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
    sycl::property_list props{sycl::ext::oneapi::property::queue::discard_events{}};

    TensorContractor() : queue_(sycl::gpu_selector_v)
    {
        auto device = queue_.get_device();
        if (!device.is_gpu())
        {
            throw std::runtime_error("No GPU device found");
        }
    }

    void optimal_contraction(cpx * A,
                             cpx * B,
                             cpx * result,
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
            onemath_matrix_multiply(A, B, result, rankC);
        }
        else
        {
            general_contraction(A, B, result,
                                rankA, rankB, rankC,
                                spanA, spanB, spanC,
                                connections);
        }
    }

    void single_contraction(cpx *A, cpx *B, cpx *C,
                            const size_t rankA, const size_t rankB, size_t rankC,
                            const unsigned char *spanA, const unsigned char *spanB, const unsigned char *spanC)
    {

        std::vector<unsigned char> connections =
            findCommonValues(std::vector<unsigned char>(spanA, spanA + rankA),
                             std::vector<unsigned char>(spanB, spanB + rankB));

        const size_t result_size = 1 << (rankC * 2);
        optimal_contraction(A,B, C,
                    rankA, rankB, rankC,
                    spanA, spanB, spanC,
                    connections);
    }

    sycl::queue queue_;

    void onemath_matrix_multiply(cpx *A, cpx *B, cpx *C, size_t rankC)
    {
        // Create SYCL buffers and perform multiplication
        size_t result_size = (1 << (rankC * 2));
        size_t leading = 1 << rankC;
        sycl::buffer<cpx> buf_A(A, sycl::range<1>(result_size));
        sycl::buffer<cpx> buf_B(B, sycl::range<1>(result_size));
        sycl::buffer<cpx> buf_C(C, sycl::range<1>(result_size));

        oneapi::math::blas::column_major::gemm(queue_, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, 
            leading, leading, leading, cpx(1.0), buf_A, leading, buf_B, leading, cpx(0.0), buf_C, leading);
    }

    void general_contraction(cpx * A, cpx * B, cpx * result,
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
        const size_t WORK_GROUP_SIZE = 32;

        queue_.submit([&](sycl::handler &h)
            {
                auto acc_indexesA = buf_indexesA.get_access<sycl::access::mode::read>(h);
                auto acc_indexesB = buf_indexesB.get_access<sycl::access::mode::read>(h);
                auto acc_connectionsA = buf_connectionsA.get_access<sycl::access::mode::read>(h);
                auto acc_connectionsB = buf_connectionsB.get_access<sycl::access::mode::read>(h);

                sycl::range<1> global{((result_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE};
                sycl::range<1> local{WORK_GROUP_SIZE};

                h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
                auto num_groups = item.get_group_range(0);

                const size_t i = item.get_global_id(0);
                if (i < result_size) {
                    sycl_classes::bitset bitsA, bitsB;

                    for (size_t k = 0; k < rankC; k++) {
                        bool bit_high = ((i >> (rankC + k)) & 1) != 0;
                        bool bit_low = ((i >> k) & 1) != 0;

                        if (acc_indexesB[k] != 255) bitsB.set(rankB + acc_indexesB[k], bit_high);
                        else                        bitsA.set(rankA + acc_indexesA[k], bit_high);

                        if (acc_indexesA[k] != 255) bitsA.set(acc_indexesA[k], bit_low);
                        else                        bitsB.set(acc_indexesB[k], bit_low);
                    }

                    cpx sum = 0;
                    sum += A[bitsA.to_ulong()] * B[bitsB.to_ulong()];
                    size_t old_gray = 0;
                    
                    for (size_t m = 1; m < (1 << num_connections); m++) {
                        size_t gray_code = m ^ (m >> 1);
                        unsigned int position_vacant = sycl_ffsll(gray_code ^ old_gray) - 1;
                        
                        unsigned char indexA = acc_connectionsA[position_vacant];
                        unsigned char indexB = acc_connectionsB[position_vacant];
                        
                        bitsA.xor_op(1ULL << (rankA + indexA));
                        bitsB.xor_op(1ULL << indexB);
                        
                        sum += A[bitsA.to_ulong()] * B[bitsB.to_ulong()];
                        
                        old_gray = gray_code;
                    }
                    
                    result[i] = sum;
                }
            });
        });
        sycl::free(A, queue_);
        sycl::free(B, queue_);
    }
    
    template <typename T>
    void scan_array(const std::vector<T>& array, std::vector<size_t>& scan_array)
    {
        scan_array.resize(array.size() + 1);
        scan_array[0] = 0;
        for (size_t i = 0; i < array.size(); i++)
        {
            if constexpr (std::is_arithmetic_v<T>)
            {
                scan_array[i + 1] = scan_array[i] + array[i];
            }
            else
            {
                scan_array[i + 1] = scan_array[i] + array[i].size();
            }
        }
    }

    void general_batched_contraction(std::vector<cpx *> &A, std::vector<cpx *> &B, std::vector<cpx *> &result,
                                    std::vector<size_t> &rankA, std::vector<size_t> &rankB, std::vector<size_t> &rankC,
                                    std::vector<const unsigned char *> &spanA, std::vector<const unsigned char *> &spanB, std::vector<const unsigned char *> &spanC,
                                    const std::vector<std::vector<unsigned char>> &all_connections)
    {
        std::vector<size_t> scan_rankC;
        scan_array(rankC, scan_rankC);

        std::vector<unsigned char> num_connections;
        for (size_t i = 0; i < all_connections.size(); i++) num_connections.push_back(all_connections[i].size());

        std::vector<size_t> scan_connections;
        scan_array(num_connections, scan_connections);

        std::vector<unsigned char> indexesA(scan_rankC.back());
        std::vector<unsigned char> indexesB(scan_rankC.back());
        std::vector<unsigned char> indexes_connectionsA(scan_connections.back());
        std::vector<unsigned char> indexes_connectionsB(scan_connections.back());

        #pragma omp parallel for
        for (size_t i = 0; i < all_connections.size(); i++)
        {
            for (size_t j = 0; j < rankC[i]; j++)
            {
                indexesA[scan_rankC[i] + j] = getIndexInSet(spanA[i], spanC[i][j], rankA[i]);
                indexesB[scan_rankC[i] + j] = getIndexInSet(spanB[i], spanC[i][j], rankB[i]);
            }
        }


        #pragma omp parallel for
        for (size_t i = 0; i < all_connections.size(); i++) {
            for (size_t j = 0; j < all_connections[i].size(); j++) {
                indexes_connectionsA[scan_connections[i] + j] = getIndexInSet(spanA[i], all_connections[i][j], rankA[i]);
                indexes_connectionsB[scan_connections[i] + j] = getIndexInSet(spanB[i], all_connections[i][j], rankB[i]);
            }
        }

        sycl::buffer<unsigned char> buf_indexesA(indexesA);
        sycl::buffer<unsigned char> buf_indexesB(indexesB);
        sycl::buffer<unsigned char> buf_connectionsA(indexes_connectionsA);
        sycl::buffer<unsigned char> buf_connectionsB(indexes_connectionsB);
        sycl::buffer<size_t> buf_scan_rankC(scan_rankC);
        sycl::buffer<cpx*> buf_A(A.data(), A.size()), buf_B(B.data(), B.size()), buf_result(result.data(), result.size());
        sycl::buffer<size_t> buf_rankC(rankC), buf_rankA(rankA), buf_rankB(rankB);
        sycl::buffer<unsigned char> buf_num_connections(num_connections);
        sycl::buffer<size_t> buf_scan_connections(scan_connections);

        size_t nels = 0;
        for (size_t i = 0; i < rankC.size(); i++)
        {
            nels += 1 << (2 * rankC[i]);
        }

        const size_t WORK_GROUP_SIZE = 32;
        const size_t batch_size = rankC.size();

        queue_.submit([&](sycl::handler &h)
        {
            auto acc_indexesA = buf_indexesA.get_access<sycl::access::mode::read>(h);
            auto acc_indexesB = buf_indexesB.get_access<sycl::access::mode::read>(h);
            auto acc_connectionsA = buf_connectionsA.get_access<sycl::access::mode::read>(h);
            auto acc_connectionsB = buf_connectionsB.get_access<sycl::access::mode::read>(h);
            auto acc_scan_rankC = buf_scan_rankC.get_access<sycl::access::mode::read>(h);
            auto acc_num_connections = buf_num_connections.get_access<sycl::access::mode::read>(h);
            auto acc_scan_connections = buf_scan_connections.get_access<sycl::access::mode::read>(h);
            auto acc_rankA = buf_rankA.get_access<sycl::access::mode::read>(h);
            auto acc_rankB = buf_rankB.get_access<sycl::access::mode::read>(h);
            auto acc_rankC = buf_rankC.get_access<sycl::access::mode::read>(h);
            auto acc_A = buf_A.get_access<sycl::access::mode::read>(h);
            auto acc_B = buf_B.get_access<sycl::access::mode::read>(h);
            auto acc_result = buf_result.get_access<sycl::access::mode::write>(h);

            sycl::range<1> global{((nels + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE};
            sycl::range<1> local{WORK_GROUP_SIZE};

            h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
            
            const size_t gid = item.get_global_id(0);

            size_t batch_idx = 0;
            size_t result_offset = 0;
            for (size_t j = 0; j < batch_size; j++)
            {
                size_t n = 1 << (2 * acc_rankC[j]);
                if (gid < result_offset + n)
                {
                    batch_idx = j;
                    break;
                }
                result_offset += n;
            }

            if (gid < nels) {
                size_t i = gid - result_offset;
                sycl_classes::bitset bitsA, bitsB;

                for (size_t k = 0; k < acc_rankC[batch_idx]; k++) {
                    bool bit_high = ((i >> (acc_rankC[batch_idx] + k)) & 1) != 0;
                    bool bit_low = ((i >> k) & 1) != 0;
                
                    size_t idx = acc_scan_rankC[batch_idx] + k;
                    if (acc_indexesB[idx] != 255) bitsB.set(acc_rankB[batch_idx] + acc_indexesB[idx], bit_high);
                    else                          bitsA.set(acc_rankA[batch_idx] + acc_indexesA[idx], bit_high);
                
                    if (acc_indexesA[idx] != 255) bitsA.set(acc_indexesA[idx], bit_low);
                    else                          bitsB.set(acc_indexesB[idx], bit_low);
                }

                cpx sum = 0;
                sum += acc_A[batch_idx][bitsA.to_ulong()] * acc_B[batch_idx][bitsB.to_ulong()];
                size_t old_gray = 0;
                
                for (size_t m = 1; m < (1 << acc_num_connections[batch_idx]); m++) {
                    size_t gray_code = m ^ (m >> 1);
                    unsigned int position_vacant = sycl_ffsll(gray_code ^ old_gray) - 1;
                    
                    size_t conn_idx = acc_scan_connections[batch_idx] + position_vacant;
                    unsigned char indexA = acc_connectionsA[conn_idx];
                    unsigned char indexB = acc_connectionsB[conn_idx];

                    bitsA.xor_op(1ULL << (acc_rankA[batch_idx] + indexA));
                    bitsB.xor_op(1ULL << indexB);
                    
                    sum += acc_A[batch_idx][bitsA.to_ulong()] * acc_B[batch_idx][bitsB.to_ulong()];
                    
                    old_gray = gray_code;
                }
                
                acc_result[batch_idx][i] = sum;
            }

            });
        });

        for (size_t i = 0; i < rankC.size(); i++)
        {
            sycl::free(A[i], queue_);
            sycl::free(B[i], queue_);
        }
    }

    struct batch_metadata {
        std::vector<contraction_t>& batch;
        std::vector<cpx*> A;
        std::vector<cpx*> B;
        std::vector<cpx*> C;
        std::vector<size_t> rankA;
        std::vector<size_t> rankB;
        std::vector<size_t> rankC;
        std::vector<const unsigned char*> spanA;
        std::vector<const unsigned char*> spanB;
        std::vector<const unsigned char*> spanC;
        std::vector<std::vector<unsigned char>> all_connections;

        batch_metadata(
            std::vector<contraction_t>& batch, 
            std::vector<size_t> & new_gate_indices,
            std::vector<std::unique_ptr<sycl_gate_wrapper>>& gate_vector,
            sycl::queue queue_
        ) : batch(batch) {
            A = std::vector<cpx*>(batch.size());
            B = std::vector<cpx*>(batch.size());
            C = std::vector<cpx*>(batch.size());
            rankA = std::vector<size_t>(batch.size());
            rankB = std::vector<size_t>(batch.size());
            rankC = std::vector<size_t>(batch.size());
            spanA = std::vector<const unsigned char*>(batch.size());
            spanB = std::vector<const unsigned char*>(batch.size());
            spanC = std::vector<const unsigned char*>(batch.size());
            all_connections = std::vector<std::vector<unsigned char>>(batch.size());

            // First pass: create all new gates for batched contractions
            for (size_t i = 0; i < batch.size(); i++) {
                std::vector<unsigned char> result_qubits;
                result_qubits.insert(result_qubits.end(), 
                    gate_vector[batch[i].g1]->span.begin(), 
                    gate_vector[batch[i].g1]->span.end());
                result_qubits.insert(result_qubits.end(), 
                    gate_vector[batch[i].g2]->span.begin(), 
                    gate_vector[batch[i].g2]->span.end());
                
                sort(result_qubits.begin(), result_qubits.end());
                result_qubits.erase(unique(result_qubits.begin(), result_qubits.end()), 
                                result_qubits.end());
        
                // Create new gate
                gate_vector.push_back(std::make_unique<sycl_gate_wrapper>(sycl_gate_wrapper{
                    sycl::malloc_device<cpx>(1 << (2 * result_qubits.size()), queue_),
                    std::move(result_qubits)
                }));
                
                new_gate_indices.push_back(gate_vector.size() - 1);
                
                // Set up contraction inputs
                A[i] = gate_vector[batch[i].g1]->unitary;
                B[i] = gate_vector[batch[i].g2]->unitary;
                C[i] = gate_vector.back()->unitary;
                spanA[i] = gate_vector[batch[i].g1]->span.data();
                spanB[i] = gate_vector[batch[i].g2]->span.data();
                spanC[i] = gate_vector.back()->span.data();
                rankA[i] = gate_vector[batch[i].g1]->span.size();
                rankB[i] = gate_vector[batch[i].g2]->span.size();
                rankC[i] = gate_vector.back()->span.size();
                all_connections[i] = batch[i].connections;
            }

        }
    };

    void batched_contraction(
        std::vector<contraction_t>& batch,
        std::vector<contraction_t>& oneMath_batch,
        std::vector<std::unique_ptr<sycl_gate_wrapper>>& gate_vector,
        std::vector<size_t>& gate_pointer,
        size_t num_qubits)
    {
        // Store the indices where new gates will be inserted
        std::vector<size_t> new_gate_indices;
        
        // Create the batch metadata
        batch_metadata batch_meta(batch, new_gate_indices, gate_vector, queue_);
        batch_metadata oneMath_batch_meta(oneMath_batch, new_gate_indices, gate_vector, queue_);

        // general_batched_contraction(A, B, C, rankA, rankB, rankC, 
        //                           spanA, spanB, spanC, all_connections);    
        general_batched_contraction(
            batch_meta.A, batch_meta.B, batch_meta.C,
            batch_meta.rankA, batch_meta.rankB, batch_meta.rankC,
            batch_meta.spanA, batch_meta.spanB, batch_meta.spanC,
            batch_meta.all_connections
        );

        for (size_t i = 0; i < oneMath_batch_meta.rankC.size(); i++)
        {
            single_contraction(oneMath_batch_meta.A[i], oneMath_batch_meta.B[i], oneMath_batch_meta.C[i],
                            oneMath_batch_meta.rankA[i], oneMath_batch_meta.rankB[i], oneMath_batch_meta.rankC[i],
                            oneMath_batch_meta.spanA[i], oneMath_batch_meta.spanB[i], oneMath_batch_meta.spanC[i]);
        }

        // Update gate pointers and cleanup
        std::vector<size_t> indices_to_remove;
        for (size_t i = 0; i < batch.size(); i++) {
            // Store higher index first to maintain validity when erasing
            indices_to_remove.push_back(batch[i].g2i);
            
            // Update the pointer to the new gate
            gate_pointer[batch[i].g1i] = new_gate_indices[i];
        }

        for (size_t i = 0; i < oneMath_batch.size(); i++)
        {
            indices_to_remove.push_back(oneMath_batch[i].g2i);
            gate_pointer[oneMath_batch[i].g1i] = new_gate_indices[i + batch.size()];
        }
        
        // Sort in descending order to remove from back to front
        std::sort(indices_to_remove.begin(), indices_to_remove.end(), 
                  std::greater<size_t>());
        
        // Remove processed gates
        for (size_t idx : indices_to_remove) {
            gate_pointer.erase(gate_pointer.begin() + idx);
        }
        
        batch.clear();
        oneMath_batch.clear();
    }
};
extern "C"
{
    void single_contraction(cpx *A, cpx *B, cpx *C,
                            const size_t rankA, const size_t rankB, const size_t rankC,
                            const unsigned char *spanA, const unsigned char *spanB, const unsigned char *spanC)
    {
        static TensorContractor contractor;
        contractor.single_contraction(A, B, C, rankA, rankB, rankC,
                                      spanA, spanB, spanC);
    }

    void contract_circuit(gate *gates, size_t num_gates, cpx *result_gate, size_t num_qubits)
    {
        auto start = std::chrono::high_resolution_clock::now();

        static TensorContractor contractor;

        // make a vector holding the pointer to the gates
        std::vector<std::unique_ptr<sycl_gate_wrapper>> gate_vector(num_gates);
        std::vector<size_t> gate_pointer(num_gates);
        #pragma omp parallel for
        for (size_t i = 0; i < num_gates; i++)
        {

            auto wrapper = std::make_unique<sycl_gate_wrapper>(sycl_gate_wrapper{
                // sycl::buffer<cpx>(gates[i].unitary, sycl::range<1>(1 << (2 * gates[i].rank))),
                sycl::malloc_device<cpx>(1 << (2 * gates[i].rank), contractor.queue_),
                std::vector<unsigned char>(gates[i].qubits, gates[i].qubits + gates[i].rank)});
            // memcpy
            contractor.queue_.memcpy(wrapper->unitary, gates[i].unitary, sizeof(cpx) * (1 << (2 * gates[i].rank)));
            gate_vector[i] = std::move(wrapper);
            gate_pointer[i] = i;
        }

        size_t threshold = num_qubits;
        bool contraction = false;

        std::vector<contraction_t> batch;
        std::vector<contraction_t> oneMath_batch;

        while (gate_pointer.size() > 1)
        {
            std::vector<bool> batched(gate_pointer.size(), false);

            contraction = false;
            for (size_t ii = 0; ii < gate_pointer.size() - 1; ii++)
            {
                // print batched array
                if (batched[ii] || batched[ii + 1]) continue;
                size_t g1 = gate_pointer[ii];
                size_t g2 = gate_pointer[ii + 1];
                // get the connections between the two gates
                std::vector<unsigned char> connections = findCommonValues(
                    gate_vector[g1]->span,
                    gate_vector[g2]->span
                );

                if ( connections.size() >= threshold )
                {
                    if (gate_vector[g1]->span == gate_vector[g2]->span && std::is_sorted(gate_vector[g1]->span.begin(), gate_vector[g1]->span.end())) oneMath_batch.push_back(contraction_t{g1, g2, ii, ii + 1, connections});
                    else batch.push_back(contraction_t{g1, g2, ii, ii + 1, connections});

                    batched[ii] = true;
                    batched[ii + 1] = true;
                    
                    ii--;
                    contraction = true;
                }
            }
            if (!contraction) threshold = threshold - 1 >= 0 ? threshold - 1 : 0;
            else {contractor.batched_contraction(batch, oneMath_batch, gate_vector, gate_pointer, num_qubits); batch.clear();}
        }

        contractor.queue_.memcpy(result_gate, gate_vector[gate_pointer[0]]->unitary, sizeof(cpx) * (1 << (2 * num_qubits))).wait();
    }
}