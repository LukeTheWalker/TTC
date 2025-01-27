import numpy as np
import time
import sys

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

from ctypes import cdll
import ctypes

import opt_einsum as oe
import cupy as cp  # For GPU support

import cuquantum as cq

from tqdm import tqdm

# Define the complex number structure to match C++
class cpx(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]

class Gate:
    def __init__(self, qubits, unitary, name=None):
        self.qubits = qubits  # span of qubits on which the gate acts
        self.unitary = unitary  # unitary matrix representing the gate
        self.name = name

    def __repr__(self):
        return f'Gate(qubits={self.qubits}, unitary=\n{self.unitary})'

def get_gate_list(qc):
    gates_list = []
    n_qubits = qc.num_qubits

    for instruction in qc.data:
        gate_operation = instruction.operation
        qubits = [qubit._index for qubit in instruction.qubits]
        gates_list.append(Gate(qubits, instruction.matrix, gate_operation.name))

    return gates_list

def create_random_unitary(num_qubits_chosen, seed):
    return random_unitary(2**num_qubits_chosen, seed=seed)

def get_error(matrix1, matrix2):
    diff = matrix1 - matrix2
    error = np.sqrt(np.sum(np.abs(diff) ** 2))
    return error

def single_contraction_test(num_qubits):
    # create a circuit with a gate spanning all 10 qubits and one spanning the first 9
    circuit = QuantumCircuit(num_qubits)

    seed = 42
    np.random.seed(seed)

    circuit.unitary(create_random_unitary(num_qubits, seed), range(num_qubits))
    circuit.unitary(create_random_unitary(num_qubits - 1, seed), range(num_qubits - 1))
    # circuit.unitary(create_random_unitary(num_qubits, seed), range(num_qubits))
    # circuit.unitary(create_random_unitary(num_qubits // 2, seed), range(0, num_qubits, 2))
    
    return circuit

def get_unitary_with_qiskit(qc):
    simulator = AerSimulator(method='unitary', device='GPU')

    transpiled_circuit = transpile(qc, simulator)
    transpiled_circuit.save_unitary()

    start_time = time.time()

    job = simulator.run(transpiled_circuit)
    result = job.result()
    unitary_matrix = result.get_unitary(transpiled_circuit)

    end_time = time.time()

    execution_time_ms = (end_time - start_time) * 1000

    return unitary_matrix, execution_time_ms

def contract_cppsim(qc):
    gate_list = get_gate_list(qc)
    A_gate = gate_list[0]
    B_gate = gate_list[1]

    # get the qubits on which the gates act
    qubits_A = A_gate.qubits
    qubits_B = B_gate.qubits

    # get the unitary matrices of the gates
    U_A = A_gate.unitary
    U_B = B_gate.unitary

    # Create the necessary arrays for the C++ function
    rankA = len(qubits_A)
    rankB = len(qubits_B)
    rankC = max(rankA, rankB)

    # Convert the unitary matrices to flat arrays of complex numbers
    A_flat = U_A.flatten()
    B_flat = U_B.flatten()
    C_flat = np.zeros(2**(2*rankC), dtype=complex)

    # Convert qubit indices to unsigned char arrays
    spanA = np.array(qubits_A, dtype=np.uint8)
    spanB = np.array(qubits_B, dtype=np.uint8)
    spanC = np.array(range(rankC), dtype=np.uint8)

    lib = cdll.LoadLibrary('./build/libTTC.so')

    # Convert numpy complex arrays to arrays of cpx structures
    A_cpx = (cpx * len(A_flat))()
    B_cpx = (cpx * len(B_flat))()
    C_cpx = (cpx * len(C_flat))()

    for i in range(len(A_flat)):
        A_cpx[i].real = A_flat[i].real
        A_cpx[i].imag = A_flat[i].imag

    for i in range(len(B_flat)):
        B_cpx[i].real = B_flat[i].real
        B_cpx[i].imag = B_flat[i].imag

    # Set up function argument types
    lib.single_contraction.argtypes = [
        ctypes.POINTER(cpx),
        ctypes.POINTER(cpx),
        ctypes.POINTER(cpx),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]

    start_time = time.time()

    # Call the C++ function
    lib.single_contraction(
        A_cpx,
        B_cpx,
        C_cpx,
        rankA,
        rankB,
        rankC,
        spanA.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        spanB.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        spanC.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        len(spanA),
        len(spanB),
        len(spanC)
    )

    end_time = time.time()

    # Convert result back to numpy array
    result = np.array([complex(c.real, c.imag) for c in C_cpx])
    return result.reshape((2**rankC, 2**rankC)), (end_time - start_time) * 1000

def get_einsum_str(qc):
    gate_list = get_gate_list(qc)
    A_gate = gate_list[0] 
    B_gate = gate_list[1]

    A_qubits = A_gate.qubits[::-1]
    B_qubits = B_gate.qubits[::-1]

    all_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    A_index = all_letters[:2 * len(A_gate.qubits)]

    qubits_in_common = set(A_gate.qubits) & set(B_gate.qubits)

    B_index = all_letters[2 * len(A_gate.qubits):2*len(A_gate.qubits) + 2*len(B_gate.qubits)]

    for qubit in qubits_in_common:
        B_index = B_index.replace(B_index[len(B_qubits) + B_qubits.index(qubit)], A_index[A_qubits.index(qubit)])

    C_index = A_index[:]

    for qubit in qubits_in_common:
        C_index = C_index.replace(C_index[A_qubits.index(qubit)], B_index[B_qubits.index(qubit)])

    einsum_str = f'{A_index},{B_index}->{C_index}'
    return einsum_str

def numpy_einsum(qc, einsum_str):
    gate_list = get_gate_list(qc)

    A_gate = gate_list[0] 
    B_gate = gate_list[1]

    U_A = A_gate.unitary.reshape([2] * 2 * len(A_gate.qubits))
    U_B = B_gate.unitary.reshape([2] * 2 * len(B_gate.qubits))

    start_time = time.time()
    result_matrix = np.einsum(einsum_str, U_A, U_B, optimize=True)
    end_time = time.time()

    C_index = einsum_str.split('->')[1]

    result_matrix = result_matrix.reshape(2 ** (len(C_index) // 2), 2 ** (len(C_index) // 2))
    return result_matrix, (end_time - start_time) * 1000

def cuquantum_einsum(qc, einsum_str):
    gate_list = get_gate_list(qc)

    A_gate = gate_list[0] 
    B_gate = gate_list[1]

    U_A = A_gate.unitary.reshape([2] * 2 * len(A_gate.qubits))
    U_B = B_gate.unitary.reshape([2] * 2 * len(B_gate.qubits))

    start_time = time.time()
    result_matrix = cq.contract(einsum_str, U_A, U_B)
    end_time = time.time()

    C_index = einsum_str.split('->')[1]

    result_matrix = result_matrix.reshape(2 ** (len(C_index) // 2), 2 ** (len(C_index) // 2))
    return result_matrix, (end_time - start_time) * 1000

def oe_einsum(qc, einsum_str):
    gate_list = get_gate_list(qc)

    A_gate = gate_list[0] 
    B_gate = gate_list[1]

    U_A = A_gate.unitary.reshape([2] * 2 * len(A_gate.qubits))
    U_B = B_gate.unitary.reshape([2] * 2 * len(B_gate.qubits))

    start_time = time.time()
    U_A_d = cp.asarray(U_A)
    U_B_d = cp.asarray(U_B)
    result_matrix_d = oe.contract(einsum_str, U_A_d, U_B_d, backend='cupy')
    result_matrix = cp.asnumpy(result_matrix_d)
    end_time = time.time()

    C_index = einsum_str.split('->')[1]

    result_matrix = result_matrix.reshape(2 ** (len(C_index) // 2), 2 ** (len(C_index) // 2))
    return result_matrix, (end_time - start_time) * 1000

def single_contraction():
    sanity_check = False
    if len(sys.argv) < 2:
        print('Usage: python main.py <num_qubits>')
        sys.exit(1)
    if len(sys.argv) == 3 and sys.argv[2] == '--sanity':
        sanity_check = True

    num_qubits = int(sys.argv[1])
    qc = single_contraction_test(num_qubits)

    # Run each method 10 times and calculate averages
    qiskit_times = []
    cpp_times = []
    oe_times = []
    cuquantum_times = []
    numpy_times = []

    einsum_str = get_einsum_str(qc)
    n_samples = 1 if sanity_check else 10
    
    for _ in tqdm(range(n_samples)):
        if not sanity_check:
            if num_qubits <= 10:
                unitary_matrix, time_ms = get_unitary_with_qiskit(qc)
                qiskit_times.append(time_ms)
                        
            unitary_matrix_oe, time_ms = oe_einsum(qc, einsum_str)
            oe_times.append(time_ms)

            unitary_matrix_cuq, time_ms = cuquantum_einsum(qc, einsum_str)
            cuquantum_times.append(time_ms)

            unitary_matrix_np, time_ms = numpy_einsum(qc, einsum_str)
            numpy_times.append(time_ms)

        unitary_matrix_cpp, time_ms = contract_cppsim(qc)
        cpp_times.append(time_ms)


    if n_samples > 2 and not sanity_check:
        # remove best and worst times
        if num_qubits <= 10:
            qiskit_times.remove(max(qiskit_times))
            qiskit_times.remove(min(qiskit_times))
        cpp_times.remove(max(cpp_times))
        cpp_times.remove(min(cpp_times))
        oe_times.remove(max(oe_times))
        oe_times.remove(min(oe_times))
        cuquantum_times.remove(max(cuquantum_times))
        cuquantum_times.remove(min(cuquantum_times))
        numpy_times.remove(max(numpy_times))
        numpy_times.remove(min(numpy_times))

    execution_time_ms = np.mean(qiskit_times)
    execution_time_ms_cpp = np.mean(cpp_times) 
    execution_time_ms_oe = np.mean(oe_times)
    execution_time_ms_cuq = np.mean(cuquantum_times)
    execution_time_ms_np = np.mean(numpy_times)

    print(f'Qiskit execution time: {execution_time_ms} ms')
    print(f'C++ execution time: {execution_time_ms_cpp} ms')
    print(f'Opteinsum execution time: {execution_time_ms_oe} ms')
    print(f'cuQuantum execution time: {execution_time_ms_cuq} ms')
    print(f'NumPy execution time: {execution_time_ms_np} ms')
    if num_qubits <= 10 and not sanity_check:
        print(f'Error between Qiskit and C++: {get_error(unitary_matrix, unitary_matrix_cpp)}')
        print(f'Error between Qiskit and Opteinsum: {get_error(unitary_matrix, unitary_matrix_oe)}')
        print(f'Error between Qiskit and cuQuantum: {get_error(unitary_matrix, unitary_matrix_cuq)}')
        print(f'Error between Qiskit and NumPy: {get_error(unitary_matrix, unitary_matrix_np)}')

if __name__ == '__main__':
    single_contraction()