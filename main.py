# Description: This file contains the main function that is called when the program is run. It is responsible for

import numpy as np
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

from ctypes import cdll
import ctypes

import opt_einsum as oe
import cupy as cp  # For GPU support

from cuquantum import contract, einsum

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

def get_unitary_with_qiskit(qc):
    simulator = AerSimulator(method='unitary', device='GPU')

    qc.save_unitary()
    transpiled_circuit = transpile(qc, simulator)

    start_time = time.time()

    job = simulator.run(transpiled_circuit)
    result = job.result()
    unitary_matrix = result.get_unitary(transpiled_circuit)

    end_time = time.time()

    execution_time_ms = (end_time - start_time) * 1000

    return unitary_matrix, execution_time_ms

def create_random_unitary(num_qubits_chosen, seed):
    return random_unitary(2**num_qubits_chosen, seed=seed)

def single_contraction_test():
    num_qubits = 2
    # create a circuit with a gate spanning all 10 qubits and one spanning the first 9
    circuit = QuantumCircuit(num_qubits)

    seed = 42
    np.random.seed(seed)

    circuit.unitary(create_random_unitary(num_qubits, seed), range(num_qubits))
    circuit.unitary(create_random_unitary(num_qubits - 1, seed), range(num_qubits - 1))
    
    return circuit

def get_error(matrix1, matrix2):
    """Calculate the Frobenius norm of the difference between two matrices."""
    diff = matrix1 - matrix2
    error = np.sqrt(np.sum(np.abs(diff) ** 2))
    return error

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

def contract_opteinsum(qc):
    gate_list = get_gate_list(qc)
    A_gate = gate_list[0]
    B_gate = gate_list[1]

    # reshape into ND arrays
    A_gate_unitary = A_gate.unitary.reshape([2] * len(A_gate.qubits) * 2)
    B_gate_unitary = B_gate.unitary.reshape([2] * len(B_gate.qubits) * 2)

    # Get all indices from the shapes of A and B 
    A_shape = A_gate_unitary.shape
    B_shape = B_gate_unitary.shape

    print(A_shape, B_shape)

    n_qubits = len(A_gate.qubits)
    
    # Create lists of indices for A and B
    A_indices = list(range(len(A_shape)))
    B_indices = list(range(len(A_shape), len(A_shape) + len(B_shape)))
    
    # Convert indices to letters for einsum notation
    letters = 'abcdefghijklmnopqrstuvwxyz'
    A_letters = [letters[i] for i in A_indices]
    B_letters = [letters[i] for i in B_indices]
    
    output_indices = ''.join(A_letters[:n_qubits]) + A_letters[-1] + ''.join(B_letters[n_qubits-1:])

    # Create the einsum string
    einsum_string = ''.join(A_letters) + ',' + ''.join(B_letters) + '->' + output_indices

    print(einsum_string)

    start_time = time.time()
    result = oe.contract(einsum_string, A_gate_unitary, B_gate_unitary)
    end_time = time.time()

    result = result.reshape(2**n_qubits, 2**n_qubits)
    return result, (end_time - start_time) * 1000


def main():
    qc = single_contraction_test()
    unitary_matrix_cpp, execution_time_ms_cpp = contract_cppsim(qc)
    unitary_matrix_oe, execution_time_ms_oe = contract_opteinsum(qc)
    unitary_matrix, execution_time_ms = get_unitary_with_qiskit(qc)

    print(f'Qiskit execution time: {execution_time_ms} ms')
    print(f'C++ execution time: {execution_time_ms_cpp} ms')
    print(f'Opteinsum execution time: {execution_time_ms_oe} ms')
    print(f'Error between Qiskit and C++: {get_error(unitary_matrix, unitary_matrix_cpp)}')
    print(f'Error between Qiskit and Opteinsum: {get_error(unitary_matrix, unitary_matrix_oe)}')

if __name__ == '__main__':
    main()