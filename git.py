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
    unitary_matrix_oe, execution_time_ms_oe = contract_opteinsum(qc)
    unitary_matrix, execution_time_ms = get_unitary_with_qiskit(qc)

    print(f'Qiskit execution time: {execution_time_ms} ms')
    print(f'Opteinsum execution time: {execution_time_ms_oe} ms')
    print(f'Error between Qiskit and Opteinsum: {get_error(unitary_matrix, unitary_matrix_oe)}')

if __name__ == '__main__':
    main()