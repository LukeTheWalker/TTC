import numpy as np
import time
import sys

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from numpy import ctypeslib

from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ctypes import cdll
import ctypes

import opt_einsum as oe
import cupy as cp  # For GPU support

import cuquantum as cq

from tqdm import tqdm

# Define the complex number structure to match C++

class cgate(ctypes.Structure):
    _fields_ = [("qubits", ctypes.POINTER(ctypes.c_ubyte)),
                ("unitary", ctypes.POINTER(ctypes.c_double)),
                ("rank", ctypes.c_size_t)]

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

def pick_random_qubits(num_qubits, seed):
    # choose a random number of qubits from 1 to num_qubits
    num_qubits_chosen = np.random.randint(1, num_qubits + 1)
    # choose a random set of qubits
    qubits = np.random.choice(num_qubits, num_qubits_chosen, replace=False)
    qubits = sorted([int(q) for q in qubits])
    return qubits, num_qubits_chosen

def create_random_unitary(num_qubits_chosen, seed):
    return random_unitary(2**num_qubits_chosen, seed=seed)

def get_error(matrix1, matrix2):
    diff = matrix1 - matrix2
    error = np.sqrt(np.sum(np.abs(diff) ** 2))
    return error

def replace_swap_with_unitary(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Replace SWAP, controlled-SWAP, X, CX, and CCX gates in a quantum circuit with custom unitary gates,
    preserving the original position of each gate.
    
    Args:
        circuit (QuantumCircuit): Input quantum circuit
        
    Returns:
        QuantumCircuit: Modified circuit with SWAP, CSWAP, X, CX, and CCX gates replaced by custom unitaries
    """
    # Define the SWAP unitary matrix
    swap_unitary = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
    
    # Define the X (NOT) gate unitary matrix
    x_unitary = np.array([[0, 1],
                         [1, 0]])
    
    # Create controlled version of the SWAP unitary (8x8 matrix)
    # Initialize with identity matrix
    cswap_unitary = np.eye(8)
    # When control qubit is |1⟩, apply SWAP
    cswap_unitary[4:8, 4:8] = swap_unitary
    
    # Create controlled-X (CNOT) unitary (4x4 matrix)
    # Initialize with identity matrix
    cx_unitary = np.eye(4)
    # When control qubit is |1⟩, apply X
    cx_unitary[2:4, 2:4] = x_unitary
    
    # Create controlled-controlled-X (Toffoli) unitary (8x8 matrix)
    # Initialize with identity matrix
    ccx_unitary = np.eye(8)
    # When both control qubits are |1⟩, apply X
    ccx_unitary[6:8, 6:8] = x_unitary
    
    # Create the custom gates
    custom_swap = UnitaryGate(swap_unitary, label='CustomSWAP')
    custom_cswap = UnitaryGate(cswap_unitary, label='CustomCSWAP')
    custom_x = UnitaryGate(x_unitary, label='CustomX')
    custom_cx = UnitaryGate(cx_unitary, label='CustomCX')
    custom_ccx = UnitaryGate(ccx_unitary, label='CustomCCX')
    
    # Create a new circuit with the same number of qubits and classical bits
    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    
    # Iterate through the circuit instructions using the new attribute access method
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        if operation.name == 'swap':
            # Replace SWAP with custom unitary
            new_circuit.append(custom_swap, qubits)
        elif operation.name == 'cswap':
            # Replace controlled-SWAP with custom controlled unitary
            new_circuit.append(custom_cswap, qubits)
        elif operation.name == 'x':
            # Replace X (NOT) with custom unitary
            new_circuit.append(custom_x, qubits)
        elif operation.name == 'cx':
            # Replace CX (CNOT) with custom controlled unitary
            new_circuit.append(custom_cx, qubits)
        elif operation.name == 'ccx':
            # Replace CCX (Toffoli) with custom controlled-controlled unitary
            new_circuit.append(custom_ccx, qubits)
        else:
            # Keep other gates as they are
            new_circuit.append(operation, qubits, clbits)
    
    return new_circuit

def circuit_contraction_test(num_qubits):
    # create a circuit with a gate spanning all 10 qubits and one spanning the first 9
    circuit = QuantumCircuit(num_qubits)
    depth = 300

    seed = 0
    np.random.seed(seed)

    ## create 100 random unitaries
    for _ in range(depth):
        qubits_chosen, num_qubits_chosen = pick_random_qubits(num_qubits, seed)
        # qubits_chosen = list(range(num_qubits))
        # num_qubits_chosen = num_qubits
        circuit.unitary(create_random_unitary(num_qubits_chosen, seed), qubits_chosen)


    # circuit = random_circuit(num_qubits, depth=300, seed=0, max_operands=4)

    # circuit = replace_swap_with_unitary(circuit)

    circuit.unitary(np.eye(2**num_qubits), list(range(num_qubits)))

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
    num_qubits = qc.num_qubits

    unitary_matrix_cpp = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)

    # Convert the gate list to a list of cpx objects
    gates = []
    for gate in gate_list:
        qubits = (ctypes.c_ubyte * len(gate.qubits))(*gate.qubits)
        rank = len(gate.qubits)
        
        unitary = np.ascontiguousarray(gate.unitary, dtype=np.complex128)
        
        gates.append(cgate(qubits, unitary.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rank))

    lib = cdll.LoadLibrary('./build/libTTC.so')

    # Set up function argument types
    lib.contract_circuit.argtypes = [
        ctypes.POINTER(cgate),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t
    ]

    start_time = time.time()

    # Call the C++ function
    lib.contract_circuit(
        (cgate * len(gates))(*gates),
        len(gates),
        unitary_matrix_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        num_qubits
    )

    end_time = time.time()

    # Convert result back to numpy array
    unitary_matrix_cpp = unitary_matrix_cpp.reshape((2**num_qubits, 2**num_qubits))

    execution_time_ms = (end_time - start_time) * 1000

    return unitary_matrix_cpp, execution_time_ms

def circuit_contraction():
    sanity_check = False
    if len(sys.argv) < 2:
        print('Usage: python main.py <num_qubits>')
        sys.exit(1)
    if len(sys.argv) == 3 and sys.argv[2] == '--sanity':
        sanity_check = True

    num_qubits = int(sys.argv[1])
    qc = circuit_contraction_test(num_qubits)

    # Run each method 10 times and calculate averages
    qiskit_times = []
    cpp_times = []

    n_samples = 1 if sanity_check else 10
    
    for _ in tqdm(range(n_samples)):
        if not sanity_check:
            if num_qubits <= 10:
                unitary_matrix, time_ms = get_unitary_with_qiskit(qc)
                qiskit_times.append(time_ms)

        unitary_matrix_cpp, time_ms = contract_cppsim(qc)
        cpp_times.append(time_ms)


    if n_samples > 2 and not sanity_check:
        # remove best and worst times
        if num_qubits <= 10:
            qiskit_times.remove(max(qiskit_times))
            qiskit_times.remove(min(qiskit_times))
        cpp_times.remove(max(cpp_times))
        cpp_times.remove(min(cpp_times))

    execution_time_ms = np.mean(qiskit_times)
    execution_time_ms_cpp = np.mean(cpp_times) 

    # print("The two matrices are:")
    # print(unitary_matrix)
    # print(unitary_matrix_cpp)

    # gate_list = get_gate_list(qc)
    # print(gate_list[-1].unitary)
    # print("The gates are:")
    # for gate in gate_list:
    #     print(gate.unitary)

    print(f'Qiskit execution time: {execution_time_ms} ms')
    print(f'C++ execution time: {execution_time_ms_cpp} ms')
    if num_qubits <= 10 and not sanity_check:
        print(f'Error between Qiskit and C++: {get_error(unitary_matrix, unitary_matrix_cpp)}')

if __name__ == '__main__':
    circuit_contraction()