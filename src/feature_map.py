import numpy as np
from qiskit import QuantumCircuit

def applye(qc,x):
# each encode block is super simple allw we need to do is two single-qubit gates Z-rotations + ZZ 
    qc.rz(-2 * x[0], 0)
    qc.rz(-2 * x[1], 1)
    phi_zz = (np.pi - x[0]) * (np.pi - x[1])
    qc.cx(0, 1)
    qc.rz(-2 * phi_zz, 1)
    qc.cx(0, 1)

def feature_map_circuit(x):
    if x.shape != (2,):
        raise ValueError
    qc = QuantumCircuit(2)
    qc.h(range(2))          # first Hadamard layer
    applye(qc, x)  # first encode block
    qc.h(range(2))          # second Hadamard layer
    applye(qc, x)  # second encode block
    return qc