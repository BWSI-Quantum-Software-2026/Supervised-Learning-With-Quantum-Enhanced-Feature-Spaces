from src.feature_map import feature_map_circuit # import our feature map circuit
from qiskit_aer import AerSimulator # import simulator
from qiskit import QuantumCircuit  
# computer + uncompute circuit --> feature map then reverse then then measure

def circuit_overlap(x , z):
    fx = feature_map_circuit(x)  # circuit that will encode the input data x to a quantum state
    fz = feature_map_circuit(z).inverse() # encodes z then reverses every gate
    qc = QuantumCircuit(2,2)
    qc.compose(fx, inplace=True)  # forward map of x and then inverse map of z in next line
    qc.compose(fz, inplace=True) # this composes the two circuits together so we can measure the overlap b/w the two states
    qc.measure(range(2), range(2))
    return qc

def estimate_kernel_entry(x, z, shots = 4096, simulator = None, seed = None):
    if simulator is None:
        simulator = AerSimulator() # if no simulator is passed then we make one
    qc = circuit_overlap(x, z)
    a = simulator.run(qc, shots=shots, seed_simulator = seed).result()
    counts = a.get_counts()
    return counts.get('00', 0) / shots # this is the overlap -- measure of how similar x and z are

