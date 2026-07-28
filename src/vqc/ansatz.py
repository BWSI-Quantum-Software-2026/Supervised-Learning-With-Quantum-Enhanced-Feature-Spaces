
from qiskit import QuantumCircuit
from qiskit import circuit
from qiskit.circuit import ParameterVector




def ansatz(qubits: int, repetitions: int = 2):
    circuit = QuantumCircuit(qubits, name="Ansatz_W(θ)")

    params = 2 * qubits * repetitions
    """
    1 Ry rotation + 1 Rz rotation (each one parameter per qubit) so that makes 2.
    You scale this by number of qubits being dealt with and the repetitions being made hence 2*qubits*repetitions.
    """
    theta = ParameterVector('θ', length=params)

    counter = 0
    for layer in range(repetitions):
        for q in range(qubits):
            circuit.ry(theta[counter], q)
            counter += 1
            circuit.rz(theta[counter], q)
            counter += 1
    """
    variational layer with parametrized single-qubit rotations acting on each each qubit.
    Paper states that both these rotations are conducted on the bloch sphere. 
    counter does work of moving through the parameters

    """

    for q in range(qubits - 1):
        circuit.cz(q, q + 1)

    """
    Entangles using CZ gates between each nearest neighbor. 
    """


    return circuit, theta