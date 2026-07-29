# how big our circuits are and how many of them we have to run
# depth is how many layers of gates run one after another, and the gates is the total count
import numpy as np
from qiskit import transpile

from src.feature_map import feature_map_circuit
from src.kernel.overlap_circuit import circuit_overlap
from src.vqc.model import VQCModel

# the only gates real ibm hardware runs. our h gates get rewritten into these, so the
# transpiled numbers are the honest ones for what a real machine would do
IBM_GATES = ['rz', 'sx', 'x', 'cx']


def show(name, qc): # prints one row of the table so we dont repeat this 5 times
    # transpile swaps our gates for ones real hardware actually has, so hw comes out
    # bigger than what we wrote
    hw = transpile(qc, basis_gates=IBM_GATES, optimization_level=1)
    # the :<27 bits are just column widths to try to make sure the tables  lines up (not a big deal tho)
    print(f"{name:<27} {qc.num_qubits:<7} {qc.depth():<7} {qc.size():<7} {hw.depth():<9} {hw.size()}")


def main():
    # any two points work, the circuit shape doesnt depend on the actual values
    x = np.array([0.7, 1.3])
    z = np.array([2.1, 0.4])

    print(f"{'circuit':<27} {'qubits':<7} {'depth':<7} {'gates':<7} {'hw depth':<9} hw gates")

    show("feature map", feature_map_circuit(x))
    show("overlap (one kernel entry)", circuit_overlap(x, z)) # feature map forward then backward
    # more layers just means more parameters to train, this shows what that costs us in depth
    for layers in [1, 2, 3]:
        show(f"vqc, {layers} ansatz layers", VQCModel(qubits=2, repetitions=layers).circuit)

    print("\ncircuits needed per run")

    # kernel only needs the top half of the matrix since K(x,z) and K(z,x) are the same
    for points in [10, 20, 30, 40]:
        pairs = points * (points - 1) // 2
        print(f"  kernel, {points} training points: {pairs} circuits ({pairs * 4096:,} shots total)")

    # the vqc runs one circuit per point, but has to redo all of them every optimizer step
    for iters in [50, 200]:
        circuits = 30 * iters
        print(f"  vqc, 30 training points, {iters} iters: {circuits} circuits ({circuits * 1024:,} shots total)")


main()
