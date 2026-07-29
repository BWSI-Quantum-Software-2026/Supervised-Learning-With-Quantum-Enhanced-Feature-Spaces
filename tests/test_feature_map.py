import numpy as np
import pytest
from src.feature_map import feature_map_circuit
from qiskit_aer import AerSimulator

def test_feature_map_is_two_qubits():
    # build a feature map and check it has 2 qubits
    qc = feature_map_circuit(np.array([0.5, 1.2]))
    assert qc.num_qubits == 2


def test_feature_map_rejects_wrong_dimension():
    # length-3 vector should blow up
    with pytest.raises(ValueError):
        feature_map_circuit(np.array([0.1, 0.2, 0.3]))


def test_inverse_undoes_forward():

    # do the map then undo it, so we should land back on |00>
    fwd = feature_map_circuit(np.array([0.5, 1.2]))
    qc = fwd.compose(fwd.inverse())
    qc.measure_all()
    # run it and check every shot is '00'
    counts = AerSimulator().run(qc, shots=1024).result().get_counts()
    assert counts["00"] == 1024