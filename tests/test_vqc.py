#VQC TEAM: YOURS TO WRITE. <<<
#Ideas: the ansatz produces a 2-qubit circuit; for fixed theta and seed the
#output is deterministic; the cost decreases over training; parity labeling maps
#bitstrings to +/-1 correctly.
#"""
import pytest
import numpy as np
from src.vqc.model import VQCModel



def test_placeholder_vqc():
    # TODO (VQC team): replace with real tests

    #initializing model
    model = VQCModel(qubits = 2, repetitions = 1)

    #testing parity mapping
    assert model.calculate_parity("00") == 1
    assert model.calculate_parity("11") == 1
    assert model.calculate_parity("01") == -1
    assert model.calculate_parity("10") == -1

    #testing circuit qubit count
    assert model.circuit.qubits == 2

    #testing if output is deterministic for the fixed x and theta values
    x_val = np.array([0.5, 1.2])
    theta_val = np.array([0.1, 0.2, 0.3, 0.4])

    score1 = model.expectation_value(x_val, theta_val)
    score2 = model.expectation_value(x_val, theta_val)

    assert np.isclose(score1, score2, atol=1e-5)

    #testing hard binary classification output
    label = model.label(x_val, theta_val)
    assert label in [1, -1]



