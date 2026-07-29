import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets import load_adhoc
from src.vqc.model import VQCModel


def main(shots=4096, seed=1):
    train_X, train_y, test_X, test_y = load_adhoc(training_size=15, test_size=5)
    qvm = VQCModel(qubits=2, repetitions=1)
    qvm.fit(train_X, train_y)
    accuracy = qvm.score(test_X, test_y)
    print(f"Test accuracy: {accuracy:.4f}")


main()
