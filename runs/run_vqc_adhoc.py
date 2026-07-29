

from src.datasets import load_adhoc
from src.vqc.model import VQCModel


def main(shots=4096, seed=1):
    train_X, train_y, test_X, test_y = load_adhoc(training_size=15, test_size=5)
    # 2 layers, 1 wasnt enough to separate the data
    qvm = VQCModel(qubits=2, repetitions=2)
    # cobyla converges way better than spsa here
    qvm.fit(train_X, train_y, method="cobyla", max_iters=200)
    accuracy = qvm.score(test_X, test_y)
    print(f"Test accuracy: {accuracy:.4f}")


main()
