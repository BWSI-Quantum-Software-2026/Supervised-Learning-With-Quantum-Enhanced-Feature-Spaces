
from src.datasets import load_adhoc
from src.vqc.model import VQCModel

def main(shots=4096, seed=1, n_train=30, n_test=10):
    train_X, train_y, test_X, test_y = load_iris_2feature(seed=seed)
    train_X, train_y = train_X[:n_train], train_y[:n_train]
    test_X, test_y = test_X[:n_test], test_y[:n_test]

    qvm = VQCModel(qubits=2, repetitions=1)
    qvm.fit(train_X, train_y)
    accuracy = qvm.score(test_X, test_y)
    print(f"Test accuracy: {accuracy:.4f}")


main()