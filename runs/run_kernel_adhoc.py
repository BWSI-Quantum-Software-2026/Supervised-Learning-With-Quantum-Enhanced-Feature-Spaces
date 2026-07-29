"""End-to-end kernel method on the paper's ad-hoc dataset."""
import numpy as np
from src.datasets import load_adhoc
from src.kernel.kernel_matrix import compute_kernel_matrix
from src.kernel.classify import train_svm, predict_svm


def main(shots=4096, seed=1):
    train_X, train_y, test_X, test_y = load_adhoc(training_size=15, test_size=5)
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")

    print("Computing training kernel matrix (quantum)...")
    K_train = compute_kernel_matrix(train_X, shots=shots, seed=seed)

    print("Training classical SVM (sklearn, precomputed kernel)...")
    clf = train_svm(K_train, train_y)

    print("Computing test kernel + predicting...")
    K_test = compute_kernel_matrix(test_X, train_X, shots=shots, seed=seed + 5000)
    preds = predict_svm(clf, K_test)
    acc = (preds == test_y).mean()
    print(f"Test accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
