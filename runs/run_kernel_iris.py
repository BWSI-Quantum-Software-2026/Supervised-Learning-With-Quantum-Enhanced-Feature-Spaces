# same kernel method as run_kernel_adhoc but on iris instead
# iris wasnt built around our feature map so whatever we get here is real
from src.datasets import load_iris_2feature
from src.kernel.kernel_matrix import compute_kernel_matrix
from src.kernel.classify import train_svm, predict_svm


def main(shots=4096, seed=1, n_train=30, n_test=10): # just the default values for shots and seed
    # gets the data. iris gives us 75 training points which is way too slow so we cut it
    # down to 30, same size as the adhoc run so the two numbers are comparable
    train_X, train_y, test_X, test_y = load_iris_2feature(seed=seed)
    train_X, train_y = train_X[:n_train], train_y[:n_train] # this is the syntax for slicing to get first 30 points and 30 labels
    test_X, test_y = test_X[:n_test], test_y[:n_test] # first 10 points and 10 labels

    # this is the quantum part - compare every training point against every other one and then build the similarity matrix
    print("Kernel matrix is being computed")
    K_train = compute_kernel_matrix(train_X, shots=shots, seed=seed)

    # matrix goes to sklearn and it can find the boundary
    clf = train_svm(K_train, train_y)

    # we have to also measure each test point against every training point
    K_test = compute_kernel_matrix(test_X, train_X, shots=shots, seed=seed + 1) # we also need a different seed (so anything but seed = seed) because we dont want the same randomness


    predictions = predict_svm(clf, K_test)   # svm turns similarities into -1/+1 guesses
    acc = (predictions == test_y).mean()     # fraction we got right
    print(f"Test accuracy: {acc:.3f}")


main()
