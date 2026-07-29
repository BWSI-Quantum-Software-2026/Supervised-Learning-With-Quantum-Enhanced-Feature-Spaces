# runs the whole kernel method start to finish on with the ad-hoc dataset directly and also the accuracy 
from src.datasets import load_adhoc
from src.kernel.kernel_matrix import compute_kernel_matrix
from src.kernel.classify import train_svm, predict_svm


def main(shots=4096, seed=1): # just the default values for shots and seed
    # gets the data + training_size is per class so 15 means 30 training points and 5 means 10 test points 
    train_X, train_y, test_X, test_y = load_adhoc(training_size=15, test_size=5)

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