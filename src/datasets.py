import numpy as np
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# shared by both teams teams - kernel and vqc pull data from here 


def load_adhoc(training_size=20, test_size=5, gap=0.3, seed=12345): # training_size is PER CLASS so 20 gives us 40 points total
    # the paper's actual dataset. its made in a way that our feature map can easily
    # separate them and it will be accurate
    # qiskit has its own global seed and so we set it so we get the same points every run
    try:
        from qiskit_machine_learning.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    except Exception:
        pass 
    # 

    train_X, train_y, test_X, test_y = ad_hoc_data(
        training_size=training_size,
        test_size=test_size,
        n=2,        # 2 features bc we only have 2 qubits
        gap=gap,    # how much space to leave between the two classes
        one_hot=False,
    )

    # comes back as 0/1 but the svm side expects -1/+1
    train_y = np.where(train_y == 0, -1, 1)
    test_y = np.where(test_y == 0, -1, 1)

    return train_X, train_y, test_X, test_y # X is the points, y is the -1/+1 labels

def load_iris_2feature(seed=0): # seed just has to be some fixed val 
    # data that is not designed around our feature map whatsoever -- kind of like a reality check
    # main test is to check if  the method does anything on data that wasn't meant for it
    data = load_iris()

    # Keep only 2 of the 4 features (petal length & width) b/c of our 2 circuit qubit and 2 of the 3 species.
    X = data.data[:, [2, 3]]
    y = data.target
    keep = y < 2
    X = X[keep]
    y = y[keep]

    # we have to rescale each feature into the 0, 2pi range for our rotation angles .
    smallest = X.min(axis=0)
    largest = X.max(axis=0)
    X = (X - smallest) / (largest - smallest) * (2 * np.pi)

    # Relabel the two classes as -1 / +1.
    y = np.where(y == 0, -1, 1)

    # default split for training & test (75/25) - as a reference - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return train_test_split(X, y, test_size=0.25, random_state=seed) 
