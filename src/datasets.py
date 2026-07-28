import numpy as np
# shared by both teams teams - kernel and vqc pull data from here 


def load_adhoc(training_size=20, test_size=5, gap=0.3, seed=12345): # training_size is PER CLASS so 20 gives us 40 points total
    # the paper's actual dataset. its made in a way that our feature map can easily
    # separate them and it will be accurate
    from qiskit_machine_learning.datasets import ad_hoc_data

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



