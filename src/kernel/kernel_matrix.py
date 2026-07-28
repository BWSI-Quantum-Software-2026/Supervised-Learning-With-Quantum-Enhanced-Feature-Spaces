import numpy as np
from qiskit_aer import AerSimulator

from .overlap_circuit import estimate_kernel_entry
def compute_kernel_matrix(A, B=None, shots=4096, seed=None):
    # if you have no B jst compare A with itself
    if B is None:
        m = len(A)
        # diagonal is already 1 bc a point vs itself is a perfect match
        K = np.eye(m)

        # only fill the top half, then mirror it this saves us half the work
        for i in range(m):
            for j in range(m):
                if j > i:
                    value = estimate_kernel_entry(A[i], A[j], shots=shots, seed=seed)
                    K[i][j] = value
                    K[j][i] = value  # same thing both ways
        return K
    # however if u got a B then jst compare A vs B
    else:
        n = len(A)
        p = len(B)
        K = np.zeros((n, p))
        #  then just run every pair: @Tanush see if there is a shortcut here if run time matters
        for i in range(n):
            for j in range(p):
                K[i][j] = estimate_kernel_entry(A[i], B[j], shots=shots, seed=seed)
        return K