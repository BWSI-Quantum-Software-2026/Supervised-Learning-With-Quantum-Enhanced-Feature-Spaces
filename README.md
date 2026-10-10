# Supervised Learning with Quantum-Enhanced Feature Spaces

**BWSI Quantum Software Final Project**

**Status:** COMPLETE. Both methods are implemented, tested, and run end to end on the simulator. Results are in the [Results](#results) section below and in `docs/RESULTS.md`.

Our project studies **quantum machine learning for binary classification** through a manual
implementation in Qiskit, where we reproduce the two methods from Havlíček et al.,
*Supervised learning with quantum-enhanced feature spaces* (Nature, 2019; arXiv:1804.11326).

Both methods described in the paper are implemented, and we split them between two sub-teams in our group:

- Quantum Kernel Estimation : the quantum computer estimates a kernel (similarity) matrix
  between data points; a classical support vector machine then finds the decision boundary.
- Variational Quantum Classifier (VQC) : a parameterized quantum circuit is trained directly
  to separate the two classes.

Both methods share the *same quantum feature map*, which is implemented once from primitive
gates (H, RZ, CNOT) and reused.

---

## Background

Steps the paper took:

1. Quantum feature map - converts data into a quantum state
2. Manufactured their own test data - labeled each point "+1" or "−1" based on the threshold
3. Quantum variational classifier - A quantum circuit with a classical optimizer that gets adjusted to minimize sorting mistakes
4. Kernel estimator - used the quantum computer only to measure how similar every pair of training points is then fed that similarity table into a classical SVM
5. Ran everything on IBM hardware - used 2 qubits
6. Applied error mitigation - ran each circuit at different speeds
7. Tested performance thoroughly - For the VQC, they tried circuits of increasing complexity and for the kernel method they tested three separate datasets


### The Feature Map
- Classical SVMs are useful for inner products efficiently evaluated by feature vectors not for classifiers based on quantum circuits.
- Integral to mapping our classical data points to Hilbert space and ensuring that our data is correctly parametrized for our VQC.
- The feature map is implemented through this unitary circuit so that it can be hard to compute classically.

Written out in plain text (the full derivation is in `docs/MATH.md`), the map applied to the all-zero state is:

```
U_Phi(x) = U_encode(x) · H^n · U_encode(x) · H^n
```

So it's Hadamards on every qubit, an encoding block, then the same thing again. For our 2 qubits the encoding block is just
Z-rotations on each qubit using the data values directly, plus one ZZ term between the two qubits whose angle is
`(pi - x0) * (pi - x1)`. That ZZ term is the entangling part and is what makes the kernel (conjecturally) hard to compute classically.
In code this is `src/feature_map.py` and it is only ~15 lines.

One gotcha we hit: Qiskit's `rz(theta)` is `exp(-i*theta*Z/2)` but the paper writes `exp(i*phi*Z)`, so every rotation angle in our
code is `-2 * phi`. See `docs/MATH.md` if you ever change the feature map.

## Method 1 - Quantum Kernel Estimation
A quantum computer used to measure how similar every pair of training points is then fed that similarity table into a classical SVM solver.

1. **Encode + overlap (quantum):** Encode point A, then run the reverse encoding for point B, then measure the chance of landing back on all zeros (similarity score).

2. **Assemble the kernel matrix (classical):** Collect all pairwise similarity scores into a table (matrix).

3. **Train the SVM (classical):** Feed that matrix into a normal SVM solver to find the best decision boundary.

4. **Classify (quantum + classical).** Get a new point's similarity to training points, and plug those scores into the trained SVM's formula to get the final answer.

## Method 2 - Variational Quantum Classifier
A quantum circuit with a classical optimizer that gets adjusted to minimize sorting mistakes

1. **Encode data as a quantum state:** Each feature of the data point sets a rotation angle on a qubit, lifting the data into a high-dimensional quantum space

2. **Apply the circuit:** A parameterized quantum circuit transforms the state

3. **Measure the result:** The qubits collapse into classical 0/1 outcomes and repeating this many times gives a probability distribution.

4. **Map outcome to label:** A rule that turns the measured probability into a classification. We use parity: an even number of 1s in the bitstring means +1, odd means −1.

5. **Train with SPSA:** Parameters start random and are adjusted to reduce wrong classification. SPSA estimates the gradient using two circuit evaluations per step (adjusting all parameters together). We also have COBYLA (through scipy) and the parameter-shift rule in `src/vqc/optimizer.py` since on the ad_hoc data COBYLA ended up converging a lot better than SPSA.


## Datasets

Both loaders live in `src/datasets.py` and return `train_X, train_y, test_X, test_y` with labels as −1/+1 so the kernel and VQC sides can use them interchangeably.

- **`ad_hoc` (primary).** The actual dataset from the paper, pulled from `qiskit_machine_learning.datasets.ad_hoc_data` with 2 features and `gap=0.3`. It is generated so that our exact feature map can separate it, which means a high accuracy here shows the implementation is *correct*, not that quantum beats classical. We seed it so everyone gets the same points.
- **`iris` (reality check).** The classic scikit-learn iris set, cut down to 2 of the 4 features (petal length and width) and 2 of the 3 species so it fits on our 2 qubits. This data was not designed around our feature map at all, so it tells us whether the method does anything on data that wasn't built for it. Each feature is rescaled into the 0 to π range before it becomes a rotation angle (not 2π, since `rz(-2x)` repeats every 2π).

---

## Results

All runs are on the Qiskit Aer simulator (`AerSimulator` for the kernel, `StatevectorSampler` for the VQC), 4096 shots per kernel entry and 1024 shots per VQC circuit. Scripts to reproduce every number are in `runs/`.

### Accuracy

| Method | Dataset | Train / test points | Test accuracy |
|---|---|---|---|
| Quantum kernel + SVM | ad_hoc | 40 / 20 | 100% |
| Quantum kernel + SVM | iris | 30 / 10 | 100% |
| VQC (2 ansatz layers, COBYLA, 200 iters) | ad_hoc | 30 / 10 | 100% |
| VQC (1 ansatz layer, SPSA, 50 iters) | iris | 30 / 10 | 100% |

ad_hoc is the "does it work" check and iris is the "does it do anything on real data" check. Because of shot noise the kernel numbers can wiggle a little between seeds.

### Resource estimation

From `runs/run_resources.py`. "hw" columns are after transpiling to the IBM basis gates (`rz`, `sx`, `x`, `cx`), which is roughly what the circuit would look like on real hardware.

| Circuit | Qubits | Depth | Gates | hw depth | hw gates |
|---|---|---|---|---|---|
| Feature map | 2 | 10 | 14 | 12 | 18 |
| Overlap (1 kernel entry) | 2 | 21 | 30 | 22 | 35 |
| VQC, 1 ansatz layer | 2 | 14 | 21 | 27 | 41 |
| VQC, 2 ansatz layers | 2 | 16 | 25 | 32 | 51 |
| VQC, 3 ansatz layers | 2 | 18 | 29 | 37 | 61 |

How many circuits one full run costs:

| Run | Circuits | Total shots |
|---|---|---|
| Kernel, 10 training points | 45 | 184,320 |
| Kernel, 20 training points | 190 | 778,240 |
| Kernel, 30 training points | 435 | 1,781,760 |
| Kernel, 40 training points | 780 | 3,194,880 |
| VQC, 30 training points, 50 iters | 1,500 | 1,536,000 |
| VQC, 30 training points, 200 iters | 6,000 | 6,144,000 |

The kernel only needs the upper half of the matrix since K(x,z) = K(z,x), so it is (n choose 2) circuits. The VQC runs one circuit per training point but has to redo all of them on every optimizer step, which is why it ends up costing more.

### Figures

Circuit diagrams and plots are in `results/figures/`: the feature map, the overlap circuit, the ansatz, the full VQC circuit, the iris scatter, and the VQC training curve.

---

## Verification (unit tests)

Tests are in `tests/` and run with pytest. All 9 pass in about 3 seconds.

```
python -m pytest
```

What they check:

- **Feature map** (`test_feature_map.py`): the circuit is 2 qubits, it rejects a 3-feature input, and applying the map then its inverse lands back on `|00>` every single shot.
- **Kernel** (`test_kernel.py`): a point compared to itself gives ~1, every entry stays in [0, 1], K(x,z) ≈ K(z,x), the full matrix is symmetric with ones on the diagonal, and a fixed seed gives the exact same answer twice. The tolerance is 0.05 to allow for shot noise.
- **VQC** (`test_vqc.py`): parity maps `00`/`11` to +1 and `01`/`10` to −1, the circuit is 2 qubits, the expectation value is deterministic for a fixed x and theta (seeded sampler), and the hard label is always ±1.


## Structure of repository
- `src/`
   - `kernel/`
     - `overlap_circuit.py` - feature map for x, inverse feature map for z, measure, count the `00`s
     - `kernel_matrix.py` - loops that over every pair to build the matrix
     - `classify.py` - wraps sklearn's SVM with `kernel="precomputed"`
   - `vqc/`
     - `ansatz.py` - the variational circuit W(θ): RY + RZ on each qubit, then CZ between neighbours
     - `model.py` - `VQCModel`: feature map + ansatz + measurement, parity labelling, `fit()` / `score()`
     - `optimizer.py` - SPSA, parameter-shift, and COBYLA training loops
     - `train.py` - standalone training script that also saves the training curve
   - `datasets.py` - `load_adhoc` and `load_iris_2feature` (shared by both teams)
   - `feature_map.py` - the feature map circuit (shared by both teams)
- `runs/` - one script per experiment: `run_kernel_adhoc.py`, `run_kernel_iris.py`, `run_vqc_adhoc.py`, `run_vqc_iris.py`, `run_resources.py`
- `tests/` - `test_feature_map.py`, `test_kernel.py`, `test_vqc.py`
- `docs/` - `MATH.md` (feature map derivation and the RZ sign convention) and `RESULTS.md`
- `results/figures/` - circuit diagrams and plots
- `papernotes.md` - our working notes on the paper
- `videoscript.md` - script for the project video
- `conftest.py`, `pytest.ini`, `requirements.txt`, `README.md`


## Setup

### Required software
- Python 3.12 or 3.13 (we used 3.13; current Qiskit supports 3.12 to 3.14)
- Visual Studio Code with the Python extension (optional but it's what we all used)
- Git + access to this repository

### How to replicate environment and setup

1. Clone the repo and open a terminal in the project folder.

   ```
   git clone https://github.com/BWSI-Quantum-Software-2026/Supervised-Learning-With-Quantum-Enhanced-Feature-Spaces.git
   cd Supervised-Learning-With-Quantum-Enhanced-Feature-Spaces
   ```

2. Make a virtual environment and activate it.

   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```

3. Install the dependencies (Qiskit 2.2.3, qiskit-aer, qiskit-machine-learning, numpy, scipy, scikit-learn, matplotlib, pytest).

   ```
   pip install -r requirements.txt
   ```

4. Run the tests to make sure everything is wired up.

   ```
   python -m pytest
   ```

5. Run an experiment. They all have to be run from the project root as modules so the `src` imports resolve.

   ```
   python -m runs.run_kernel_adhoc
   python -m runs.run_kernel_iris
   python -m runs.run_vqc_adhoc
   python -m runs.run_vqc_iris
   python -m runs.run_resources
   ```

   The kernel runs take a minute or two since they simulate a few hundred circuits. The VQC ad_hoc run is the slowest one because COBYLA does 200 iterations.

## Team

| Member | Sub-team | Tasks (interchangeable) |
|---|---|---|
| Tanush Kandpal | Kernel | Overlap circuit + kernel matrix + experiments |
| Eshanth Penumatsa | Kernel | Classical SVM + results |
| Avanti Moghe | VQC | Ansatz + circuit |
| Haya Fatima | VQC | Training loop + results |

Both files that are jointly shared: `feature_map.py`, `datasets.py` — before any changes both teams have to review.


## References

1. V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, J. M. Gambetta,
   *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212 (2019).
   https://arxiv.org/abs/1804.11326
2. Qiskit documentation. https://docs.quantum.ibm.com/
3. Qiskit Machine Learning (`ad_hoc_data`). https://qiskit-community.github.io/qiskit-machine-learning/
4. scikit-learn `SVC` with a precomputed kernel. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
5. M. Schuld, N. Killoran, *Quantum machine learning in feature Hilbert spaces*, PRL 122, 040504 (2019). https://arxiv.org/abs/1803.07128
