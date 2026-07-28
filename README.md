# Supervised Learning with Quantum-Enhanced Feature Spaces

**BWSI Quantum Software Final Project**

**Status:** WORK IN PROGRESS. Environment and repo skeleton is complete; working on code and steps ATM

**Tentative Deadlines:** code & simulation: 07/30 | video: 07/31 · 

Our project studies **quantum machine learning for binary classification** through a manual
implementation in Qiskit, where we reprodruce the two methods from Havlíček et al.,
*Supervised learning with quantum-enhanced feature spaces* (Nature, 2019; arXiv:1804.11326).

 Both methods described in the paper are implemented, and we split them between two sub-teams in our group:

- Quantum Kernel Estimation : the quantum computer estimates a kernel (similarity) matrix
  between data points; a classical support vector machine then finds the decision boundary.
- Variational Quantum Classifier (VQC) : a parameterized quantum circuit is trained directly
  to separate the two classes.

Both methods share the *same quantum feature map*, which is implemented once from primitive
gates (H, RZ, CNOT) and reused. [tanush will finish writing this ]
---

## Background

-- just sumamrize from paper

### The feature map
-- explanation needs to be written

## Method 1 — Quantum Kernel Estimation
(tanush and eshanth -- expand --)
1. **Encode + overlap (quantum).** 
2. **Assemble the kernel matrix (classical).** 
3. **Train the SVM (classical).** 
4. **Classify (quantum + classical).** 

## Method 2 — Variational Quantum Classifier
(avanti and haya willw rite the steps for this)


---

## Datasets

- **`ad_hoc` (primary).** The actual dataset from the paper 

+ most likely iris as well --->
- **`iris` (for check - not sure yet).** 
> `ad_hoc_data` - 
> `qiskit_machine_learning`. - 

---

## Verification (unit tests)

[Still to be written]


## Structure of repistory 
[Still to be written]


## Setup

### Required software
- Python 3.12 (3.12–3.14 supported by current Qiskit)
- Visual Studio Code with the Python extension
- Git + access to this repository

## How to replicate environment and setup
[Still to be written]

## Team

| Member | Sub-team | Tasks (interchangeable) |
|---|---|---|
| [Tanush Kandpal] | Kernel | Overlap circuit + kernel matrix |
| [Eshanth Penumatsa] | Kernel | Classical SVM + experiments + results |
-- 
| [Avanti Moghe] | VQC | Ansatz + circuit |
| [Haya Fatima] | VQC | Training loop + results |

Both files that are cojointedly shared: `feature_map.py`, `datasets.py` — before any changes both teams have to review.


## References

1. V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, J. M. Gambetta,
   *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212 (2019).
   https://arxiv.org/abs/1804.11326
2. TBD: Qiskit documentation

