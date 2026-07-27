# VQC sub-team
(Avanti and Haya Read this)
This folder is for the Variational Quantum Classifier implementation.

We share `src/feature_map.py` and `src/datasets.py` with the kernel team.

Files needed:
- `ansatz.py`   - the variational circuit W(theta), which will be built from the gates
- `train.py`    - classical optimizer loop (e.g. SPSA / COBYLA via scipy)
- `run_vqc.py`  - to load the data, then train then print the results 

method description is in the main README

