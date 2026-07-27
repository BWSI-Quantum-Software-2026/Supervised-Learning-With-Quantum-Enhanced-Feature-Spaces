# VQC sub-team
(Avanti and Haya Read this)
This folder is for the Variational Quantum Classifier implementation.

We share `src/feature_map.py` and `src/datasets.py` with the kernel team.

Planned files:
- `ansatz.py`   -- the variational circuit W(theta), which will be built from primitive gates
- `train.py`    -- classical optimizer loop (e.g. SPSA / COBYLA via scipy)
- `run_vqc.py`  -- end-to-end runner

method description is in the main README

