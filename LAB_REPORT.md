# AI53 — MLOps Project Phase 1

## Milestone 0 — Starter Audit

### 0.1 Objective

The objective of this milestone is to audit the starter project and identify issues that prevent the machine-learning experiment from being fully reproducible.

The audit focuses on random seeds, data validation, environment management, pipeline configuration, experiment tracking and model provenance.

### 0.2 Identified Issues

| File / Setting | Problem | Proposed Fix |
|---|---|---|
| `src/reproducibility.py` — `set_reproducibility()` | NumPy and PyTorch random seeds are not initialized, and CUDA/cuDNN reproducibility settings are incomplete. | Seed all relevant random-number generators and configure CUDA/cuDNN when available. |
| `src/reproducibility.py` — `make_generator()` | The DataLoader generator is created without applying the requested seed. | Initialize the generator using the provided seed. |
| `src/reproducibility.py` — `seed_worker()` | The function is not implemented, leaving DataLoader worker randomness uncontrolled. | Initialize Python and NumPy random generators using the PyTorch worker seed. |
| `src/validate_data.py` — `validate_dataset()` | The function raises `NotImplementedError` instead of validating the dataset. Missing classes, incorrect image counts and corrupted images are not detected before training. | Implement the required data-quality checks and generate a deterministic JSON report. |
| `src/provenance.py` — `build_run_manifest()` | The function raises `NotImplementedError` and does not collect experiment provenance, including the Git commit, data identity, environment versions, hyperparameters and MLflow Run ID. | Implement provenance collection and generate `reports/run_manifest.json` to make training runs traceable. |
| `src/train.py` — MLflow logging | The MLflow integration contains incomplete TODOs. Training runs do not yet record all the required parameters, epoch-level metrics, dataset inputs, artifacts and final PyTorch model. | Complete MLflow tracking to record experiment parameters, metrics, provenance artifacts and the trained model for each run. |

### 0.3 Initial Reproducibility Test Results

Command executed:

`uv run --locked pytest -q tests/test_reproducibility.py`

Results:

| Test | Result | Observation |
|---|---|---|
| `test_global_seeding_repeats_python_numpy_torch` | FAILED | NumPy random values differed between executions with the same seed. |
| `test_generator_uses_requested_seed` | FAILED | Expected seed 7, but the generator returned 67280421310721. |
| `test_seed_worker_controls_numpy_and_python` | FAILED | Python and NumPy random values did not match the expected seeded values. |

**Summary:** 3 tests failed, confirming that the starter's reproducibility functions require implementation.

These failures are expected before Milestone 1 and provide evidence for the initial audit.

### 0.4 Initial Data Validation Test Results

Command executed:

`uv run --locked pytest -q tests/test_data_validation.py tests/test_validation_contract.py`

| Test / Check | Result | Observation |
|---|---|---|
| Corrupted image detection | FAILED | The validator raises `NotImplementedError` instead of detecting corrupted images. |
| Valid dataset acceptance | FAILED | The validator cannot process a valid dataset. |
| Missing split detection | FAILED | Missing dataset splits are not checked. |
| Unexpected class detection | FAILED | Additional classes are not validated. |
| Empty class detection | FAILED | Empty classes are not checked. |
| Incorrect image count detection | FAILED | Expected image counts are not verified. |
| Deterministic quality report | FAILED | The validator does not generate a reproducible report. |

**Summary:** All 7 tests failed because `validate_dataset()` is not implemented. The data-quality gate must be completed in Milestone 3.

### 0.5 Initial Provenance Test Results

Command executed:

`uv run --locked pytest -q tests/test_provenance.py`

| Test | Result | Observation |
|---|---|---|
| `test_run_manifest_contains_required_provenance` | FAILED | `build_run_manifest()` raises `NotImplementedError` instead of collecting experiment provenance. |

**Summary:** The provenance test failed because the starter does not yet record the Git commit, data identity, environment versions, training settings and MLflow Run ID required to reproduce an experiment.

### 0.6 Engineering Checkpoint 1

**Question:** Why is a working `model.pt` insufficient evidence that an experiment is reproducible?

**Answer:**

A working `model.pt` only demonstrates that a trained model has been saved and can be used. It does not provide sufficient information to reproduce the training process.

Reproducibility requires the exact code version, dataset identity and splits, software dependencies, hyperparameters, random seeds and training configuration. Without this information, another developer cannot reliably reconstruct the experiment or verify how the model was obtained.

Therefore, a reproducible ML experiment requires version-controlled code, tracked data, a locked environment and documented training runs, rather than only a saved model file.