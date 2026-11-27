# AI53 - MLOps Project Phase 1 - Iyad EL HAJJ CHEHADE

## Milestone 0 - Starter Audit

### 0.1 Objective
The objective of this milestone is to audit the starter project and identify issues that prevent the machine-learning experiment from being fully reproducible. The audit focuses on random seeds, data validation, environment management, pipeline configuration, experiment tracking and model provenance.

### 0.2 Identified Issues
- 'src/reproducibility.py' - 'set_reproducibility()' | NumPy and PyTorch random seeds are not initialized, and CUDA/cuDNN reproducibility settings are incomplete. | Seed all relevant random-number generators and configure CUDA/cuDNN when available.
- 'src/reproducibility.py' - 'make_generator()' | The DataLoader generator is created without applying the requested seed. | Initialize the generator using the provided seed.
- 'src/reproducibility.py' - 'seed_worker()' | The function is not implemented, leaving DataLoader worker randomness uncontrolled. | Initialize Python and NumPy random generators using the PyTorch worker seed.
- 'src/validate_data.py' - 'validate_dataset()' | The function raises 'NotImplementedError' instead of validating the dataset. Missing classes, incorrect image counts and corrupted images are not detected before training. | Implement the required data-quality checks and generate a deterministic JSON report.
- 'src/provenance.py' - 'build_run_manifest()' | The function raises 'NotImplementedError' and does not collect experiment provenance, including the Git commit, data identity, environment versions, hyperparameters and MLflow Run ID. | Implement provenance collection and generate 'reports/run_manifest.json' to make training runs traceable.
- 'src/train.py' - MLflow logging | The MLflow integration contains incomplete TODOs. Training runs do not yet record all the required parameters, epoch-level metrics, dataset inputs, artifacts and final PyTorch model. | Complete MLflow tracking to record experiment parameters, metrics, provenance artifacts and the trained model for each run.

### 0.3 Initial Reproducibility Test Results

Command executed: 'uv run --locked pytest -q tests/test_reproducibility.py'

Results:
- 'test_global_seeding_repeats_python_numpy_torch' | FAILED | NumPy random values differed between executions with the same seed.
- 'test_generator_uses_requested_seed' | FAILED | Expected seed 7, but the generator returned 67280421310721.
- 'test_seed_worker_controls_numpy_and_python' | FAILED | Python and NumPy random values did not match the expected seeded values.

**Summary:** 3 tests failed, confirming that the starter's reproducibility functions require implementation. These failures are expected before Milestone 1 and provide evidence for the initial audit.

### 0.4 Initial Data Validation Test Results

Command executed: 'uv run --locked pytest -q tests/test_data_validation.py tests/test_validation_contract.py'

Results:
- Corrupted image detection | FAILED | The validator raises 'NotImplementedError' instead of detecting corrupted images.
- Valid dataset acceptance | FAILED | The validator cannot process a valid dataset.
- Missing split detection | FAILED | Missing dataset splits are not checked.
- Unexpected class detection | FAILED | Additional classes are not validated.
- Empty class detection | FAILED | Empty classes are not checked.
- Incorrect image count detection | FAILED | Expected image counts are not verified.
- Deterministic quality report | FAILED | The validator does not generate a reproducible report.

**Summary:** All 7 tests failed because 'validate_dataset()' is not implemented. The data-quality gate must be completed.

### 0.5 Initial Provenance Test Results

Command executed: 'uv run --locked pytest -q tests/test_provenance.py'

- 'test_run_manifest_contains_required_provenance' | FAILED | 'build_run_manifest()' raises 'NotImplementedError' instead of collecting experiment provenance.

**Summary:** The provenance test failed because the starter does not yet record the Git commit, data identity, environment versions, training settings and MLflow Run ID required to reproduce an experiment.

### 0.6 Engineering Checkpoint 1

Why is a working 'model.pt' insufficient evidence that an experiment is reproducible?

A working 'model.pt' only demonstrates that a trained model has been saved and can be used. It does not provide sufficient information to reproduce the training process.
Reproducibility requires the exact code version, dataset identity and splits, software dependencies, hyperparameters, random seeds and training configuration. Without this information, another developer cannot reliably reconstruct the experiment or verify how the model was obtained.
Therefore, a reproducible ML experiment requires version-controlled code, tracked data, a locked environment and documented training runs, rather than only a saved model file.

## Milestone 1 - Make Training Repeatable

### 1.1 Objective

The objective of this milestone is to control the main sources of randomness in the training pipeline to improve experiment reproducibility.

### 1.2 Implementation

The following functions were implemented in 'src/reproducibility.py':

- **'set_reproducibility(seed)'**: Initializes Python, NumPy and PyTorch random number generators. It also configures CUDA seeding when available and sets cuDNN deterministic and benchmark flags.
- **'make_generator(seed)'**: Creates a PyTorch generator initialized with the specified seed to support reproducible DataLoader shuffling.
- **'seed_worker(worker_id)'**: Initializes NumPy and Python random generators in DataLoader workers using the PyTorch worker seed.

### 1.3 Test Results

Command executed: 'uv run --locked pytest -q tests/test_reproducibility.py'

**Result: 3 passed.**

All three reproducibility tests now pass, whereas they initially failed during the starter audit.

### 1.4 Engineering Checkpoint 2 - Reproducibility

Seeding Python alone is insufficient because NumPy and PyTorch maintain separate random number generators. DataLoader shuffling and worker-level random transformations also require controlled seeds. The implemented functions improve reproducibility under the same execution conditions. However, seeds alone do not guarantee identical results across different hardware, library versions or nondeterministic GPU operations.

## Milestone 2 - Extract and Track the Images

### 2.1 Objective

The objective of this milestone is to ensure dataset versioning and reproducibility using Data Version Control (DVC). This allows a specific version of the dataset to be associated with a specific version of the source code without storing large image files directly in Git.

### 2.2 Dataset Extraction and Verification

The Beans dataset was extracted from the three provided ZIP archives using the dataset download script. All three archives successfully passed SHA-256 checksum verification. The extracted dataset was then independently verified using 'scripts/check_dataset.py'. 

The following results were obtained:

| Class | Training | Validation | Test | Total |
| Angular leaf spot | 345 | 44 | 43 | 432 |
| Bean rust | 348 | 45 | 43 | 436 |
| Healthy | 341 | 44 | 42 | 427 |
| **Total** | **1034** | **133** | **128** | **1295** |

All expected classes and image counts were confirmed. The dataset is well balanced across the three classes.

### 2.3 Dataset Versioning with DVC

DVC was initialized within the existing Git repository, and the raw dataset located in 'data/raw/' was tracked. 

The generated 'data/raw.dvc' metadata file contains the following information:

| Property | Value |
| Directory hash | '2b62d5a2e86898b281d0136cef5fdf1c.dir' |
| Hash algorithm | MD5 |
| Dataset size | 179,914,877 bytes |
| Number of tracked files | 1296 |
| Tracked path | 'raw' |

Although the dataset contains 1295 JPG images, DVC reports 1296 files because the directory also contains an additional non-image file named 'healthy_train.120tore', with a size of 6148 bytes. The additional file was preserved to avoid modifying the original extracted dataset. Its exact format has not been confirmed.

### 2.4 DVC Remote Configuration

A local DVC remote named 'lab-data' was configured as the default remote.
The storage directory is located outside the project repository at '../ai53-phase1-dvc-storage'.
The remote configuration was successfully verified using 'dvc remote list'.
This configuration separates the dataset storage from the Git repository and enables the retrieval of versioned data after the corresponding data objects have been pushed.

### 2.5 Engineering Checkpoint 3

**What does the Git remote store?**
The Git remote stores the source code, configuration files, documentation, DVC metadata files and commit history.

**What does the DVC remote store?**
The DVC remote stores the actual content of versioned datasets and other DVC-managed artifacts.

**Does git push transfer the DVC-managed raw images?**
No, git push transfers the DVC metadata files, while dvc push transfers the actual DVC-managed images to the configured DVC remote.

## Milestone 3 - Implement a Core Data-Quality Gate

### 3.1 Objective

The objective of this milestone is to implement an automated data-quality gate that validates the raw Beans dataset before data preparation and model training. Although DVC ensures dataset versioning and traceability, it does not guarantee that the dataset satisfies the structural and quality requirements of the machine learning application. Therefore, an explicit validation mechanism is required to detect invalid data and prevent unnecessary downstream computations.

### 3.2 Implementation

The 'validate_dataset()' function was implemented in 'src/validate_data.py' to perform the following checks:
- Verify the existence of the required training, validation and test splits.
- Ensure that each split contains exactly the three expected classes: 'angular_leaf_spot', 'bean_rust' and 'healthy'.
- Verify that the number of images in each split matches the expected dataset counts.
- Fully decode every supported image using Pillow's 'Image.load()' method to detect corrupted or unreadable images.
- Ensure that no class or split is empty.
- Generate a deterministic, JSON-serializable quality report.

Image files are identified using the supported extensions '.jpg', '.jpeg' and '.png'. Non-image files are excluded from the validation process without modifying the original dataset. 
The implementation follows a fail-fast approach: whenever a validation requirement is violated, an exception is raised immediately. Once integrated into the DVC pipeline, this mechanism will prevent dependent stages from executing when the dataset is invalid.

### 3.3 Unit Test Results

The following command was executed:

'''powershell
uv run --locked pytest -q tests/test_data_validation.py tests/test_validation_contract.py
'''

Result:

'''text
7 passed in 0.96s
'''

All seven validation tests passed, confirming that the implementation correctly accepts valid datasets, rejects invalid datasets and produces deterministic reports.

### 3.4 Real Dataset Validation

The implemented validator was executed on the actual Beans dataset using:

'''powershell
uv run --locked python -m src.validate_data
'''

The validation report was successfully generated at 'reports/data_quality.json'.

The following results were obtained:
| Class | Train | Validation | Test | Total |
| Angular leaf spot | 345 | 44 | 43 | 432 |
| Bean rust | 348 | 45 | 43 | 436 |
| Healthy | 341 | 44 | 42 | 427 |
| **Total** | **1034** | **133** | **128** | **1295** |

The validation status was 'PASS', confirming that all 1,295 images were successfully decoded and that all structural and image-count requirements were satisfied.
The results also demonstrate that the dataset is well balanced across the three classes.

### 3.5 Report Determinism

To experimentally verify report determinism, the validator was executed twice on the same dataset. After each execution, the SHA-256 hash of the generated report was calculated using:

'''powershell
Get-FileHash reports/data_quality.json -Algorithm SHA256
'''

Both executions produced the same SHA-256 hash:

'''text
38616E73BB9ED3DD315C4B3C28D5F1146976B4F2330AABBF35EE66D527E97AB3
'''

The identical hashes provide strong evidence that both executions generated byte-for-byte identical reports.
Determinism was achieved by sorting the class names, consistently serializing the JSON report and excluding timestamps and other execution-dependent information.
This property is important for DVC because unnecessary changes in output files can trigger unnecessary recomputation of dependent pipeline stages.

### 3.6 Full Test Suite Results

The complete project test suite was executed using:

'''powershell
uv run --locked pytest -q
'''

Result:

'''text
1 failed, 18 passed in 17.52s
'''

Compared with the results obtained after Milestone 1, the number of passing tests increased from 11 to 18, while the number of failing tests decreased from 8 to 1. The remaining failure corresponds to 'test_run_manifest_contains_required_provenance' in 'tests/test_provenance.py', because the 'build_run_manifest()' function in 'src/provenance.py' has not yet been implemented. 

### 3.7 Conclusion

The data-quality gate was successfully implemented and validated through automated unit tests and execution on the real dataset. The project can now automatically detect structural inconsistencies, incorrect image counts, empty classes and corrupted images before proceeding to downstream machine learning operations. The deterministic quality report also provides a reliable output for integration into the DVC pipeline in the next milestone.

## Milestone 4 — Design the DVC Pipeline from Scratch

### 4.1 Objective

The objective of this milestone is to define a reproducible and dependency-aware machine learning pipeline using DVC. The pipeline automates data validation, data preparation, candidate model training and final evaluation while avoiding unnecessary recomputation and preserving the independence of the test dataset.

### 4.2 Pipeline Architecture

A 'dvc.yaml' file was created at the project root with four stages: 'validate', 'prepare', 'train' and 'evaluate'.

The pipeline follows the architecture below:

'''text
          data/raw.dvc
           /       \
      validate      |
           \       /
            prepare
            /     \
         train   evaluate
'''

The TRAIN and EVALUATE stages are separate branches. TRAIN generates candidate models using the training and validation datasets, whereas EVALUATE loads an explicitly selected model from MLflow and measures its performance on the independent test dataset.

### 4.3 Stage Definition

| Stage | Main inputs | Main outputs |
| VALIDATE | Raw dataset and validation code | 'reports/data_quality.json' |
| PREPARE | Raw dataset, quality report and image size | Processed images, CSV manifests and 'reports/data_summary.json' |
| TRAIN | Prepared training and validation images, corresponding manifests, training parameters and Python modules | Local model checkpoint, training summary, training curves and run manifest |
| EVALUATE | Prepared test images, manifests, selected MLflow Run ID and evaluation parameters | Test metrics and confusion matrix |

Each stage includes 'pyproject.toml' and 'uv.lock' as dependencies to track changes to the Python environment. Relevant local Python modules are also declared as dependencies, including indirect imports.

### 4.4 Parameter Dependency Management

Individual parameters from 'params.yaml' are declared under the 'params' section of each stage.

PREPARE tracks 'data.image_size', while TRAIN tracks the parameters related to image size, training, augmentation, computation and MLflow configuration.

EVALUATE tracks 'selection.run_id', 'data.image_size', 'evaluation.batch_size', 'compute.device', 'tracking.uri' and 'tracking.experiment'.

This strategy prevents unrelated parameter modifications from triggering unnecessary stage executions.

### 4.5 Output and Cache Management

Large generated outputs, including processed images, dataset manifests and the local model checkpoint, are managed through the DVC cache.

Small report files are declared with 'cache: false' so that they can be versioned directly using Git.

The files 'reports/train_summary.json' and 'reports/test_metrics.json' are declared as DVC metrics because they contain numerical model performance results.

### 4.6 Dependency Design Questions

1. Should PREPARE depend on 'reports/data_quality.json'?
Yes. This dependency ensures that the data-quality gate succeeds before data preparation begins, preventing invalid datasets from reaching subsequent stages.

2. Should PREPARE also depend directly on 'data/raw'?
Yes. Raw image content can change without modifying the image counts or class distribution reported by VALIDATE. The direct dependency ensures that changes to the original images trigger data preparation even when the quality report remains unchanged.

3. Which changes in 'params.yaml' should make TRAIN run again?
Changes to 'data.image_size', training hyperparameters, augmentation settings, the computing device and MLflow tracking configuration should trigger TRAIN. Changes to 'evaluation.batch_size' and 'selection.run_id' should not trigger training because they concern final evaluation.

4. Why should EVALUATE track 'selection.run_id'?
The Run ID identifies the exact model selected from MLflow. Tracking this parameter ensures that changing the selected model invalidates previous evaluation results without triggering a new training execution.

5. Which file contains scores that DVC should track as metrics?
The training summary, 'reports/train_summary.json', contains training and validation performance metrics. The final evaluation report, 'reports/test_metrics.json', contains test accuracy and test Macro-F1. Both files are declared as DVC metrics.

### 4.7 Pipeline Verification

The pipeline definition was verified using the following commands:

'''bash
uv run --locked dvc stage list
uv run --locked dvc dag
'''

The stage listing successfully identified all four stages:

'''text
validate
prepare
train
evaluate
'''

The generated dependency graph confirmed that:
- VALIDATE depends on the raw dataset.
- PREPARE depends on both the raw dataset and the validation report.
- TRAIN depends on the prepared training and validation data.
- EVALUATE depends on PREPARE but does not depend directly on TRAIN.
- The pipeline contains no circular dependencies.


### 4.8 Conclusion

The four-stage DVC pipeline was successfully defined and its dependency graph was verified.

The architecture supports selective recomputation, dataset traceability, reproducible experiment management and independent final evaluation of a model selected from MLflow.

## 5. Milestone 5 — Experiment Provenance and MLflow Tracking

### 5.1 Objective

The objective of this milestone is to improve experiment traceability by recording the code version, dataset identity, software environment, training configuration, performance metrics and trained model associated with each training Run.

### 5.2 Experiment Provenance

The 'build_run_manifest()' function was implemented in 'src/provenance.py' to collect the following information:

| Category | Recorded information |
| Git | Commit SHA and working tree status |
| Data | SHA-256 identity of the training and validation manifests |
| Environment | Operating system, Python, PyTorch, torchvision, DVC, MLflow and CUDA versions |
| Configuration | Image size, batch size, epochs, learning rate, dropout and augmentation settings |
| Execution | Device, random seed and MLflow Run ID |

The provenance manifest is saved as 'reports/run_manifest.json' at the beginning of each training Run, before training generates additional outputs.

### 5.3 MLflow Integration

The seven TODO blocks in 'src/train.py' were implemented to provide experiment tracking.

1 | Capture experiment provenance
2 | Log the eleven required Run parameters
3 | Register the training manifest as an MLflow Dataset input
4 | Log training and validation metrics at every epoch
5 | Log total training duration
6 | Log the three required report artifacts
7 | Log the final trained PyTorch model

The recorded metrics are 'train_loss', 'val_loss', 'val_accuracy', 'val_macro_f1' and 'training_seconds'.

The following artifacts are logged separately for each Run:
- 'reports/training_curves.png'
- 'reports/train_summary.json'
- 'reports/run_manifest.json'

### 5.4 Model Packaging

The final CNN is configured to be logged in MLflow under the name 'bean-cnn'.

The model packaging implementation includes:
- A normalized float32 input example obtained from the validation dataset.
- A model signature describing the expected input and output tensor shapes.
- Metadata containing the class names and image size.
- CPU-based model serialization in evaluation mode.
- The model source code through 'code_paths=["src"]'.
- Pickle serialization, as required by the laboratory instructions.

The test dataset is not used during model training or model packaging.

### 5.5 Test Results

The complete automated test suite was executed using:

'''bash
uv run --locked pytest -q
'''

Result:

'''text
19 passed in 8.84s
'''

All automated tests passed successfully, corresponding to a 100% success rate.
The actual MLflow logging and model export will be verified during the controlled training experiments.

### 5.6 Engineering Checkpoint 4
- **Git:** Records the version and history of the source code and configuration files used for a training Run.
- **DVC:** Records dataset versions, pipeline dependencies, parameters and generated outputs associated with pipeline executions.
- **MLflow:** Records experiment parameters, dataset inputs, performance metrics, artifacts and trained models associated with each Run.

### 5.7 Conclusion
The experiment provenance and MLflow tracking implementations have been completed, and all automated tests pass. The project is ready for the controlled training experiments, during which the complete tracking workflow will be verified.

## 6. Milestone 6 — DVC Recomputation and Controlled Experiments

### 6.1 Objective

The objective of this milestone is to investigate DVC's dependency-based recomputation mechanism and compare two controlled training Runs using different learning rates.

### 6.2 Predicted Pipeline Behaviour

The following predictions assume that all pipeline outputs are initially up to date, each modification is considered independently, and the selected MLflow Run ID remains fixed.

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py` without changing the quality report | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

### 6.3 Engineering Checkpoint 5

**Learning-rate modification**

VALIDATE is reused because its source code, raw dataset and environment dependencies remain unchanged. PREPARE is reused because its preparation dependencies and image-size parameter remain unchanged. TRAIN must rerun because `train.learning_rate` is a declared parameter of the training stage. EVALUATE is reused because the selected MLflow Run ID, prepared data and evaluation parameters remain unchanged.

**Validation source-code modification**

VALIDATE must rerun because `src/validate_data.py` is a declared dependency. PREPARE is reused because the raw dataset and generated quality report remain identical. TRAIN is reused because the prepared training and validation data and their manifests remain unchanged. EVALUATE is reused because its dependencies and selected MLflow Run ID remain unchanged.

These predictions demonstrate that DVC determines stage recomputation based on declared dependencies, parameters and output changes rather than automatically rerunning the entire pipeline.

### 6.4 Model Selection Criteria and Experimental Hypothesis

The primary model selection metric is the final-epoch validation macro-F1 score because it gives equal importance to all three classes and considers both precision and recall.

In the event of a tie, the model with the highest final-epoch validation accuracy will be preferred. If both metrics are identical, the model with the shortest training duration will be selected.

Run A uses the baseline learning rate of 0.001, while Run B uses a lower learning rate of 0.0003. All other training parameters remain unchanged.

A lower learning rate is expected to produce smaller parameter updates and potentially more stable optimization. However, because training is limited to four epochs, Run B may converge more slowly and achieve lower final validation performance than Run A.

This hypothesis will be verified through controlled experiments.

### 6.5 Baseline Experiment — Run A

The MLflow tracking server was successfully started using a local SQLite backend. The baseline experiment was executed using: `uv run --locked dvc repro train`
DVC executed the VALIDATE, PREPARE and TRAIN stages successfully. The EVALUATE stage was not executed.
The dataset validation report confirmed that all 1,295 images passed the data-quality checks. The preparation stage resized the images to 96 × 96 pixels and generated the dataset manifests.

#### Baseline Results

| Metric | Value |
| Learning rate | 0.001 |
| Epochs | 4 |
| Training loss | 0.867199 |
| Validation loss | 0.832899 |
| Validation accuracy | 0.639098 |
| Validation macro-F1 | 0.625419 |
| Training duration | 11.2569 seconds |
| Total MLflow Run duration | 19.7 seconds |
| Device | CPU |

**MLflow Run ID:** `e220c216b7ff4f4bacca5a3eb8de09f5`

#### Training Behaviour

Both training and validation losses decreased throughout the four epochs, indicating successful optimization. Validation accuracy and macro-F1 reached their highest values at epoch 3 before decreasing slightly at epoch 4.
Since the pipeline saves the final-epoch model, the epoch 4 metrics are retained for the controlled experiment comparison.

#### Provenance Verification
The generated provenance manifest recorded the Git commit, training data identity, software environment, hyperparameters, random seed, execution device and MLflow Run ID.
The training data identity was successfully computed from the training and validation manifests:
`8fa943ff6c17a3c874d2b152e9d7980602a5cd50c0d8156625bd00590a3d8c6c`

Although the Git working tree was clean before pipeline execution, the provenance manifest recorded a dirty working tree at the beginning of TRAIN. This is consistent with DVC generating uncommitted pipeline outputs during the preceding stages.
The MLflow dashboard confirmed that Run A finished successfully and that the training dataset was registered under the name `bean-training` with the training context.

### 6.6 DVC Recomputation Checks

Two recomputation checks were performed after the baseline experiment using `uv run --locked dvc repro train`.

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
| Initial baseline execution | RERUN | RERUN | RERUN | NOT RUN |
| No changes | REUSED | REUSED | REUSED | NOT RUN |
| Comment added to `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN |

#### Check 1 — No Changes

Relevant terminal output:

```
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.
```

All stages were skipped because their dependencies, parameters and outputs remained unchanged. No additional MLflow Run was created.

#### Check 2 — Validation Source-Code Modification

A comment was added to `src/validate_data.py` without modifying its executable code.

Relevant terminal output:

```
Running stage 'validate':
> uv run --locked python -m src.validate_data
Updating lock file 'dvc.lock'

Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
```

VALIDATE was executed because its source-code dependency changed. However, the generated quality report remained identical. Therefore, PREPARE and TRAIN reused their existing outputs.

Both recomputation checks matched the initial predictions. These results demonstrate that DVC avoids unnecessary computation when downstream dependencies remain unchanged.