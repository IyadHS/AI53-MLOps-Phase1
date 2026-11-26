from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import torch

def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    TODO: collect at least:
      - Git commit SHA and whether the working tree is dirty;
      - identity of the prepared training + validation manifests;
      - Python, PyTorch, torchvision, DVC and MLflow versions;
      - OS/platform and CUDA runtime (when available);
      - compute device; random seed; image size; training parameters;
      - MLflow Run ID.

    Hints: subprocess, platform, importlib.metadata, torch,
    training_data_identity().
    """
    # 1. Collect Git information
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        git_dirty = bool(git_status)

    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unavailable"
        git_dirty = None

    # 2. Collect training-data identity
    data_identity = training_data_identity()

    # 3. Collect environment information
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "torchvision": metadata.version("torchvision"),
        "dvc": metadata.version("dvc"),
        "mlflow": metadata.version("mlflow"),
        "cuda": torch.version.cuda
    }

    # 4. Collect configuration information
    configuration = {
        "image_size": params["data"]["image_size"],
        "batch_size": params["train"]["batch_size"],
        "epochs": params["train"]["epochs"],
        "learning_rate": params["train"]["learning_rate"],
        "dropout": params["train"]["dropout"],
        "augmentation": {
            "enabled": params["augmentation"]["enabled"],
            "horizontal_flip": params["augmentation"]["horizontal_flip"],
            "rotation_degrees": params["augmentation"]["rotation_degrees"]
        }
    }

    # 5. Collect execution information
    execution = {
        "device": str(device),
        "random_state": params["train"]["random_state"],
        "mlflow_run_id": mlflow_run_id
    }

    manifest = {
        "git": {
            "commit": git_commit,
            "dirty": git_dirty
        },
        "data": {
            "training_data_identity": data_identity
        },
        "environment": environment,
        "configuration": configuration,
        "execution": execution
    }

    return manifest


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
