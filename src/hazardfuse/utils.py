"""Seeding, determinism and run provenance (environment, git commit, data checksums)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

PACKAGES = ("tensorflow", "keras", "numpy", "pandas", "scipy", "scikit-learn", "rasterio", "matplotlib", "pyyaml", "joblib")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow. With `deterministic`, TensorFlow ops (and tf.data) run deterministically:
    the same seed, data, versions and hardware give the same weights."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(int(seed))
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:          # TensorFlow < 2.9
            pass


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "packages": {}}
    for p in PACKAGES:
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    try:
        import tensorflow as tf
        info["tensorflow_devices"] = [d.name for d in tf.config.list_physical_devices()]
        info["tensorflow_build"] = {k: str(v) for k, v in tf.sysconfig.get_build_info().items()
                                    if k in ("cuda_version", "cudnn_version")}
    except Exception:
        pass
    return info


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except Exception:
        return None


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_run_manifest(out_dir: Path, cfg: dict, files: dict, extra: dict | None = None, repo_root: Path | None = None):
    """run_manifest.json: configuration, environment, git commit and SHA-256 of every input file."""
    man = {"created": datetime.now().isoformat(timespec="seconds"), "config": {k: v for k, v in cfg.items()},
           "environment": environment_info(), "git_commit": git_commit(repo_root) if repo_root else None,
           "inputs": {k: {"path": str(p), "sha256": sha256(p)} for k, p in files.items() if p and Path(p).exists()}}
    man.update(extra or {})
    (Path(out_dir) / "run_manifest.json").write_text(json.dumps(man, indent=2, default=str))
    return man


def slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name.replace("+", "plus")).strip("_")
