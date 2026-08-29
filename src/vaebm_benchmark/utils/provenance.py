"""Reproducibility metadata: file checksums, dataset download manifests, and
environment capture (package versions, CPU/GPU). Every artifact this
package downloads or produces should be traceable back through here.

Mirrors DTEA's `dtea.datasets.base` checksum-manifest pattern (SHA256 per
raw file, recorded once at download time, re-verified on demand) - the same
idea, reimplemented standalone here rather than imported, since this is a
separate, non-modified repository.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Optional

import yaml


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: Path, extra: Optional[dict] = None) -> Path:
    """Writes MANIFEST.yaml recording the SHA256 of every file currently in
    `directory` (except the manifest itself). Call this immediately after
    downloading/generating a dataset artifact, before it is ever read."""
    files = {
        p.name: sha256_of(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != "MANIFEST.yaml"
    }
    manifest = {"files": files}
    if extra:
        manifest.update(extra)
    manifest_path = directory / "MANIFEST.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=True)
    return manifest_path


def verify_manifest(directory: Path) -> tuple[bool, list[str]]:
    """Recomputes checksums for every file in `directory` and compares
    against MANIFEST.yaml. Returns (ok, problems) - problems is a list of
    human-readable mismatch/missing-file descriptions, empty iff ok."""
    manifest_path = directory / "MANIFEST.yaml"
    if not manifest_path.exists():
        return False, [f"no MANIFEST.yaml in {directory}"]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    problems = []
    for filename, expected in manifest.get("files", {}).items():
        fpath = directory / filename
        if not fpath.exists():
            problems.append(f"missing file: {filename}")
            continue
        actual = sha256_of(fpath)
        if actual != expected:
            problems.append(f"checksum mismatch: {filename} (expected {expected[:12]}..., got {actual[:12]}...)")
    return (len(problems) == 0), problems


def capture_environment() -> dict:
    """Best-effort snapshot of the runtime: Python version, OS, and the
    version of every package this project depends on that happens to be
    importable right now (missing ones are simply omitted, not errors -
    not every experiment needs every optional dependency)."""
    info: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }

    packages = [
        "numpy", "scipy", "sklearn", "gensim", "torch", "tensorflow",
        "sentence_transformers", "fastopic", "topmost", "transformers",
    ]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    info["packages"] = versions

    info["gpu"] = _capture_gpu()
    return info


def _capture_gpu() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "backend": "torch/cuda",
                "device_name": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return {"available": True, "backend": "tensorflow", "devices": [g.name for g in gpus]}
    except Exception:
        pass
    return {"available": False}
