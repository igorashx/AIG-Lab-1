"""Utilities for reproducibility, serialization, and device selection."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Configure all supported random generators for repeatable experiments."""
    if seed < 0:
        raise ValueError("Seed-ul trebuie să fie un întreg nenegativ.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a separately seeded generator for one DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside each DataLoader worker process."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device() -> torch.device:
    """Return a CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_json(value: Any, path: str | Path) -> None:
    with ensure_parent(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def save_yaml(value: dict[str, Any], path: str | Path) -> None:
    with ensure_parent(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, allow_unicode=True, sort_keys=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Configurația din {path} trebuie să fie un mapping YAML.")
    return data