"""Utility functions for reproducibility, paths, and serialization."""

import json
import os
import random
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
DATA_V2_PROC = os.path.join(ROOT_DIR, "data", "v2", "processed")


def set_seed(seed=42):
    """Set random seed across standard python, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs():
    """Ensure output directories exist."""
    for d in (MODELS_DIR, RESULTS_DIR, CONFIGS_DIR):
        os.makedirs(d, exist_ok=True)


def save_json(data, filepath):
    """Save dict/list data as formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
