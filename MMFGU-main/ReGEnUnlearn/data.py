from __future__ import annotations

from pathlib import Path

from mmfgu.data import build_global_graph as _build_global_graph
from mmfgu.data import split_clients as _split_clients


def build_global_graph(data_dir: Path, seed: int):
    """Reuse the exact same graph construction as the existing mmfgu codebase."""

    return _build_global_graph(data_dir, seed, task="node_classification")


def split_clients(global_data, num_clients: int, seed: int):
    """Reuse the exact same client partition logic as mmfgu/FedEraser."""

    return _split_clients(global_data, num_clients, seed)
