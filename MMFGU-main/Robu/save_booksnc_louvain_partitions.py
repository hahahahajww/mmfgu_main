from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmfgu.data import _merge_louvain_communities, build_global_graph


DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "books-nc"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "client_partitions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Louvain client partitions for books-nc"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Building Louvain partition for seed={seed} ===")
        global_data = build_global_graph(data_dir, seed, "node_classification")
        parts = _merge_louvain_communities(global_data, args.num_clients, seed)
        payload = {
            "data_dir": str(data_dir),
            "seed": int(seed),
            "num_clients": int(args.num_clients),
            "num_nodes": int(global_data.num_nodes),
            "num_edges": int(global_data.edge_index.size(1)),
            "client_nodes": [nodes.cpu().clone() for nodes in parts],
            "client_sizes": [int(nodes.numel()) for nodes in parts],
        }
        output_path = output_dir / f"louvain_numclients{args.num_clients}_seed{seed}.pt"
        torch.save(payload, output_path)
        print(f"Saved: {output_path}")
        print("Client sizes:", payload["client_sizes"])


if __name__ == "__main__":
    main()
