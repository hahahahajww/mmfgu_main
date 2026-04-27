from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "books-nc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Robu" / "outputs" / "books_nc_mmfgu_gcn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMFGU client-unlearning robustness on books-nc"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn-type", choices=["sage", "gcn", "gat"], default="gcn")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-count", type=int, default=50)
    parser.add_argument("--purge-rounds", type=int, default=6)
    parser.add_argument("--purge-local-epochs", type=int, default=2)
    parser.add_argument("--prototype-threshold", type=float, default=0.65)
    parser.add_argument("--lambda-neg", type=float, default=0.15)
    parser.add_argument("--alpha-dec", type=float, default=1.0)
    parser.add_argument("--alpha-anchor", type=float, default=1.0)
    parser.add_argument("--beta-mm", type=float, default=1.0)
    parser.add_argument("--delta-bd", type=float, default=0.5)
    parser.add_argument(
        "--client-unlearn-ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = PROJECT_ROOT / "Robustness" / "run_mmfgu_client_robustness.py"
    cmd = [
        sys.executable,
        str(runner),
        "--data-dir",
        str(Path(args.data_dir)),
        "--task",
        "node_classification",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output-dir",
        str(Path(args.output_dir)),
        "--num-clients",
        str(args.num_clients),
        "--federated-rounds",
        str(args.federated_rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--gnn-type",
        args.gnn_type,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--probe-count",
        str(args.probe_count),
        "--purge-rounds",
        str(args.purge_rounds),
        "--purge-local-epochs",
        str(args.purge_local_epochs),
        "--prototype-threshold",
        str(args.prototype_threshold),
        "--lambda-neg",
        str(args.lambda_neg),
        "--alpha-dec",
        str(args.alpha_dec),
        "--alpha-anchor",
        str(args.alpha_anchor),
        "--beta-mm",
        str(args.beta_mm),
        "--delta-bd",
        str(args.delta_bd),
        "--client-unlearn-ratios",
        args.client_unlearn_ratios,
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
