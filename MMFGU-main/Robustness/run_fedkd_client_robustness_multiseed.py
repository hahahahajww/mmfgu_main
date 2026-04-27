from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def parse_ratio_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def metric_mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def resolve_cuda_type(device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedKD robustness across multiple seeds")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn-type", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--distill-epochs", type=int, default=8)
    parser.add_argument("--distill-lr", type=float, default=5e-4)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--client-unlearn-ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    return parser.parse_args()


def build_command(args: argparse.Namespace, seed: int, output_dir: Path) -> list[str]:
    return [
        args.python,
        str(Path(__file__).resolve().parent / "run_fedkd_client_robustness.py"),
        "--data-dir", args.data_dir,
        "--task", "node_classification",
        "--seed", str(seed),
        "--device", args.device,
        "--output-dir", str(output_dir),
        "--num-clients", str(args.num_clients),
        "--federated-rounds", str(args.federated_rounds),
        "--local-epochs", str(args.local_epochs),
        "--eval-interval", str(args.eval_interval),
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--gnn-type", args.gnn_type,
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--distill-epochs", str(args.distill_epochs),
        "--distill-lr", str(args.distill_lr),
        "--distill-temperature", str(args.distill_temperature),
        "--client-unlearn-ratios", args.client_unlearn_ratios,
    ]


def main() -> None:
    args = parse_args()
    dataset_name = Path(args.data_dir).name
    cuda_type = resolve_cuda_type(args.device)
    output_root = Path(args.output_root) / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    ratios = parse_ratio_list(args.client_unlearn_ratios)

    per_seed = []
    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running FedKD robustness seed={seed} ===")
        subprocess.run(build_command(args, seed, run_dir), check=True)
        with open(run_dir / "fedkd_robustness_summary.json", "r", encoding="utf-8") as file:
            per_seed.append(json.load(file))

    ratio_rows = []
    for ratio in ratios:
        matched_runs = []
        for seed_payload in per_seed:
            for row in seed_payload["runs"]:
                if abs(float(row["client_unlearn_ratio"]) - float(ratio)) < 1e-9:
                    matched_runs.append(row)
                    break
        before_test = [row["before_global_metrics"]["avg_test_acc"] for row in matched_runs]
        after_test = [row["after_global_metrics"]["avg_test_acc"] for row in matched_runs]
        delta_test = [row["metric_delta"]["avg_test_acc_delta"] for row in matched_runs]
        ratio_rows.append({
            "client_unlearn_ratio": ratio,
            "before_test": metric_mean_std(before_test),
            "after_test": metric_mean_std(after_test),
            "delta_test": metric_mean_std(delta_test),
        })

    summary = {
        "data_dir": args.data_dir,
        "seeds": args.seeds,
        "cuda_type": cuda_type,
        "ratios": ratios,
        "summary": ratio_rows,
    }
    with open(output_root / "fedkd_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== FedKD Robustness Mean ± Std Summary ===")
    for row in ratio_rows:
        print(f"ratio={row['client_unlearn_ratio']:.2f} after_test={row['after_test']['mean']*100:.2f} ± {row['after_test']['std']*100:.2f}")
    print(f"Saved summary to: {output_root / 'fedkd_summary.json'}")


if __name__ == "__main__":
    main()
