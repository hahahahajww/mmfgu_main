from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


MODE_TO_ENTRYPOINT = {
    "unlearning": "MoDe.run_mode",
    "retrain": "MoDe.run_mode_retrain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoDe across multiple seeds")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=list(MODE_TO_ENTRYPOINT), default="unlearning")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="E:\\MMFGU\\MoDe\\outputs_multiseed")

    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--forget-client-id", type=int, default=0)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=5)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn-type", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--momentum-coeff", type=float, default=0.95)
    parser.add_argument("--degradation-rounds", type=int, default=5)
    parser.add_argument("--guidance-rounds", type=int, default=8)
    return parser.parse_args()


def metric_mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def format_pm(stats: dict[str, float]) -> str:
    return f"{stats['mean'] * 100:.2f} ± {stats['std'] * 100:.2f}"


def build_command(args: argparse.Namespace, seed: int, output_dir: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        MODE_TO_ENTRYPOINT[args.mode],
        "--data-dir",
        args.data_dir,
        "--task",
        "node_classification",
        "--num-clients",
        str(args.num_clients),
        "--forget-client-id",
        str(args.forget_client_id),
        "--device",
        args.device,
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
        "--momentum-coeff",
        str(args.momentum_coeff),
        "--degradation-rounds",
        str(args.degradation_rounds),
        "--guidance-rounds",
        str(args.guidance_rounds),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.mode == "unlearning":
        command.append("--run-unlearning")
    return command


def run_once(args: argparse.Namespace, seed: int, output_dir: Path) -> dict:
    script_dir = Path(__file__).resolve().parent
    command = build_command(args, seed, output_dir)
    print(f"\n=== Running mode={args.mode} seed={seed} ===")
    subprocess.run(command, check=True, cwd=script_dir.parent)
    return json.loads((output_dir / "experiment_summary.json").read_text(encoding="utf-8"))


def summarize_runs(rows: list[dict], mode: str) -> dict:
    avg_val_key = "avg_val_acc"
    avg_test_key = "avg_test_acc"
    result: dict[str, object] = {}

    if mode == "unlearning":
        before_val = [row["client_unlearning"]["before_global_metrics"][avg_val_key] for row in rows]
        before_test = [row["client_unlearning"]["before_global_metrics"][avg_test_key] for row in rows]
        after_val = [row["client_unlearning"]["after_global_metrics"][avg_val_key] for row in rows]
        after_test = [row["client_unlearning"]["after_global_metrics"][avg_test_key] for row in rows]
        delta_val = [row["client_unlearning"]["metric_delta"][f"{avg_val_key}_delta"] for row in rows]
        delta_test = [row["client_unlearning"]["metric_delta"][f"{avg_test_key}_delta"] for row in rows]
        result["client_unlearning"] = {
            f"before_{avg_val_key}": metric_mean_std(before_val),
            f"before_{avg_test_key}": metric_mean_std(before_test),
            f"after_{avg_val_key}": metric_mean_std(after_val),
            f"after_{avg_test_key}": metric_mean_std(after_test),
            f"delta_{avg_val_key}": metric_mean_std(delta_val),
            f"delta_{avg_test_key}": metric_mean_std(delta_test),
        }
    else:
        retrain_val = [row["client_retrain_baseline"]["final_metrics"][avg_val_key] for row in rows]
        retrain_test = [row["client_retrain_baseline"]["final_metrics"][avg_test_key] for row in rows]
        result["client_retrain"] = {
            avg_val_key: metric_mean_std(retrain_val),
            avg_test_key: metric_mean_std(retrain_test),
        }
    return result


def main() -> None:
    args = parse_args()
    dataset_name = Path(args.data_dir).name
    root_dir = Path(args.output_root) / dataset_name / args.mode
    root_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        output_dir = root_dir / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows.append(run_once(args, seed, output_dir))

    aggregated = {
        "data_dir": args.data_dir,
        "mode": args.mode,
        "seeds": args.seeds,
        "summary": summarize_runs(rows, args.mode),
    }
    pretty_summary: dict[str, dict[str, str]] = {}
    for group_name, group_values in aggregated["summary"].items():
        pretty_summary[group_name] = {
            metric_name: format_pm(metric_stats)
            for metric_name, metric_stats in group_values.items()
        }
    aggregated["summary_pretty"] = pretty_summary

    summary_path = root_dir / "summary.json"
    summary_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    print("\n=== Mean ± Std Summary ===")
    for group_name, group_values in pretty_summary.items():
        print(f"[{group_name}]")
        for metric_name, metric_value in group_values.items():
            print(f"{metric_name}: {metric_value}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
