import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedEraser across multiple seeds")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--task", choices=["node_classification", "link_prediction"], default="node_classification")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--forget-client-id", type=int, default=0)
    parser.add_argument("--retain-interval", type=int, default=2)
    parser.add_argument("--calibration-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "FedEraser" / "runs_multiseed"))
    return parser.parse_args()


def metric_mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_once(args: argparse.Namespace, seed: int, output_dir: Path) -> dict:
    command = [
        args.python,
        str(PROJECT_ROOT / "FedEraser" / "run_fed_eraser.py"),
        "--data-dir",
        args.data_dir,
        "--task",
        args.task,
        "--seed",
        str(seed),
        "--num-clients",
        str(args.num_clients),
        "--federated-rounds",
        str(args.federated_rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--forget-client-id",
        str(args.forget_client_id),
        "--retain-interval",
        str(args.retain_interval),
        "--calibration-ratio",
        str(args.calibration_ratio),
        "--device",
        args.device,
        "--eval-interval",
        str(args.eval_interval),
        "--output-dir",
        str(output_dir),
    ]
    print(f"\n=== Running {Path(args.data_dir).name} | seed={seed} ===")
    subprocess.run(command, check=True)
    return json.loads((output_dir / "experiment_summary.json").read_text(encoding="utf-8"))


def summarize(rows: list[dict], task: str) -> dict[str, object]:
    metric_name = "auc_roc" if task == "link_prediction" else "acc"
    before_key = f"avg_test_{metric_name}"
    after_key = f"avg_test_{metric_name}"
    delta_key = f"avg_test_{metric_name}_delta"
    fed_rows = [row["fed_eraser"] for row in rows]
    return {
        "before_test": metric_mean_std([row["before_global_metrics"][before_key] for row in fed_rows]),
        "after_test": metric_mean_std([row["after_global_metrics"][after_key] for row in fed_rows]),
        "delta_test": metric_mean_std([row["metric_delta"][delta_key] for row in fed_rows]),
        "reconstruction_seconds": metric_mean_std([row["reconstruction_seconds"] for row in fed_rows]),
    }


def format_pm(stats: dict[str, float]) -> str:
    return f"{stats['mean'] * 100:.2f} ± {stats['std'] * 100:.2f}"


def main() -> None:
    args = parse_args()
    dataset_name = Path(args.data_dir).name
    output_root = Path(args.output_root) / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows.append(run_once(args, seed, run_dir))

    summary = {
        "data_dir": args.data_dir,
        "task": args.task,
        "seeds": args.seeds,
        "retain_interval": args.retain_interval,
        "calibration_ratio": args.calibration_ratio,
        "runs": rows,
        "summary": summarize(rows, args.task),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Mean ± Std Summary ===")
    print("Before test:", format_pm(summary["summary"]["before_test"]))
    print("After test :", format_pm(summary["summary"]["after_test"]))
    print("Delta test :", format_pm(summary["summary"]["delta_test"]))
    secs = summary["summary"]["reconstruction_seconds"]
    print(f"Reconstruct: {secs['mean']:.2f} ± {secs['std']:.2f} seconds")
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
