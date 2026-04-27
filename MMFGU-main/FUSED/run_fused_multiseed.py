import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FUSED across multiple seeds")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--task",
        choices=["node_classification", "link_prediction"],
        default="node_classification",
    )
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn-type", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--forget-client-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "FUSED" / "runs_multiseed"),
    )
    parser.add_argument("--cli-local-epochs", type=int, default=1)
    parser.add_argument("--cli-topk-layers", type=int, default=4)
    parser.add_argument("--fused-rounds", type=int, default=5)
    parser.add_argument("--fused-local-epochs", type=int, default=2)
    parser.add_argument("--adapter-density", type=float, default=0.05)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    return parser.parse_args()


def metric_mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def format_pm(stats: dict[str, float], scale: float = 100.0) -> str:
    return f"{stats['mean'] * scale:.2f} ± {stats['std'] * scale:.2f}"


def run_once(args: argparse.Namespace, seed: int, output_dir: Path) -> dict:
    command = [
        args.python,
        str(PROJECT_ROOT / "FUSED" / "run_fused.py"),
        "--data-dir",
        args.data_dir,
        "--task",
        args.task,
        "--seed",
        str(seed),
        "--num-clients",
        str(args.num_clients),
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
        "--batch-size",
        str(args.batch_size),
        "--federated-rounds",
        str(args.federated_rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--eval-interval",
        str(args.eval_interval),
        "--forget-client-id",
        str(args.forget_client_id),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--cli-local-epochs",
        str(args.cli_local_epochs),
        "--cli-topk-layers",
        str(args.cli_topk_layers),
        "--fused-rounds",
        str(args.fused_rounds),
        "--fused-local-epochs",
        str(args.fused_local_epochs),
        "--adapter-density",
        str(args.adapter_density),
        "--adapter-lr",
        str(args.adapter_lr),
    ]
    print(f"\n=== Running {Path(args.data_dir).name} | seed={seed} ===")
    subprocess.run(command, check=True)
    return json.loads((output_dir / "experiment_summary.json").read_text(encoding="utf-8"))


def summarize(rows: list[dict], task: str) -> dict[str, object]:
    metric_name = "auc_roc" if task == "link_prediction" else "acc"
    avg_val_key = f"avg_val_{metric_name}"
    avg_test_key = f"avg_test_{metric_name}"
    delta_val_key = f"{avg_val_key}_delta"
    delta_test_key = f"{avg_test_key}_delta"

    fused_rows = [row["fused"] for row in rows]
    return {
        "before_val": metric_mean_std(
            [row["before_global_metrics"][avg_val_key] for row in fused_rows]
        ),
        "before_test": metric_mean_std(
            [row["before_global_metrics"][avg_test_key] for row in fused_rows]
        ),
        "after_val": metric_mean_std(
            [row["after_global_metrics"][avg_val_key] for row in fused_rows]
        ),
        "after_test": metric_mean_std(
            [row["after_global_metrics"][avg_test_key] for row in fused_rows]
        ),
        "delta_val": metric_mean_std(
            [row["metric_delta"][delta_val_key] for row in fused_rows]
        ),
        "delta_test": metric_mean_std(
            [row["metric_delta"][delta_test_key] for row in fused_rows]
        ),
        "forgotten_after_test": metric_mean_std(
            [row["forgotten_client_metrics_after"][f"test_{metric_name}"] for row in fused_rows]
        ),
        "compression_ratio": metric_mean_std(
            [row["adapter"]["compression_ratio_vs_full_model"] for row in fused_rows]
        ),
        "active_adapter_count": metric_mean_std(
            [float(row["adapter"]["active_adapter_count"]) for row in fused_rows]
        ),
    }


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
        "config": {
            "num_clients": args.num_clients,
            "federated_rounds": args.federated_rounds,
            "local_epochs": args.local_epochs,
            "forget_client_id": args.forget_client_id,
            "cli_local_epochs": args.cli_local_epochs,
            "cli_topk_layers": args.cli_topk_layers,
            "fused_rounds": args.fused_rounds,
            "fused_local_epochs": args.fused_local_epochs,
            "adapter_density": args.adapter_density,
            "adapter_lr": args.adapter_lr,
        },
        "runs": rows,
        "summary": summarize(rows, args.task),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Mean ± Std Summary ===")
    print("Before test      :", format_pm(summary["summary"]["before_test"]))
    print("After test       :", format_pm(summary["summary"]["after_test"]))
    print("Delta test       :", format_pm(summary["summary"]["delta_test"]))
    print("Forgotten test   :", format_pm(summary["summary"]["forgotten_after_test"]))
    print(
        "Compression ratio:",
        format_pm(summary["summary"]["compression_ratio"]),
    )
    active = summary["summary"]["active_adapter_count"]
    print(f"Active adapters  : {active['mean']:.2f} ± {active['std']:.2f}")
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
