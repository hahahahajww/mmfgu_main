from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


MODE_TO_ENTRYPOINT = {
    "unlearning": "ReGEnUnlearn.run_regenunlearn",
    "retrain": "ReGEnUnlearn.run_regenunlearn_retrain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReGEnUnlearn across multiple seeds"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--mode", choices=list(MODE_TO_ENTRYPOINT), default="unlearning"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="E:\\MMFGU\\ReGEnUnlearn\\outputs_multiseed")

    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--target-client-ids", type=str, default="0")
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=5)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--gnn-type", choices=["sage", "gcn", "gat"], default="sage"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--sampling-rate", type=float, default=0.25)
    parser.add_argument("--sampler-hidden-dim", type=int, default=128)
    parser.add_argument("--sampler-lr", type=float, default=5e-4)
    parser.add_argument("--sampler-steps", type=int, default=20)
    parser.add_argument("--sampler-hops", type=int, default=1)
    parser.add_argument("--overlap-penalty-weight", type=float, default=0.2)

    parser.add_argument("--prompt-token-count", type=int, default=16)
    parser.add_argument("--prompt-similarity-threshold", type=float, default=0.55)
    parser.add_argument("--prompt-message-passing-steps", type=int, default=2)
    parser.add_argument("--prompt-insertion-ratio", type=float, default=0.2)

    parser.add_argument("--unlearn-epochs", type=int, default=12)
    parser.add_argument("--repair-rounds", type=int, default=4)
    parser.add_argument("--repair-local-epochs", type=int, default=2)
    parser.add_argument("--affected-threshold", type=float, default=0.25)

    parser.add_argument("--lambda-ascend", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.02)
    parser.add_argument("--lambda-retain", type=float, default=0.5)
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
        "--num-clients",
        str(args.num_clients),
        "--target-client-ids",
        args.target_client_ids,
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
        "--sampling-rate",
        str(args.sampling_rate),
        "--sampler-hidden-dim",
        str(args.sampler_hidden_dim),
        "--sampler-lr",
        str(args.sampler_lr),
        "--sampler-steps",
        str(args.sampler_steps),
        "--sampler-hops",
        str(args.sampler_hops),
        "--overlap-penalty-weight",
        str(args.overlap_penalty_weight),
        "--prompt-token-count",
        str(args.prompt_token_count),
        "--prompt-similarity-threshold",
        str(args.prompt_similarity_threshold),
        "--prompt-message-passing-steps",
        str(args.prompt_message_passing_steps),
        "--prompt-insertion-ratio",
        str(args.prompt_insertion_ratio),
        "--unlearn-epochs",
        str(args.unlearn_epochs),
        "--repair-rounds",
        str(args.repair_rounds),
        "--repair-local-epochs",
        str(args.repair_local_epochs),
        "--affected-threshold",
        str(args.affected_threshold),
        "--lambda-ascend",
        str(args.lambda_ascend),
        "--lambda-reg",
        str(args.lambda_reg),
        "--lambda-retain",
        str(args.lambda_retain),
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
        pretrain_val = [row["pretrain_final"][avg_val_key] for row in rows]
        pretrain_test = [row["pretrain_final"][avg_test_key] for row in rows]
        after_val = [row["unlearning"]["after_remaining_metrics"][avg_val_key] for row in rows]
        after_test = [row["unlearning"]["after_remaining_metrics"][avg_test_key] for row in rows]
        result["pretrain"] = {
            avg_val_key: metric_mean_std(pretrain_val),
            avg_test_key: metric_mean_std(pretrain_test),
        }
        result["unlearning"] = {
            f"after_{avg_val_key}": metric_mean_std(after_val),
            f"after_{avg_test_key}": metric_mean_std(after_test),
        }
    else:
        retrain_val = [row["retrain_baseline"]["final_metrics"][avg_val_key] for row in rows]
        retrain_test = [row["retrain_baseline"]["final_metrics"][avg_test_key] for row in rows]
        result["retrain"] = {
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
        summary = run_once(args, seed, output_dir)
        rows.append(summary)

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
