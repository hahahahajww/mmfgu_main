import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

#解析命令行参数（数据集、种子、模式、任务等）
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple seeds and summarize mean ± std"
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="One or more dataset directories",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "pretrain",
            "unlearning",
            "retrain",
            "client_unlearning",
            "client_retrain",
        ],
        default="pretrain",
        help="Which experiment entrypoint to run",
    )
    parser.add_argument(
        "--run-unlearning",
        action="store_true",
        help="Run unlearning after pretraining",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="multi_seed_runs",
        help="Root directory for all run outputs",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["node_classification", "link_prediction"],
        default="node_classification",
        help="Task type used by the selected entrypoint",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate every N federated rounds",
    )
    # 添加链接预测相关参数
    parser.add_argument("--prototype-threshold", type=float, default=0.25, help="Prototype similarity threshold")
    parser.add_argument("--purge-rounds", type=int, default=8, help="Number of purge rounds")
    parser.add_argument(
        "--purge-local-epochs",
        type=int,
        default=2,
        help="Number of local epochs in each purge round",
    )
    parser.add_argument("--unlearn-local-epochs", type=int, default=15, help="Local epochs for unlearning")
    parser.add_argument("--lambda-neg", type=float, default=1.0, help="Negative sample loss weight")
    parser.add_argument("--alpha-dec", type=float, default=4.0, help="Relation decay loss weight")
    parser.add_argument("--alpha-anchor", type=float, default=0.2, help="Anchor loss weight")
    parser.add_argument("--beta-mm", type=float, default=0.25, help="Retained node loss weight")
    parser.add_argument("--delta-bd", type=float, default=0.1, help="Boundary node loss weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for link prediction")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--gnn-type", type=str, default="sage", choices=["sage", "gcn", "gat"], help="GNN type")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--federated-rounds", type=int, default=100, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Number of local epochs")
    return parser.parse_args()


def dataset_name(data_dir: str) -> str:
    return Path(data_dir).name

#执行一次实验（调用对应的 .py 文件）
def run_once(
    python_exec: str,
    data_dir: str,
    seed: int,
    output_dir: Path,
    mode: str,
    task: str,
    eval_interval: int,
    prototype_threshold: float,
    purge_rounds: int,
    purge_local_epochs: int,
    unlearn_local_epochs: int,
    lambda_neg: float,
    alpha_dec: float,
    alpha_anchor: float,
    beta_mm: float,
    delta_bd: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    gnn_type: str,
    num_clients: int,
    federated_rounds: int,
    local_epochs: int,
) -> Dict[str, object]:
    entrypoint = {
        "pretrain": "run_multimodal_unlearning.py",
        "unlearning": "run_multimodal_unlearning.py",
        "retrain": "run_multimodal_retrain_baseline.py",
        "client_unlearning": "run_client_unlearning.py",
        "client_retrain": "run_client_retrain_baseline.py",
    }[mode]
    command = [
        python_exec,
        entrypoint,
        "--task",
        task,
        "--data-dir",
        data_dir,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--eval-interval",
        str(eval_interval),
        "--prototype-threshold",
        str(prototype_threshold),
        "--purge-rounds",
        str(purge_rounds),
        "--purge-local-epochs",
        str(purge_local_epochs),
        "--unlearn-local-epochs",
        str(unlearn_local_epochs),
        "--lambda-neg",
        str(lambda_neg),
        "--alpha-dec",
        str(alpha_dec),
        "--alpha-anchor",
        str(alpha_anchor),
        "--beta-mm",
        str(beta_mm),
        "--delta-bd",
        str(delta_bd),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--batch-size",
        str(batch_size),
        "--hidden-dim",
        str(hidden_dim),
        "--dropout",
        str(dropout),
        "--gnn-type",
        gnn_type,
        "--num-clients",
        str(num_clients),
        "--federated-rounds",
        str(federated_rounds),
        "--local-epochs",
        str(local_epochs),
    ]
    if mode == "unlearning":
        command.append("--run-unlearning")

    print(f"\n=== Running {dataset_name(data_dir)} | seed={seed} ===")
    subprocess.run(command, check=True)

    summary_path = output_dir / "experiment_summary.json"
    with summary_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def metric_mean_std(values: List[float]) -> Dict[str, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std}

#把所有种子的结果收集起来，算平均和标准差
def summarize_runs(
    rows: List[Dict[str, object]], mode: str, task: str
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    metric_name = "auc_roc" if task == "link_prediction" else "acc"
    avg_train_key = f"avg_train_{metric_name}"
    avg_val_key = f"avg_val_{metric_name}"
    avg_test_key = f"avg_test_{metric_name}"

    if mode in {"pretrain", "unlearning"}:
        pretrain_test = [row["pretrain_final"][avg_test_key] for row in rows]
        pretrain_val = [row["pretrain_final"][avg_val_key] for row in rows]
        result["pretrain"] = {
            avg_val_key: metric_mean_std(pretrain_val),
            avg_test_key: metric_mean_std(pretrain_test),
        }

    if mode == "unlearning":
        before_test = [
            row["unlearning"]["before_global_metrics"][avg_test_key] for row in rows
        ]
        after_test = [
            row["unlearning"]["after_global_metrics"][avg_test_key] for row in rows
        ]
        delta_test = [
            row["unlearning"]["metric_delta"][f"{avg_test_key}_delta"] for row in rows
        ]
        before_val = [
            row["unlearning"]["before_global_metrics"][avg_val_key] for row in rows
        ]
        after_val = [
            row["unlearning"]["after_global_metrics"][avg_val_key] for row in rows
        ]
        delta_val = [
            row["unlearning"]["metric_delta"][f"{avg_val_key}_delta"] for row in rows
        ]

        result["unlearning"] = {
            f"before_{avg_val_key}": metric_mean_std(before_val),
            f"before_{avg_test_key}": metric_mean_std(before_test),
            f"after_{avg_val_key}": metric_mean_std(after_val),
            f"after_{avg_test_key}": metric_mean_std(after_test),
            f"delta_{avg_val_key}": metric_mean_std(delta_val),
            f"delta_{avg_test_key}": metric_mean_std(delta_test),
        }

    if mode == "retrain":
        retrain_test = [
            row["retrain_baseline"]["final_metrics"][avg_test_key] for row in rows
        ]
        retrain_val = [
            row["retrain_baseline"]["final_metrics"][avg_val_key] for row in rows
        ]
        result["retrain"] = {
            avg_val_key: metric_mean_std(retrain_val),
            avg_test_key: metric_mean_std(retrain_test),
        }

    if mode == "client_unlearning":
        before_test = [
            row["client_unlearning"]["before_global_metrics"][avg_test_key]
            for row in rows
        ]
        after_test = [
            row["client_unlearning"]["after_global_metrics"][avg_test_key]
            for row in rows
        ]
        delta_test = [
            row["client_unlearning"]["metric_delta"][f"{avg_test_key}_delta"]
            for row in rows
        ]
        before_val = [
            row["client_unlearning"]["before_global_metrics"][avg_val_key]
            for row in rows
        ]
        after_val = [
            row["client_unlearning"]["after_global_metrics"][avg_val_key]
            for row in rows
        ]
        delta_val = [
            row["client_unlearning"]["metric_delta"][f"{avg_val_key}_delta"]
            for row in rows
        ]

        result["client_unlearning"] = {
            f"before_{avg_val_key}": metric_mean_std(before_val),
            f"before_{avg_test_key}": metric_mean_std(before_test),
            f"after_{avg_val_key}": metric_mean_std(after_val),
            f"after_{avg_test_key}": metric_mean_std(after_test),
            f"delta_{avg_val_key}": metric_mean_std(delta_val),
            f"delta_{avg_test_key}": metric_mean_std(delta_test),
        }

    if mode == "client_retrain":
        retrain_test = [
            row["client_retrain_baseline"]["final_metrics"][avg_test_key]
            for row in rows
        ]
        retrain_val = [
            row["client_retrain_baseline"]["final_metrics"][avg_val_key] for row in rows
        ]
        result["client_retrain"] = {
            avg_val_key: metric_mean_std(retrain_val),
            avg_test_key: metric_mean_std(retrain_test),
        }

    return result

#把数字变成 88.52 ± 1.23 格式
def format_pm(stats: Dict[str, float]) -> str:
    return f"{stats['mean'] * 100:.2f} ± {stats['std'] * 100:.2f}"


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {}
    for data_dir in args.data_dirs:
        name = dataset_name(data_dir)
        dataset_root = root / name
        dataset_root.mkdir(parents=True, exist_ok=True)

        rows = []
        for seed in args.seeds:
            run_dir = dataset_root / f"seed_{seed}"
            rows.append(
                run_once(
                    python_exec=args.python,
                    data_dir=data_dir,
                    seed=seed,
                    output_dir=run_dir,
                    mode=args.mode,
                    task=args.task,
                    eval_interval=args.eval_interval,
                    prototype_threshold=args.prototype_threshold,
                    purge_rounds=args.purge_rounds,
                    purge_local_epochs=args.purge_local_epochs,
                    unlearn_local_epochs=args.unlearn_local_epochs,
                    lambda_neg=args.lambda_neg,
                    alpha_dec=args.alpha_dec,
                    alpha_anchor=args.alpha_anchor,
                    beta_mm=args.beta_mm,
                    delta_bd=args.delta_bd,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    gnn_type=args.gnn_type,
                    num_clients=args.num_clients,
                    federated_rounds=args.federated_rounds,
                    local_epochs=args.local_epochs,
                )
            )

        summary = summarize_runs(rows, args.mode, args.task)
        all_results[name] = {
            "data_dir": data_dir,
            "task": args.task,
            "seeds": args.seeds,
            "runs": rows,
            "summary": summary,
        }

    summary_path = root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)

    print("\n=== Mean ± Std Summary ===")
    metric_name = "auc_roc" if args.task == "link_prediction" else "acc"
    avg_val_key = f"avg_val_{metric_name}"
    avg_test_key = f"avg_test_{metric_name}"
    for name, payload in all_results.items():
        summary = payload["summary"]
        print(f"\n[{name}]")
        if args.mode in {"pretrain", "unlearning"}:
            print(
                "Pretrain test:",
                format_pm(summary["pretrain"][avg_test_key]),
            )
            print(
                "Pretrain val :",
                format_pm(summary["pretrain"][avg_val_key]),
            )
        if args.mode == "unlearning":
            print(
                "Before test  :",
                format_pm(summary["unlearning"][f"before_{avg_test_key}"]),
            )
            print(
                "After test   :",
                format_pm(summary["unlearning"][f"after_{avg_test_key}"]),
            )
            print(
                "Delta test   :",
                format_pm(summary["unlearning"][f"delta_{avg_test_key}"]),
            )
        if args.mode == "retrain":
            print(
                "Retrain test :",
                format_pm(summary["retrain"][avg_test_key]),
            )
            print(
                "Retrain val  :",
                format_pm(summary["retrain"][avg_val_key]),
            )
        if args.mode == "client_unlearning":
            print(
                "Before test  :",
                format_pm(summary["client_unlearning"][f"before_{avg_test_key}"]),
            )
            print(
                "After test   :",
                format_pm(summary["client_unlearning"][f"after_{avg_test_key}"]),
            )
            print(
                "Delta test   :",
                format_pm(summary["client_unlearning"][f"delta_{avg_test_key}"]),
            )
        if args.mode == "client_retrain":
            print(
                "Retrain test :",
                format_pm(summary["client_retrain"][avg_test_key]),
            )
            print(
                "Retrain val  :",
                format_pm(summary["client_retrain"][avg_val_key]),
            )

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
