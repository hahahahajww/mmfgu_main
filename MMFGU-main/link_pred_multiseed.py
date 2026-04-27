import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


MODE_TO_ENTRYPOINT = {
    "plain": "link_pred.py",
    "modality_unlearning": "run_multimodal_unlearning.py",
    "modality_retrain": "run_multimodal_retrain_baseline.py",
    "client_unlearning": "run_client_unlearning.py",
    "client_retrain": "run_client_retrain_baseline.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run link prediction across multiple seeds")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=list(MODE_TO_ENTRYPOINT), default="plain")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gnn-type", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-neg", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="lp_multiseed_runs")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--unlearn-local-epochs", type=int, default=15)
    parser.add_argument("--purge-rounds", type=int, default=8)
    parser.add_argument("--purge-local-epochs", type=int, default=2)
    parser.add_argument("--forget-client-id", type=int, default=0)
    parser.add_argument("--forget-ratio", type=float, default=0.2)
    parser.add_argument("--prototype-threshold", type=float, default=0.25)
    parser.add_argument("--lambda-neg", type=float, default=1.0)
    parser.add_argument("--probe-count", type=int, default=50)
    parser.add_argument("--probe-topk", type=int, default=10)
    parser.add_argument("--alpha-dec", type=float, default=4.0)
    parser.add_argument("--alpha-anchor", type=float, default=0.2)
    parser.add_argument("--beta-mm", type=float, default=0.25)
    parser.add_argument("--delta-bd", type=float, default=0.1)
    args = parser.parse_args()

    argv = set(sys.argv[1:])
    # 取消对 Sports 数据集的特殊处理，所有数据集使用统一参数
    # dataset_name = Path(args.data_dir).name.lower()
    # if dataset_name == "sports" and args.mode != "plain":
    #     if "--num-clients" not in argv:
    #         args.num_clients = 10
    #     if "--federated-rounds" not in argv:
    #         args.federated_rounds = 100
    #     if "--local-epochs" not in argv:
    #         args.local_epochs = 5
    #     if "--eval-interval" not in argv:
    #         args.eval_interval = 10
    #     if "--batch-size" not in argv:
    #         args.batch_size = 16384
    #     if "--hidden-dim" not in argv:
    #         args.hidden_dim = 512
    #     if "--dropout" not in argv:
    #         args.dropout = 0.05
    #     if "--lr" not in argv:
    #         args.lr = 1e-3
    #     if "--weight-decay" not in argv:
    #         args.weight_decay = 5e-5
    #     if "--gnn-type" not in argv:
    #         args.gnn_type = "sage"

    return args


def metric_mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build_command(args: argparse.Namespace, seed: int, output_dir: Path) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    entrypoint = script_dir / MODE_TO_ENTRYPOINT[args.mode]
    command = [args.python, str(entrypoint)]

    if args.mode == "plain":
        command.extend(
            [
                "--data-dir",
                args.data_dir,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
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
        "--num-neg",
        str(args.num_neg),
        "--device",
        args.device,
                "--output-dir",
                str(output_dir),
            ]
        )
        return command

    command.extend(
        [
            "--task",
            "link_prediction",
            "--data-dir",
            args.data_dir,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--num-clients",
            str(args.num_clients),
            "--federated-rounds",
            str(args.federated_rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--eval-interval",
            str(args.eval_interval),
            "--unlearn-local-epochs",
            str(args.unlearn_local_epochs),
            "--purge-rounds",
            str(args.purge_rounds),
            "--purge-local-epochs",
            str(args.purge_local_epochs),
            "--forget-client-id",
            str(args.forget_client_id),
            "--forget-ratio",
            str(args.forget_ratio),
            "--prototype-threshold",
            str(args.prototype_threshold),
            "--lambda-neg",
            str(args.lambda_neg),
            "--probe-count",
            str(args.probe_count),
            "--probe-topk",
            str(args.probe_topk),
            "--batch-size",
            str(args.batch_size),
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
            "--device",
            args.device,
            "--alpha-dec",
            str(args.alpha_dec),
            "--alpha-anchor",
            str(args.alpha_anchor),
            "--beta-mm",
            str(args.beta_mm),
            "--delta-bd",
            str(args.delta_bd),
        ]
    )
    if args.mode == "modality_unlearning":
        command.append("--run-unlearning")
    return command


def run_once(args: argparse.Namespace, seed: int, output_dir: Path) -> dict:
    script_dir = Path(__file__).resolve().parent
    command = build_command(args, seed, output_dir)
    print(f"\n=== Running mode={args.mode} seed={seed} ===")
    subprocess.run(command, check=True, cwd=script_dir)

    summary_name = "lp_summary.json" if args.mode == "plain" else "experiment_summary.json"
    return json.loads((output_dir / summary_name).read_text(encoding="utf-8"))


def summarize_plain(rows: list[dict]) -> dict:
    result = {}
    for split in ["best_valid", "test"]:
        result[split] = {}
        for metric in ["mrr", "hits@3", "hits@10", "auc_roc"]:
            result[split][metric] = metric_mean_std([row[split][metric] for row in rows])
    return result


def summarize_federated(rows: list[dict], mode: str) -> dict:
    result = {}
    tracked_metrics = ["mrr", "hits@3", "auc_roc"]

    def add_metric_group(target: dict, rows_to_read: list[dict], key_prefix: str) -> None:
        for metric in tracked_metrics:
            val_key = f"avg_val_{metric}"
            test_key = f"avg_test_{metric}"
            out_val_key = f"{key_prefix}_{val_key}" if key_prefix else val_key
            out_test_key = f"{key_prefix}_{test_key}" if key_prefix else test_key
            if rows_to_read and val_key in rows_to_read[0]:
                target[out_val_key] = metric_mean_std(
                    [row[val_key] for row in rows_to_read]
                )
            if rows_to_read and test_key in rows_to_read[0]:
                target[out_test_key] = metric_mean_std(
                    [row[test_key] for row in rows_to_read]
                )

    def add_metric_delta_group(target: dict, rows_to_read: list[dict]) -> None:
        for metric in tracked_metrics:
            val_key = f"avg_val_{metric}_delta"
            test_key = f"avg_test_{metric}_delta"
            if rows_to_read and val_key in rows_to_read[0]:
                target[f"delta_avg_val_{metric}"] = metric_mean_std(
                    [row[val_key] for row in rows_to_read]
                )
            if rows_to_read and test_key in rows_to_read[0]:
                target[f"delta_avg_test_{metric}"] = metric_mean_std(
                    [row[test_key] for row in rows_to_read]
                )

    if mode == "modality_unlearning":
        pretrain_rows = [row["pretrain_final"] for row in rows]
        before_rows = [row["unlearning"]["before_global_metrics"] for row in rows]
        after_rows = [row["unlearning"]["after_global_metrics"] for row in rows]
        delta_rows = [row["unlearning"]["metric_delta"] for row in rows]
        result["pretrain"] = {}
        result["modality_unlearning"] = {}
        add_metric_group(result["pretrain"], pretrain_rows, "")
        add_metric_group(result["modality_unlearning"], before_rows, "before")
        add_metric_group(result["modality_unlearning"], after_rows, "after")
        add_metric_delta_group(result["modality_unlearning"], delta_rows)
    elif mode == "modality_retrain":
        retrain_rows = [row["retrain_baseline"]["final_metrics"] for row in rows]
        result["modality_retrain"] = {}
        add_metric_group(result["modality_retrain"], retrain_rows, "")
    elif mode == "client_unlearning":
        before_rows = [row["client_unlearning"]["before_global_metrics"] for row in rows]
        after_rows = [row["client_unlearning"]["after_global_metrics"] for row in rows]
        delta_rows = [row["client_unlearning"]["metric_delta"] for row in rows]
        result["client_unlearning"] = {}
        add_metric_group(result["client_unlearning"], before_rows, "before")
        add_metric_group(result["client_unlearning"], after_rows, "after")
        add_metric_delta_group(result["client_unlearning"], delta_rows)
    elif mode == "client_retrain":
        retrain_rows = [row["client_retrain_baseline"]["final_metrics"] for row in rows]
        result["client_retrain"] = {}
        add_metric_group(result["client_retrain"], retrain_rows, "")

    return result


def print_plain_summary(summary: dict) -> None:
    for split in ["best_valid", "test"]:
        print(f"[{split}]")
        for metric in ["mrr", "hits@3", "hits@10", "auc_roc"]:
            stats = summary[split][metric]
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def print_federated_summary(summary: dict, mode: str) -> None:
    def print_metric_block(target: dict, prefix: str) -> None:
        for metric in ["mrr", "hits@3", "auc_roc"]:
            val_key = f"{prefix}avg_val_{metric}"
            test_key = f"{prefix}avg_test_{metric}"
            if val_key in target:
                print(f"{val_key}: {target[val_key]['mean']:.4f} ± {target[val_key]['std']:.4f}")
            if test_key in target:
                print(f"{test_key}: {target[test_key]['mean']:.4f} ± {target[test_key]['std']:.4f}")

    if mode == "modality_unlearning":
        print_metric_block(summary["pretrain"], "")
        target = summary[mode]
        print_metric_block(target, "before_")
        print_metric_block(target, "after_")
        print_metric_block(target, "delta_")
    elif mode in {"modality_retrain", "client_retrain"}:
        print_metric_block(summary[mode], "")
    elif mode == "client_unlearning":
        target = summary[mode]
        print_metric_block(target, "before_")
        print_metric_block(target, "after_")
        print_metric_block(target, "delta_")


def main() -> None:
    args = parse_args()
    dataset_name = Path(args.data_dir).name
    output_root = Path(args.output_root) / args.mode / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows.append(run_once(args, seed, run_dir))

    summary = summarize_plain(rows) if args.mode == "plain" else summarize_federated(rows, args.mode)
    payload = {
        "mode": args.mode,
        "data_dir": args.data_dir,
        "seeds": args.seeds,
        "runs": rows,
        "summary": summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Mean ± Std Summary ===")
    print_plain_summary(summary) if args.mode == "plain" else print_federated_summary(summary, args.mode)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
