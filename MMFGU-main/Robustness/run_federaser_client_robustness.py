from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmfgu.config import Config
from mmfgu.utils import set_seed

from Robustness.fed_eraser_runner import RobustnessFedEraserRunner


def parse_ratio_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolve_cuda_type(device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedEraser client-unlearning robustness sweep"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", choices=["node_classification"], default="node_classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--federated-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--retain-interval", type=int, default=2)
    parser.add_argument("--calibration-ratio", type=float, default=0.5)

    parser.add_argument(
        "--client-unlearn-ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=args.data_dir,
        task=args.task,
        num_clients=args.num_clients,
        federated_rounds=args.federated_rounds,
        local_epochs=args.local_epochs,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)
    cuda_type = resolve_cuda_type(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = parse_ratio_list(args.client_unlearn_ratios)

    print("Running FedEraser robustness configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "retain_interval": args.retain_interval,
                "calibration_ratio": args.calibration_ratio,
                "client_unlearn_ratios": ratios,
                "cuda_type": cuda_type,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )

    runner = RobustnessFedEraserRunner(
        config=config,
        retain_interval=args.retain_interval,
        calibration_ratio=args.calibration_ratio,
    )
    runner.pretrain_with_retention()
    pretrained_runner = copy.deepcopy(runner)

    rng = random.Random(config.seed)
    client_order = list(range(config.num_clients))
    rng.shuffle(client_order)

    runs = []
    for ratio in ratios:
        target_count = max(1, min(config.num_clients - 1, int(round(ratio * config.num_clients))))
        target_ids = client_order[:target_count]
        print(f"\n=== Robustness ratio={ratio:.2f} target_ids={target_ids} ===")
        ratio_runner = copy.deepcopy(pretrained_runner)
        result = ratio_runner.reconstruct_without_clients(target_ids)
        result["client_unlearn_ratio"] = ratio
        runs.append(result)

    summary = {
        "config": {
            "data_dir": config.data_dir,
            "task": config.task,
            "num_clients": config.num_clients,
            "federated_rounds": config.federated_rounds,
            "local_epochs": config.local_epochs,
            "eval_interval": config.eval_interval,
            "device": config.device,
            "seed": config.seed,
            "retain_interval": args.retain_interval,
            "calibration_ratio": args.calibration_ratio,
        },
        "cuda_type": cuda_type,
        "client_order": client_order,
        "pretrain_final": pretrained_runner.summary["pretrain_final"],
        "runs": runs,
    }
    with open(output_dir / "fed_eraser_robustness_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== Robustness Summary ===")
    for row in runs:
        print(
            f"ratio={row['client_unlearn_ratio']:.2f} after_test={row['after_global_metrics']['avg_test_acc']:.4f} targets={row['target_client_count']}"
        )
    print(f"Saved summary to: {output_dir / 'fed_eraser_robustness_summary.json'}")


if __name__ == "__main__":
    main()
