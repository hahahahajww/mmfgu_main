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

from FedKD.config import Config
from Robustness.fedkd_runner import RobustnessFedKDServer
from mmfgu.utils import set_seed


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
    parser = argparse.ArgumentParser(description="Run FedKD client-unlearning robustness sweep")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", choices=["node_classification"], default="node_classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
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


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=args.data_dir,
        task=args.task,
        num_clients=args.num_clients,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        federated_rounds=args.federated_rounds,
        local_epochs=args.local_epochs,
        eval_interval=args.eval_interval,
        forget_client_id=0,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        distill_epochs=args.distill_epochs,
        distill_lr=args.distill_lr,
        distill_temperature=args.distill_temperature,
    )


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state_dict.items()}


def build_pretrained_snapshot(server: RobustnessFedKDServer) -> dict[str, object]:
    for client in server.clients:
        client._device_data_cache = {}
    return {
        "global_state": clone_state_dict(server.global_state),
        "history": copy.deepcopy(server.history),
        "pretrain_final": copy.deepcopy(server.experiment_summary["pretrain_final"]),
        "target_deltas": [clone_state_dict(delta) for delta in server.target_deltas],
        "all_client_deltas": [
            [clone_state_dict(delta) for delta in round_deltas]
            for round_deltas in server.all_client_deltas
        ],
    }


def restore_server_from_snapshot(config: Config, snapshot: dict[str, object]) -> RobustnessFedKDServer:
    server = RobustnessFedKDServer(config)
    server.global_state = clone_state_dict(snapshot["global_state"])
    server.history = copy.deepcopy(snapshot["history"])
    server.experiment_summary["pretrain_final"] = copy.deepcopy(snapshot["pretrain_final"])
    server.target_deltas = [clone_state_dict(delta) for delta in snapshot["target_deltas"]]
    server.all_client_deltas = [
        [clone_state_dict(delta) for delta in round_deltas]
        for round_deltas in snapshot["all_client_deltas"]
    ]
    for client in server.clients:
        client._device_data_cache = {}
    return server


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)
    cuda_type = resolve_cuda_type(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = parse_ratio_list(args.client_unlearn_ratios)

    print("Running FedKD robustness configuration:")
    print(json.dumps({
        "data_dir": config.data_dir,
        "num_clients": config.num_clients,
        "federated_rounds": config.federated_rounds,
        "local_epochs": config.local_epochs,
        "distill_epochs": config.distill_epochs,
        "distill_lr": config.distill_lr,
        "distill_temperature": config.distill_temperature,
        "client_unlearn_ratios": ratios,
        "cuda_type": cuda_type,
        "output_dir": str(output_dir),
    }, indent=2))

    server = RobustnessFedKDServer(config)
    server.pretrain()
    pretrained_snapshot = build_pretrained_snapshot(server)

    rng = random.Random(config.seed)
    client_order = list(range(config.num_clients))
    rng.shuffle(client_order)

    runs = []
    for ratio in ratios:
        target_count = max(1, min(config.num_clients - 1, int(round(ratio * config.num_clients))))
        target_ids = client_order[:target_count]
        print(f"\n=== Robustness ratio={ratio:.2f} target_ids={target_ids} ===")
        ratio_server = restore_server_from_snapshot(config, pretrained_snapshot)
        result = ratio_server.run_client_unlearning_for_clients(target_ids)
        result["client_unlearn_ratio"] = ratio
        runs.append(result)

    summary = {
        "config": config.__dict__,
        "cuda_type": cuda_type,
        "client_order": client_order,
        "pretrain_final": copy.deepcopy(pretrained_snapshot["pretrain_final"]),
        "runs": runs,
    }
    with open(output_dir / "fedkd_robustness_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== Robustness Summary ===")
    for row in runs:
        print(f"ratio={row['client_unlearn_ratio']:.2f} after_test={row['after_global_metrics']['avg_test_acc']:.4f} targets={row['target_client_count']}")
    print(f"Saved summary to: {output_dir / 'fedkd_robustness_summary.json'}")


if __name__ == "__main__":
    main()
