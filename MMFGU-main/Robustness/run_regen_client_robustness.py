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

from ReGEnUnlearn.config import Config
from ReGEnUnlearn.server import ReGEnUnlearnServer
from ReGEnUnlearn.utils import set_seed


def parse_ratio_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolve_cuda_type(device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return "cpu"


def build_target_ids(client_order: list[int], num_clients: int, ratio: float) -> list[int]:
    target_count = max(1, min(num_clients - 1, int(round(ratio * num_clients))))
    return client_order[:target_count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReGEnUnlearn client-unlearning robustness sweep")
    parser.add_argument("--data-dir", required=True)
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

    parser.add_argument(
        "--client-unlearn-ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, target_ids: tuple[int, ...], output_dir: str) -> Config:
    return Config(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        target_client_ids=target_ids,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        federated_rounds=args.federated_rounds,
        local_epochs=args.local_epochs,
        eval_interval=args.eval_interval,
        sampling_rate=args.sampling_rate,
        sampler_hidden_dim=args.sampler_hidden_dim,
        sampler_lr=args.sampler_lr,
        sampler_steps=args.sampler_steps,
        sampler_hops=args.sampler_hops,
        overlap_penalty_weight=args.overlap_penalty_weight,
        prompt_token_count=args.prompt_token_count,
        prompt_similarity_threshold=args.prompt_similarity_threshold,
        prompt_message_passing_steps=args.prompt_message_passing_steps,
        prompt_insertion_ratio=args.prompt_insertion_ratio,
        unlearn_epochs=args.unlearn_epochs,
        repair_rounds=args.repair_rounds,
        repair_local_epochs=args.repair_local_epochs,
        affected_threshold=args.affected_threshold,
        lambda_ascend=args.lambda_ascend,
        lambda_reg=args.lambda_reg,
        lambda_retain=args.lambda_retain,
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        run_unlearning=True,
        run_retrain_baseline=False,
    )


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state_dict.items()}


def build_pretrained_snapshot(server: ReGEnUnlearnServer) -> dict[str, object]:
    for client in server.clients:
        client._device_data_cache = {}
    return {
        "global_state": clone_state_dict(server.global_state),
        "history": copy.deepcopy(server.history),
        "experiment_summary": copy.deepcopy(server.experiment_summary),
    }


def restore_server_from_snapshot(
    args: argparse.Namespace, snapshot: dict[str, object], output_dir: str
) -> ReGEnUnlearnServer:
    config = build_config(args, tuple([0]), output_dir)
    config.run_unlearning = False
    server = ReGEnUnlearnServer(config)
    server.global_state = clone_state_dict(snapshot["global_state"])
    server.history = copy.deepcopy(snapshot["history"])
    server.experiment_summary = copy.deepcopy(snapshot["experiment_summary"])
    for client in server.clients:
        client._device_data_cache = {}
    return server


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cuda_type = resolve_cuda_type(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = parse_ratio_list(args.client_unlearn_ratios)

    rng = random.Random(args.seed)
    client_order = list(range(args.num_clients))
    rng.shuffle(client_order)

    print("Running ReGEnUnlearn robustness configuration:")
    print(json.dumps({
        "data_dir": args.data_dir,
        "num_clients": args.num_clients,
        "federated_rounds": args.federated_rounds,
        "local_epochs": args.local_epochs,
        "sampling_rate": args.sampling_rate,
        "sampler_steps": args.sampler_steps,
        "prompt_insertion_ratio": args.prompt_insertion_ratio,
        "affected_threshold": args.affected_threshold,
        "unlearn_epochs": args.unlearn_epochs,
        "repair_rounds": args.repair_rounds,
        "repair_local_epochs": args.repair_local_epochs,
        "lambda_ascend": args.lambda_ascend,
        "lambda_reg": args.lambda_reg,
        "lambda_retain": args.lambda_retain,
        "client_unlearn_ratios": ratios,
        "cuda_type": cuda_type,
        "output_dir": str(output_dir),
    }, indent=2))

    pretrain_config = build_config(args, tuple([0]), str(output_dir))
    pretrain_config.run_unlearning = False
    server = ReGEnUnlearnServer(pretrain_config)
    server.pretrain()
    pretrained_snapshot = build_pretrained_snapshot(server)

    runs = []
    for ratio in ratios:
        target_ids = build_target_ids(client_order, args.num_clients, ratio)
        print(f"\n=== Robustness ratio={ratio:.2f} target_ids={target_ids} ===")
        ratio_server = restore_server_from_snapshot(args, pretrained_snapshot, str(output_dir))
        ratio_server.config.target_client_ids = tuple(target_ids)
        ratio_server.run_unlearning()
        run_payload = copy.deepcopy(ratio_server.experiment_summary["unlearning"])
        run_payload["client_unlearn_ratio"] = ratio
        run_payload["target_client_count"] = len(target_ids)
        runs.append(run_payload)

    summary = {
        "config": {
            "data_dir": args.data_dir,
            "num_clients": args.num_clients,
            "federated_rounds": args.federated_rounds,
            "local_epochs": args.local_epochs,
            "eval_interval": args.eval_interval,
            "sampling_rate": args.sampling_rate,
            "sampler_hidden_dim": args.sampler_hidden_dim,
            "sampler_lr": args.sampler_lr,
            "sampler_steps": args.sampler_steps,
            "sampler_hops": args.sampler_hops,
            "overlap_penalty_weight": args.overlap_penalty_weight,
            "prompt_token_count": args.prompt_token_count,
            "prompt_similarity_threshold": args.prompt_similarity_threshold,
            "prompt_message_passing_steps": args.prompt_message_passing_steps,
            "prompt_insertion_ratio": args.prompt_insertion_ratio,
            "unlearn_epochs": args.unlearn_epochs,
            "repair_rounds": args.repair_rounds,
            "repair_local_epochs": args.repair_local_epochs,
            "affected_threshold": args.affected_threshold,
            "lambda_ascend": args.lambda_ascend,
            "lambda_reg": args.lambda_reg,
            "lambda_retain": args.lambda_retain,
            "seed": args.seed,
            "device": args.device,
        },
        "cuda_type": cuda_type,
        "client_order": client_order,
        "pretrain_final": copy.deepcopy(pretrained_snapshot["experiment_summary"]["pretrain_final"]),
        "runs": runs,
    }
    with open(output_dir / "regen_robustness_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== Robustness Summary ===")
    for row in runs:
        print(f"ratio={row['client_unlearn_ratio']:.2f} after_test={row['after_remaining_metrics']['avg_test_acc']:.4f} targets={row['target_client_count']}")
    print(f"Saved summary to: {output_dir / 'regen_robustness_summary.json'}")


if __name__ == "__main__":
    main()
