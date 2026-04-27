from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


def _parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


@dataclass
class Config:
    data_dir: str = r"E:\MMFGU\datasets\Toys"
    num_clients: int = 10
    target_client_ids: tuple[int, ...] = (0,)

    hidden_dim: int = 256
    dropout: float = 0.2
    gnn_type: str = "sage"

    lr: float = 1e-3
    weight_decay: float = 1e-4

    federated_rounds: int = 80
    local_epochs: int = 3
    eval_interval: int = 5

    sampling_rate: float = 0.25
    sampler_hidden_dim: int = 128
    sampler_lr: float = 5e-4
    sampler_steps: int = 20
    sampler_hops: int = 1
    overlap_penalty_weight: float = 0.2

    prompt_token_count: int = 16
    prompt_similarity_threshold: float = 0.55
    prompt_message_passing_steps: int = 2
    prompt_insertion_ratio: float = 0.2

    unlearn_epochs: int = 12
    repair_rounds: int = 4
    repair_local_epochs: int = 2
    affected_threshold: float = 0.25

    lambda_ascend: float = 1.0
    lambda_reg: float = 0.02
    lambda_retain: float = 0.5

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "ReGEnUnlearn/outputs"
    run_unlearning: bool = False
    run_retrain_baseline: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="ReGEnUnlearn for multimodal federated graph unlearning"
    )
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--num-clients", type=int, default=Config.num_clients)
    parser.add_argument(
        "--target-client-ids",
        type=str,
        default=",".join(map(str, Config.target_client_ids)),
        help="comma-separated client ids to unlearn",
    )
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument(
        "--gnn-type", choices=["sage", "gcn", "gat"], default=Config.gnn_type
    )
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument(
        "--federated-rounds", type=int, default=Config.federated_rounds
    )
    parser.add_argument("--local-epochs", type=int, default=Config.local_epochs)
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--sampling-rate", type=float, default=Config.sampling_rate)
    parser.add_argument(
        "--sampler-hidden-dim", type=int, default=Config.sampler_hidden_dim
    )
    parser.add_argument("--sampler-lr", type=float, default=Config.sampler_lr)
    parser.add_argument("--sampler-steps", type=int, default=Config.sampler_steps)
    parser.add_argument("--sampler-hops", type=int, default=Config.sampler_hops)
    parser.add_argument(
        "--overlap-penalty-weight",
        type=float,
        default=Config.overlap_penalty_weight,
    )
    parser.add_argument(
        "--prompt-token-count", type=int, default=Config.prompt_token_count
    )
    parser.add_argument(
        "--prompt-similarity-threshold",
        type=float,
        default=Config.prompt_similarity_threshold,
    )
    parser.add_argument(
        "--prompt-message-passing-steps",
        type=int,
        default=Config.prompt_message_passing_steps,
    )
    parser.add_argument(
        "--prompt-insertion-ratio",
        type=float,
        default=Config.prompt_insertion_ratio,
    )
    parser.add_argument("--unlearn-epochs", type=int, default=Config.unlearn_epochs)
    parser.add_argument("--repair-rounds", type=int, default=Config.repair_rounds)
    parser.add_argument(
        "--repair-local-epochs", type=int, default=Config.repair_local_epochs
    )
    parser.add_argument(
        "--affected-threshold", type=float, default=Config.affected_threshold
    )
    parser.add_argument("--lambda-ascend", type=float, default=Config.lambda_ascend)
    parser.add_argument("--lambda-reg", type=float, default=Config.lambda_reg)
    parser.add_argument("--lambda-retain", type=float, default=Config.lambda_retain)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--run-unlearning", action="store_true")
    parser.add_argument("--run-retrain-baseline", action="store_true")
    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        target_client_ids=tuple(_parse_int_list(args.target_client_ids)),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        federated_rounds=args.federated_rounds,
        local_epochs=args.local_epochs,
        eval_interval=max(1, args.eval_interval),
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
        output_dir=args.output_dir,
        run_unlearning=args.run_unlearning,
        run_retrain_baseline=args.run_retrain_baseline,
    )
