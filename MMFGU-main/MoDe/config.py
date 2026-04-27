from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass
class Config:
    data_dir: str = r"E:\MMFGU\datasets\Toys"
    task: str = "node_classification"
    num_clients: int = 10

    hidden_dim: int = 256
    dropout: float = 0.2
    gnn_type: str = "sage"

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096

    federated_rounds: int = 100
    local_epochs: int = 2
    eval_interval: int = 5

    forget_client_id: int = 0
    forget_ratio: float = 0.2

    momentum_coeff: float = 0.95
    degradation_rounds: int = 5
    guidance_rounds: int = 8

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "MoDe/outputs"

    run_unlearning: bool = False
    run_retrain_baseline: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="MoDe baseline for federated client unlearning"
    )
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument(
        "--task",
        type=str,
        default=Config.task,
        choices=["node_classification"],
    )
    parser.add_argument("--num-clients", type=int, default=Config.num_clients)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument(
        "--gnn-type",
        type=str,
        default=Config.gnn_type,
        choices=["sage", "gcn", "gat"],
    )
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--federated-rounds", type=int, default=Config.federated_rounds)
    parser.add_argument("--local-epochs", type=int, default=Config.local_epochs)
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--forget-client-id", type=int, default=Config.forget_client_id)
    parser.add_argument("--forget-ratio", type=float, default=Config.forget_ratio)
    parser.add_argument("--momentum-coeff", type=float, default=Config.momentum_coeff)
    parser.add_argument(
        "--degradation-rounds", type=int, default=Config.degradation_rounds
    )
    parser.add_argument("--guidance-rounds", type=int, default=Config.guidance_rounds)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--run-unlearning", action="store_true")
    parser.add_argument("--run-retrain-baseline", action="store_true")

    args = parser.parse_args()
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
        eval_interval=max(1, args.eval_interval),
        forget_client_id=args.forget_client_id,
        forget_ratio=args.forget_ratio,
        momentum_coeff=args.momentum_coeff,
        degradation_rounds=args.degradation_rounds,
        guidance_rounds=max(args.degradation_rounds, args.guidance_rounds),
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        run_unlearning=args.run_unlearning,
        run_retrain_baseline=args.run_retrain_baseline,
    )
