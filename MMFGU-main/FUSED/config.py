from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import torch

from mmfgu.config import Config


@dataclass
class FUSEDConfig:
    data_dir: str = Config.data_dir
    task: str = Config.task
    num_clients: int = Config.num_clients
    hidden_dim: int = Config.hidden_dim
    dropout: float = Config.dropout
    gnn_type: str = Config.gnn_type
    lr: float = Config.lr
    weight_decay: float = Config.weight_decay
    batch_size: int = Config.batch_size
    federated_rounds: int = Config.federated_rounds
    local_epochs: int = Config.local_epochs
    eval_interval: int = Config.eval_interval
    forget_client_id: int = Config.forget_client_id
    seed: int = Config.seed
    device: str = Config.device
    output_dir: str = "FUSED/outputs"

    cli_local_epochs: int = 1
    cli_topk_layers: int = 4
    fused_rounds: int = 5
    fused_local_epochs: int = 2
    adapter_density: float = 0.05
    adapter_lr: float = 1e-3

    def to_base_config(self) -> Config:
        return Config(
            data_dir=self.data_dir,
            task=self.task,
            num_clients=self.num_clients,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            gnn_type=self.gnn_type,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            federated_rounds=self.federated_rounds,
            local_epochs=self.local_epochs,
            eval_interval=self.eval_interval,
            forget_client_id=self.forget_client_id,
            seed=self.seed,
            device=self.device,
            output_dir=self.output_dir,
        )

    def asdict(self) -> dict:
        return asdict(self)


def parse_args() -> FUSEDConfig:
    parser = argparse.ArgumentParser(description="FUSED client-unlearning runner")
    parser.add_argument("--data-dir", type=str, default=FUSEDConfig.data_dir)
    parser.add_argument(
        "--task",
        type=str,
        default=FUSEDConfig.task,
        choices=["node_classification", "link_prediction"],
    )
    parser.add_argument("--num-clients", type=int, default=FUSEDConfig.num_clients)
    parser.add_argument("--hidden-dim", type=int, default=FUSEDConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=FUSEDConfig.dropout)
    parser.add_argument(
        "--gnn-type",
        type=str,
        default=FUSEDConfig.gnn_type,
        choices=["sage", "gcn", "gat"],
    )
    parser.add_argument("--lr", type=float, default=FUSEDConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=FUSEDConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=FUSEDConfig.batch_size)
    parser.add_argument("--federated-rounds", type=int, default=FUSEDConfig.federated_rounds)
    parser.add_argument("--local-epochs", type=int, default=FUSEDConfig.local_epochs)
    parser.add_argument("--eval-interval", type=int, default=FUSEDConfig.eval_interval)
    parser.add_argument("--forget-client-id", type=int, default=FUSEDConfig.forget_client_id)
    parser.add_argument("--seed", type=int, default=FUSEDConfig.seed)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=str, default=FUSEDConfig.output_dir)

    parser.add_argument("--cli-local-epochs", type=int, default=FUSEDConfig.cli_local_epochs)
    parser.add_argument("--cli-topk-layers", type=int, default=FUSEDConfig.cli_topk_layers)
    parser.add_argument("--fused-rounds", type=int, default=FUSEDConfig.fused_rounds)
    parser.add_argument("--fused-local-epochs", type=int, default=FUSEDConfig.fused_local_epochs)
    parser.add_argument("--adapter-density", type=float, default=FUSEDConfig.adapter_density)
    parser.add_argument("--adapter-lr", type=float, default=FUSEDConfig.adapter_lr)

    args = parser.parse_args()
    return FUSEDConfig(**vars(args))
