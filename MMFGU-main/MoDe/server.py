from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from mmfgu.data import build_global_graph, split_clients
from mmfgu.model import make_model
from mmfgu.utils import model_state_to_cpu

from .client import MoDeClient
from .config import Config


class MoDeServer:
    def __init__(self, config: Config):
        self.config = config
        self.global_data = build_global_graph(
            Path(config.data_dir), config.seed, config.task
        )
        self.clients: List[MoDeClient] = []
        for client_id, (global_ids, data) in enumerate(
            split_clients(self.global_data, config.num_clients, config.seed)
        ):
            self.clients.append(
                MoDeClient(client_id, global_ids, data, config, self.global_data)
            )

        self.global_state = model_state_to_cpu(make_model(config, self.global_data))
        self.history = {"pretrain": [], "unlearning": [], "retrain": []}
        self.experiment_summary: Dict[str, object] = {
            "pretrain_final": None,
            "client_unlearning": None,
            "client_retrain_baseline": None,
        }

    def _aggregate_metric_rows(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        keys = metrics[0].keys()
        return {f"avg_{key}": float(np.mean([m[key] for m in metrics])) for key in keys}

    def _average_state_dicts(
        self, states: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        result = {}
        for key in states[0]:
            avg = states[0][key].float().clone()
            for state in states[1:]:
                avg += state[key].float()
            result[key] = avg / len(states)
        return result

    def _blend_with_degradation(
        self,
        base_state: Dict[str, torch.Tensor],
        degraded_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        lam = float(self.config.momentum_coeff)
        return {
            key: lam * base_state[key].float() + (1.0 - lam) * degraded_state[key].float()
            for key in base_state
        }

    def _should_eval_round(self, round_idx: int, total_rounds: int) -> bool:
        interval = max(1, self.config.eval_interval)
        return ((round_idx + 1) % interval == 0) or (round_idx + 1 == total_rounds)

    def evaluate_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._aggregate_metric_rows([client.evaluate(state_dict) for client in self.clients])

    def _evaluate_selected_clients(
        self, clients: List[MoDeClient], state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return self._aggregate_metric_rows([client.evaluate(state_dict) for client in clients])

    def pretrain(self) -> None:
        for round_idx in range(self.config.federated_rounds):
            states = [client.supervised_train(self.global_state) for client in self.clients]
            self.global_state = self._average_state_dicts(states)
            if self._should_eval_round(round_idx, self.config.federated_rounds):
                metrics = self.evaluate_state(self.global_state)
                metrics["round"] = round_idx + 1
                self.history["pretrain"].append(metrics)
                print(
                    f"[Pretrain] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f}"
                )
            else:
                print(f"[Pretrain] round={round_idx + 1} train_only")

        self.experiment_summary["pretrain_final"] = self.evaluate_state(self.global_state)

    def run_client_unlearning(self) -> None:
        target_client = self.clients[self.config.forget_client_id]
        remaining_clients = [
            client for client in self.clients if client.client_id != self.config.forget_client_id
        ]

        before_metrics = self.evaluate_state(self.global_state)
        current_state = copy.deepcopy(self.global_state)
        degraded_state = model_state_to_cpu(make_model(self.config, self.global_data))

        for round_idx in range(self.config.guidance_rounds):
            if round_idx < self.config.degradation_rounds:
                degraded_states = [
                    client.supervised_train(degraded_state) for client in remaining_clients
                ]
                degraded_state = self._average_state_dicts(degraded_states)
                current_state = self._blend_with_degradation(current_state, degraded_state)

            local_states = []
            for client in self.clients:
                if client.client_id == self.config.forget_client_id:
                    local_states.append(client.guided_train(current_state, degraded_state))
                else:
                    local_states.append(client.supervised_train(current_state))
            current_state = self._average_state_dicts(local_states)

            if self._should_eval_round(round_idx, self.config.guidance_rounds):
                metrics = self.evaluate_state(current_state)
                metrics["round"] = round_idx + 1
                self.history["unlearning"].append(metrics)
                agreement = target_client.evaluate_pseudo_agreement(current_state, degraded_state)
                print(
                    f"[MoDe] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f} pseudo_agreement={agreement:.4f}"
                )
            else:
                print(f"[MoDe] round={round_idx + 1} train_only")

        self.global_state = current_state
        after_metrics = self.evaluate_state(self.global_state)
        self.experiment_summary["client_unlearning"] = {
            "removed_client_id": self.config.forget_client_id,
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                f"{key}_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "momentum_coeff": self.config.momentum_coeff,
            "degradation_rounds": self.config.degradation_rounds,
            "guidance_rounds": self.config.guidance_rounds,
            "remaining_client_count": len(remaining_clients),
        }

    def run_client_retrain_baseline(self) -> None:
        retained_clients = [
            client for client in self.clients if client.client_id != self.config.forget_client_id
        ]
        retrain_state = model_state_to_cpu(make_model(self.config, self.global_data))
        for round_idx in range(self.config.federated_rounds):
            states = [client.supervised_train(retrain_state) for client in retained_clients]
            retrain_state = self._average_state_dicts(states)
            if self._should_eval_round(round_idx, self.config.federated_rounds):
                metrics = self._evaluate_selected_clients(retained_clients, retrain_state)
                metrics["round"] = round_idx + 1
                self.history["retrain"].append(metrics)
                print(
                    f"[Retrain] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f}"
                )
            else:
                print(f"[Retrain] round={round_idx + 1} train_only")

        self.global_state = retrain_state
        self.clients = retained_clients
        self.experiment_summary["client_retrain_baseline"] = {
            "removed_client_id": self.config.forget_client_id,
            "remaining_client_count": len(retained_clients),
            "final_metrics": self._evaluate_selected_clients(retained_clients, retrain_state),
        }

    def save_outputs(self) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_state, out_dir / "final_global_model.pt")
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)
        with open(out_dir / "config.json", "w", encoding="utf-8") as file:
            json.dump(asdict(self.config), file, indent=2)
        with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as file:
            json.dump(self.experiment_summary, file, indent=2)
