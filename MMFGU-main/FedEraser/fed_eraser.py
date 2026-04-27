from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from mmfgu.client import FederatedClient
from mmfgu.config import Config
from mmfgu.model import make_model
from mmfgu.server import FederatedServer
from mmfgu.training_utils import masked_cross_entropy
from mmfgu.utils import load_model_state, model_state_to_cpu, to_device_data


class FedEraserRunner:
    """A lightweight FedEraser baseline for client-level unlearning."""

    def __init__(self, config: Config, retain_interval: int = 2, calibration_ratio: float = 0.5):
        self.config = config
        self.server = FederatedServer(config)
        self.retain_interval = max(1, int(retain_interval))
        self.calibration_ratio = float(calibration_ratio)
        self.calibration_local_epochs = max(1, int(round(config.local_epochs * self.calibration_ratio)))
        self.initial_state = {k: v.clone() for k, v in self.server.global_state.items()}
        self.retained_rounds: list[dict] = []
        self.history = {"pretrain": [], "reconstruction": []}
        self.summary: dict[str, object] = {
            "pretrain_final": None,
            "fed_eraser": None,
        }

    def _average_state_dicts(self, states: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.server._average_state_dicts(states)

    def _client_weight(self, client: FederatedClient) -> float:
        if self.config.task == "link_prediction" and hasattr(client.data, "lp_train_source_node"):
            return float(max(1, int(client.data.lp_train_source_node.numel())))
        return float(max(1, int(client.data.train_mask.sum().item())))

    def _state_delta(self, new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: new_state[k].float() - old_state[k].float() for k in new_state}

    def _apply_delta(self, state: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: state[k].float() + delta[k].float() for k in state}

    def _aggregate_weighted_deltas(self, deltas: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
        total_weight = float(sum(weights))
        result = {}
        for key in deltas[0]:
            acc = deltas[0][key].float() * float(weights[0])
            for delta, weight in zip(deltas[1:], weights[1:]):
                acc += delta[key].float() * float(weight)
            result[key] = acc / total_weight
        return result

    def _train_client_epochs(self, client: FederatedClient, base_state: Dict[str, torch.Tensor], epochs: int) -> Dict[str, torch.Tensor]:
        model = client.new_model()
        load_model_state(model, base_state, self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        data = client.device_data()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            if self.config.task == "link_prediction":
                loss = client._link_prediction_loss(model, data, data.lp_train_source_node, data.lp_train_target_node)
            else:
                logits, _ = model(data)
                loss = masked_cross_entropy(logits, data.y, data.train_mask)
            loss.backward()
            optimizer.step()

        return model_state_to_cpu(model)

    def _calibrate_delta(self, historical_delta: Dict[str, torch.Tensor], calibration_delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        calibrated = {}
        for key in historical_delta:
            hist = historical_delta[key].float()
            cal = calibration_delta[key].float()
            cal_norm = torch.norm(cal)
            if float(cal_norm.item()) == 0.0:
                calibrated[key] = torch.zeros_like(hist)
                continue
            hist_norm = torch.norm(hist)
            calibrated[key] = hist_norm * (cal / cal_norm)
        return calibrated

    def pretrain_with_retention(self) -> None:
        current_state = {k: v.clone() for k, v in self.server.global_state.items()}
        round_start_state = {k: v.clone() for k, v in current_state.items()}

        for round_idx in range(self.config.federated_rounds):
            client_states = [client.supervised_train(current_state) for client in self.server.clients]
            current_state = self._average_state_dicts(client_states)

            if self.server._should_eval_round(round_idx, self.config.federated_rounds):
                metrics = self.server.evaluate_state(current_state)
                metrics["round"] = round_idx + 1
                self.history["pretrain"].append(metrics)
                val_key, test_key = self.server._main_metric_names()
                print(f"[FedEraser Pretrain] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}")
            else:
                print(f"[FedEraser Pretrain] round={round_idx + 1} train_only")

            if ((round_idx + 1) % self.retain_interval == 0) or (round_idx + 1 == self.config.federated_rounds):
                retained_client_deltas = []
                for client_state in client_states:
                    retained_client_deltas.append(self._state_delta(client_state, round_start_state))
                self.retained_rounds.append(
                    {
                        "round": round_idx + 1,
                        "base_state": {k: v.clone() for k, v in round_start_state.items()},
                        "client_deltas": retained_client_deltas,
                    }
                )
                round_start_state = {k: v.clone() for k, v in current_state.items()}

        self.server.global_state = current_state
        self.summary["pretrain_final"] = self.server.evaluate_state(current_state)

    def reconstruct_without_client(self) -> None:
        forget_id = self.config.forget_client_id
        retained_clients = [client for client in self.server.clients if client.client_id != forget_id]
        before_metrics = self.server._evaluate_clients_state(retained_clients, self.server.global_state)

        reconstruction_state = {k: v.clone() for k, v in self.initial_state.items()}
        start_time = time.time()

        total_retained_rounds = len(self.retained_rounds)
        for retained_idx, retained in enumerate(self.retained_rounds):
            base_state = retained["base_state"]
            calibrated_deltas = []
            weights = []

            for client in retained_clients:
                calibration_state = self._train_client_epochs(client, reconstruction_state, self.calibration_local_epochs)
                calibration_delta = self._state_delta(calibration_state, reconstruction_state)
                historical_delta = retained["client_deltas"][client.client_id]
                calibrated_deltas.append(self._calibrate_delta(historical_delta, calibration_delta))
                weights.append(self._client_weight(client))

            aggregated_delta = self._aggregate_weighted_deltas(calibrated_deltas, weights)
            reconstruction_state = self._apply_delta(reconstruction_state, aggregated_delta)
            if self.server._should_eval_round(retained_idx, total_retained_rounds):
                metrics = self.server._evaluate_clients_state(retained_clients, reconstruction_state)
                metrics["round"] = retained["round"]
                self.history["reconstruction"].append(metrics)
                val_key, test_key = self.server._main_metric_names()
                print(f"[FedEraser Reconstruct] round={retained['round']} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}")
            else:
                print(f"[FedEraser Reconstruct] round={retained['round']} reconstruct_only")

        elapsed = time.time() - start_time
        self.server.global_state = reconstruction_state
        self.server.clients = retained_clients
        after_metrics = self.server._evaluate_clients_state(retained_clients, reconstruction_state)
        self.summary["fed_eraser"] = {
            "removed_client_id": forget_id,
            "retain_interval": self.retain_interval,
            "calibration_ratio": self.calibration_ratio,
            "calibration_local_epochs": self.calibration_local_epochs,
            "retained_round_count": len(self.retained_rounds),
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {key + "_delta": after_metrics[key] - before_metrics[key] for key in before_metrics},
            "reconstruction_seconds": elapsed,
        }

    def save_outputs(self) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.server.global_state, output_dir / "final_global_model.pt")
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            payload = {
                **asdict(self.config),
                "retain_interval": self.retain_interval,
                "calibration_ratio": self.calibration_ratio,
                "calibration_local_epochs": self.calibration_local_epochs,
            }
            json.dump(payload, f, indent=2)
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
