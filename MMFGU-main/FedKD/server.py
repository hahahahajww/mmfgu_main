from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mmfgu.client import FederatedClient
from mmfgu.data import build_global_graph, split_clients
from mmfgu.model import make_model
from mmfgu.utils import load_model_state, model_state_to_cpu

from .config import Config


class FedKDServer:
    def __init__(self, config: Config):
        self.config = config
        self.global_data = build_global_graph(
            Path(config.data_dir), config.seed, config.task
        )
        self.clients: List[FederatedClient] = []
        for client_id, (global_ids, data) in enumerate(
            split_clients(self.global_data, config.num_clients, config.seed)
        ):
            self.clients.append(
                FederatedClient(client_id, global_ids, data, config, self.global_data)
            )

        self.global_state = model_state_to_cpu(make_model(config, self.global_data))
        self.history = {"pretrain": [], "distill": [], "retrain": []}
        self.experiment_summary: Dict[str, object] = {
            "pretrain_final": None,
            "fedkd": None,
            "client_retrain_baseline": None,
        }
        self.target_deltas: List[Dict[str, torch.Tensor]] = []

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

    def _state_delta(
        self, new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {k: new_state[k].float() - old_state[k].float() for k in new_state}

    def _apply_target_subtraction(
        self, final_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self.target_deltas:
            return {k: v.clone() for k, v in final_state.items()}
        avg_target_delta = {}
        for key in self.target_deltas[0]:
            acc = self.target_deltas[0][key].float().clone()
            for delta in self.target_deltas[1:]:
                acc += delta[key].float()
            avg_target_delta[key] = acc / len(self.target_deltas)

        scale = 1.0 / float(self.config.num_clients)
        return {
            key: final_state[key].float() - scale * avg_target_delta[key].float()
            for key in final_state
        }

    def _should_eval_round(self, round_idx: int, total_rounds: int) -> bool:
        interval = max(1, self.config.eval_interval)
        return ((round_idx + 1) % interval == 0) or (round_idx + 1 == total_rounds)

    def evaluate_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._aggregate_metric_rows([client.evaluate(state_dict) for client in self.clients])

    def _evaluate_selected_clients(
        self, clients: List[FederatedClient], state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return self._aggregate_metric_rows([client.evaluate(state_dict) for client in clients])

    def _main_metric_names(self) -> tuple[str, str]:
        return "avg_val_acc", "avg_test_acc"

    def pretrain(self) -> None:
        current_state = {k: v.clone() for k, v in self.global_state.items()}
        for round_idx in range(self.config.federated_rounds):
            base_state = {k: v.clone() for k, v in current_state.items()}
            client_states = [client.supervised_train(base_state) for client in self.clients]
            target_state = client_states[self.config.forget_client_id]
            self.target_deltas.append(self._state_delta(target_state, base_state))
            current_state = self._average_state_dicts(client_states)
            if self._should_eval_round(round_idx, self.config.federated_rounds):
                metrics = self.evaluate_state(current_state)
                metrics["round"] = round_idx + 1
                self.history["pretrain"].append(metrics)
                print(
                    f"[Pretrain] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f}"
                )
            else:
                print(f"[Pretrain] round={round_idx + 1} train_only")

        self.global_state = current_state
        self.experiment_summary["pretrain_final"] = self.evaluate_state(self.global_state)

    def _build_public_pool(self) -> List[object]:
        return [
            client.device_data()
            for client in self.clients
            if client.client_id != self.config.forget_client_id
        ]

    def _distill(self, teacher_state: Dict[str, torch.Tensor], student_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        teacher = make_model(self.config, self.global_data).to(self.config.device)
        student = make_model(self.config, self.global_data).to(self.config.device)
        load_model_state(teacher, teacher_state, self.config.device)
        load_model_state(student, student_state, self.config.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=self.config.distill_lr,
            weight_decay=self.config.weight_decay,
        )
        public_pool = self._build_public_pool()
        temperature = float(self.config.distill_temperature)

        for epoch in range(self.config.distill_epochs):
            student.train()
            total_loss = 0.0
            used_graphs = 0
            for graph in public_pool:
                optimizer.zero_grad()
                student_logits, _ = student(graph)
                with torch.no_grad():
                    teacher_logits, _ = teacher(graph)

                loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                used_graphs += 1

            snapshot = self.evaluate_state(model_state_to_cpu(student))
            snapshot["epoch"] = epoch + 1
            snapshot["distill_loss"] = total_loss / max(1, used_graphs)
            self.history["distill"].append(snapshot)
            print(
                f"[Distill] epoch={epoch + 1} val={snapshot['avg_val_acc']:.4f} test={snapshot['avg_test_acc']:.4f} loss={snapshot['distill_loss']:.4f}"
            )

        return model_state_to_cpu(student)

    def run_client_unlearning(self) -> None:
        retained_clients = [
            client for client in self.clients if client.client_id != self.config.forget_client_id
        ]
        before_metrics = self._evaluate_selected_clients(retained_clients, self.global_state)
        teacher_state = copy.deepcopy(self.global_state)
        subtracted_state = self._apply_target_subtraction(self.global_state)
        distilled_state = self._distill(teacher_state, subtracted_state)

        self.global_state = distilled_state
        self.clients = retained_clients
        after_metrics = self._evaluate_selected_clients(retained_clients, self.global_state)
        self.experiment_summary["fedkd"] = {
            "removed_client_id": self.config.forget_client_id,
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                f"{key}_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "distill_epochs": self.config.distill_epochs,
            "distill_lr": self.config.distill_lr,
            "distill_temperature": self.config.distill_temperature,
            "historical_round_count": len(self.target_deltas),
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
