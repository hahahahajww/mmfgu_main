from __future__ import annotations

import copy
from typing import Dict, List

import torch

from FedKD.server import FedKDServer
from mmfgu.model import make_model
from mmfgu.utils import load_model_state, model_state_to_cpu


class RobustnessFedKDServer(FedKDServer):
    def __init__(self, config):
        super().__init__(config)
        self.all_client_deltas: List[List[Dict[str, torch.Tensor]]] = []

    def pretrain(self) -> None:
        current_state = {k: v.clone() for k, v in self.global_state.items()}
        for round_idx in range(self.config.federated_rounds):
            base_state = {k: v.clone() for k, v in current_state.items()}
            client_states = [client.supervised_train(base_state) for client in self.clients]
            self.all_client_deltas.append(
                [self._state_delta(client_state, base_state) for client_state in client_states]
            )
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

    def _apply_target_subtraction_multi(
        self, final_state: Dict[str, torch.Tensor], forget_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        if not self.all_client_deltas:
            return {k: v.clone() for k, v in final_state.items()}
        subtract_delta = {}
        for key in final_state:
            acc = torch.zeros_like(final_state[key], dtype=torch.float32)
            for round_deltas in self.all_client_deltas:
                for client_id in forget_ids:
                    acc += round_deltas[client_id][key].float()
            subtract_delta[key] = acc

        scale = 1.0 / float(self.config.num_clients)
        return {
            key: final_state[key].float() - scale * subtract_delta[key].float()
            for key in final_state
        }

    def _build_public_pool_multi(self, forget_ids: List[int]) -> List[object]:
        forget_set = set(forget_ids)
        return [
            client.device_data()
            for client in self.clients
            if client.client_id not in forget_set
        ]

    def _distill_multi(
        self,
        teacher_state: Dict[str, torch.Tensor],
        student_state: Dict[str, torch.Tensor],
        forget_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
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
        public_pool = self._build_public_pool_multi(forget_ids)
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
                loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
                    torch.nn.functional.softmax(teacher_logits / temperature, dim=-1),
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

    def run_client_unlearning_for_clients(self, forget_ids: List[int]) -> dict[str, object]:
        forget_set = set(forget_ids)
        retained_clients = [
            client for client in self.clients if client.client_id not in forget_set
        ]
        before_metrics = self._evaluate_selected_clients(retained_clients, self.global_state)
        teacher_state = copy.deepcopy(self.global_state)
        subtracted_state = self._apply_target_subtraction_multi(self.global_state, forget_ids)
        distilled_state = self._distill_multi(teacher_state, subtracted_state, forget_ids)
        after_metrics = self._evaluate_selected_clients(retained_clients, distilled_state)

        return {
            "target_client_ids": list(forget_ids),
            "target_client_count": len(forget_ids),
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                f"{key}_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "distill_epochs": self.config.distill_epochs,
            "distill_lr": self.config.distill_lr,
            "distill_temperature": self.config.distill_temperature,
            "historical_round_count": len(self.all_client_deltas),
        }
