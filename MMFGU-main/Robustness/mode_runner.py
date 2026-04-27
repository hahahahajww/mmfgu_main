from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import torch

from MoDe.server import MoDeServer
from mmfgu.model import make_model
from mmfgu.utils import model_state_to_cpu


class RobustnessMoDeServer(MoDeServer):
    def run_client_unlearning_for_clients(self, target_ids: List[int]) -> dict[str, object]:
        target_set = set(target_ids)
        target_clients = [self.clients[client_id] for client_id in target_ids]
        remaining_clients = [
            client for client in self.clients if client.client_id not in target_set
        ]

        before_metrics = self._evaluate_selected_clients(remaining_clients, self.global_state)
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
                if client.client_id in target_set:
                    local_states.append(client.guided_train(current_state, degraded_state))
                else:
                    local_states.append(client.supervised_train(current_state))
            current_state = self._average_state_dicts(local_states)

            if self._should_eval_round(round_idx, self.config.guidance_rounds):
                metrics = self._evaluate_selected_clients(remaining_clients, current_state)
                metrics["round"] = round_idx + 1
                self.history["unlearning"].append(metrics)
                print(
                    f"[MoDe Robustness] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f} targets={len(target_ids)}"
                )
            else:
                print(f"[MoDe Robustness] round={round_idx + 1} train_only")

        after_metrics = self._evaluate_selected_clients(remaining_clients, current_state)
        return {
            "target_client_ids": list(target_ids),
            "target_client_count": len(target_ids),
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
