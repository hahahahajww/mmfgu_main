from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import torch

from FUSED.fused import FUSEDRunner
from mmfgu.client import FederatedClient


class RobustnessFUSEDRunner(FUSEDRunner):
    def run_unlearning_for_clients(self, forget_ids: List[int]) -> dict[str, object]:
        forget_set = set(forget_ids)
        retained_clients = [
            client for client in self.server.clients if client.client_id not in forget_set
        ]

        self.base_state = copy.deepcopy(self.server.global_state)
        before_metrics = self._evaluate_clients_state(retained_clients, self.base_state)

        self.selected_layers, self.cli_scores = self._cli_parameter_scores(self.base_state)
        self.adapter_masks = self._build_sparse_masks(self.base_state, self.selected_layers)
        self.adapter_state = self._zero_adapters(self.base_state)

        for round_idx in range(self.config.fused_rounds):
            local_adapters = [
                self._train_client_adapter(
                    client,
                    self.base_state,
                    self.adapter_state,
                    self.adapter_masks,
                )
                for client in retained_clients
            ]
            self.adapter_state = self._aggregate_adapters(local_adapters, retained_clients)
            merged_state = self._merge_state(
                self.base_state, self.adapter_state, self.adapter_masks
            )
            if self.server._should_eval_round(round_idx, self.config.fused_rounds):
                metrics = self._evaluate_clients_state(retained_clients, merged_state)
                metrics["round"] = round_idx + 1
                self.history["unlearning"].append(metrics)
                val_key, test_key = self._main_metric_names()
                print(
                    f"[FUSED Robustness] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f} targets={len(forget_ids)}"
                )
            else:
                print(f"[FUSED Robustness] round={round_idx + 1} train_only")

        self.final_state = self._merge_state(
            self.base_state, self.adapter_state, self.adapter_masks
        )
        after_metrics = self._evaluate_clients_state(retained_clients, self.final_state)
        selected_param_count = int(sum(self.base_state[key].numel() for key in self.selected_layers))
        active_adapter_count = int(
            sum(int(self.adapter_masks[key].sum().item()) for key in self.selected_layers)
        )
        total_param_count = int(
            sum(value.numel() for value in self.base_state.values() if torch.is_floating_point(value))
        )

        return {
            "target_client_ids": list(forget_ids),
            "target_client_count": len(forget_ids),
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                key + "_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "cli": {
                "local_epochs": self.config.cli_local_epochs,
                "topk_layers": self.config.cli_topk_layers,
                "selected_layers": self.selected_layers,
                "scores": self.cli_scores,
            },
            "adapter": {
                "density": self.config.adapter_density,
                "selected_param_count": selected_param_count,
                "active_adapter_count": active_adapter_count,
                "total_float_param_count": total_param_count,
                "compression_ratio_vs_full_model": (
                    float(active_adapter_count) / float(total_param_count)
                    if total_param_count > 0
                    else 0.0
                ),
            },
        }
