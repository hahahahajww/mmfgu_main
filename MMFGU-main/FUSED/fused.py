from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from mmfgu.client import FederatedClient
from mmfgu.server import FederatedServer
from mmfgu.training_utils import masked_cross_entropy
from mmfgu.utils import set_seed, to_device_data

from .config import FUSEDConfig


class FUSEDRunner:
    def __init__(self, config: FUSEDConfig):
        self.config = config
        self.base_config = config.to_base_config()
        self.server = FederatedServer(self.base_config)
        self.history: dict[str, list[dict]] = {"pretrain": [], "unlearning": []}
        self.summary: dict[str, object] = {
            "pretrain_final": None,
            "fused": None,
        }
        self.selected_layers: list[str] = []
        self.cli_scores: list[dict[str, float]] = []
        self.adapter_masks: dict[str, torch.Tensor] = {}
        self.adapter_state: dict[str, torch.Tensor] = {}
        self.final_state: dict[str, torch.Tensor] | None = None
        self.base_state: dict[str, torch.Tensor] | None = None

    def _main_metric_names(self) -> tuple[str, str]:
        return self.server._main_metric_names()

    def _client_weight(self, client: FederatedClient) -> float:
        if self.base_config.task == "link_prediction" and hasattr(
            client.data, "lp_train_source_node"
        ):
            return float(max(1, int(client.data.lp_train_source_node.numel())))
        return float(max(1, int(client.data.train_mask.sum().item())))

    def _evaluate_clients_state(
        self, clients: Sequence[FederatedClient], state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return self.server._aggregate_metric_rows(
            [client.evaluate(state_dict) for client in clients]
        )

    def _device_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.config.device) for k, v in state.items()}

    def _train_state_for_cli(
        self, client: FederatedClient, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        original_epochs = client.config.local_epochs
        client.config.local_epochs = self.config.cli_local_epochs
        try:
            return client.supervised_train(state_dict)
        finally:
            client.config.local_epochs = original_epochs

    def _cli_parameter_scores(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> tuple[list[str], list[dict[str, float]]]:
        trainable_keys = []
        for key, value in state_dict.items():
            if not torch.is_floating_point(value):
                continue
            if value.numel() == 0:
                continue
            trainable_keys.append(key)

        weighted_scores = {key: 0.0 for key in trainable_keys}
        total_weight = 0.0
        for client in self.server.clients:
            local_state = self._train_state_for_cli(client, state_dict)
            weight = self._client_weight(client)
            total_weight += weight
            for key in trainable_keys:
                diff = (local_state[key].float() - state_dict[key].float()).abs().sum()
                weighted_scores[key] += float(diff.item()) * weight

        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight

        ranked = sorted(weighted_scores.items(), key=lambda item: item[1], reverse=True)
        score_rows = [{"layer": key, "diff": score} for key, score in ranked]
        topk = min(self.config.cli_topk_layers, len(ranked))
        return [key for key, _ in ranked[:topk]], score_rows

    def _build_sparse_masks(
        self, state_dict: Dict[str, torch.Tensor], target_keys: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.config.seed)
        masks: dict[str, torch.Tensor] = {}
        for key in target_keys:
            reference = state_dict[key]
            mask = (torch.rand(reference.shape, generator=generator) < self.config.adapter_density).to(
                reference.dtype
            )
            if mask.sum() == 0:
                flat = mask.view(-1)
                flat[0] = 1.0
                mask = flat.view_as(mask)
            masks[key] = mask.cpu()
        return masks

    def _zero_adapters(self, state_dict: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: torch.zeros_like(state_dict[key]) for key in self.selected_layers}

    def _merge_state(
        self,
        base_state: Dict[str, torch.Tensor],
        adapters: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        merged = {key: value.clone() for key, value in base_state.items()}
        for key in adapters:
            merged[key] = base_state[key].float() + adapters[key].float() * masks[key].float()
        return merged

    def _link_prediction_loss(
        self,
        model,
        params: Dict[str, torch.Tensor],
        data,
        batch_size: int,
    ) -> torch.Tensor:
        src = data.lp_train_source_node
        pos = data.lp_train_target_node
        if src.numel() == 0:
            return next(iter(params.values())).sum() * 0.0

        _, cache = functional_call(model, params, (data,))
        node_h = cache["propagated_h"]
        losses = []
        for start in range(0, src.size(0), batch_size):
            batch_src = src[start : start + batch_size]
            batch_pos = pos[start : start + batch_size]
            neg = torch.randint(0, data.num_nodes, batch_pos.shape, device=batch_src.device)
            pos_score = model.score_pairs(node_h, batch_src, batch_pos)
            neg_score = model.score_pairs(node_h, batch_src, neg)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    torch.cat([pos_score, neg_score], dim=0),
                    torch.cat(
                        [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
                    ),
                )
            )
        return torch.stack(losses).mean()

    def _train_client_adapter(
        self,
        client: FederatedClient,
        base_state: Dict[str, torch.Tensor],
        global_adapters: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        model = client.new_model().to(self.config.device)
        model.train()
        base_device = self._device_state(base_state)
        mask_device = {key: value.to(self.config.device) for key, value in masks.items()}
        adapter_params = {
            key: torch.nn.Parameter(global_adapters[key].to(self.config.device).clone())
            for key in self.selected_layers
        }
        optimizer = torch.optim.Adam(adapter_params.values(), lr=self.config.adapter_lr)
        data = to_device_data(client.data, self.config.device)
        batch_size = max(1, self.base_config.batch_size)

        for _ in range(self.config.fused_local_epochs):
            optimizer.zero_grad()
            merged_params = dict(base_device)
            for key in self.selected_layers:
                merged_params[key] = base_device[key] + adapter_params[key] * mask_device[key]

            if self.base_config.task == "link_prediction":
                loss = self._link_prediction_loss(model, merged_params, data, batch_size)
            else:
                logits, _ = functional_call(model, merged_params, (data,))
                loss = masked_cross_entropy(logits, data.y, data.train_mask)

            loss.backward()
            optimizer.step()

        return {key: value.detach().cpu().clone() for key, value in adapter_params.items()}

    def _aggregate_adapters(
        self,
        local_adapters: Sequence[Dict[str, torch.Tensor]],
        clients: Sequence[FederatedClient],
    ) -> Dict[str, torch.Tensor]:
        total_weight = sum(self._client_weight(client) for client in clients)
        result: dict[str, torch.Tensor] = {}
        for key in self.selected_layers:
            acc = torch.zeros_like(local_adapters[0][key], dtype=torch.float32)
            for adapter, client in zip(local_adapters, clients):
                acc += adapter[key].float() * self._client_weight(client)
            result[key] = acc / float(total_weight)
        return result

    def pretrain(self) -> None:
        self.server.pretrain()
        self.history["pretrain"] = list(self.server.history["pretrain"])
        self.summary["pretrain_final"] = self.server.evaluate_state(self.server.global_state)

    def run_unlearning(self) -> None:
        forget_id = self.config.forget_client_id
        retained_clients = [
            client for client in self.server.clients if client.client_id != forget_id
        ]
        forgotten_client = self.server.clients[forget_id]

        self.base_state = copy.deepcopy(self.server.global_state)
        before_metrics = self._evaluate_clients_state(retained_clients, self.base_state)
        forgotten_before = forgotten_client.evaluate(self.base_state)

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
                    f"[FUSED] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}"
                )
            else:
                print(f"[FUSED] round={round_idx + 1} train_only")

        self.final_state = self._merge_state(
            self.base_state, self.adapter_state, self.adapter_masks
        )
        self.server.global_state = self.final_state
        self.server.clients = retained_clients

        after_metrics = self._evaluate_clients_state(retained_clients, self.final_state)
        forgotten_after = forgotten_client.evaluate(self.final_state)
        selected_param_count = int(sum(self.base_state[key].numel() for key in self.selected_layers))
        active_adapter_count = int(
            sum(int(self.adapter_masks[key].sum().item()) for key in self.selected_layers)
        )
        total_param_count = int(
            sum(value.numel() for value in self.base_state.values() if torch.is_floating_point(value))
        )

        self.summary["fused"] = {
            "removed_client_id": forget_id,
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                key + "_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "forgotten_client_metrics_before": forgotten_before,
            "forgotten_client_metrics_after": forgotten_after,
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
            "reversible": {
                "base_model_path": "base_global_model.pt",
                "adapter_path": "adapter_state.pt",
                "mask_path": "adapter_masks.pt",
                "restore_note": "Removing adapters restores the pretrained base model.",
            },
        }

    def save_outputs(self) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.base_state is not None:
            torch.save(self.base_state, output_dir / "base_global_model.pt")
        if self.final_state is not None:
            torch.save(self.final_state, output_dir / "final_global_model.pt")
        torch.save(self.adapter_state, output_dir / "adapter_state.pt")
        torch.save(self.adapter_masks, output_dir / "adapter_masks.pt")

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)


def run(config: FUSEDConfig) -> FUSEDRunner:
    set_seed(config.seed)
    runner = FUSEDRunner(config)
    runner.pretrain()
    runner.run_unlearning()
    runner.save_outputs()
    return runner
