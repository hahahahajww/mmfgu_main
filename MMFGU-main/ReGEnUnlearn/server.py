from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .client import ClientArtifacts, ReGEnUnlearnClient
from .config import Config
from .data import build_global_graph, split_clients
from .model import make_model
from .utils import (
    average_state_dicts,
    load_model_state,
    masked_cross_entropy,
    model_state_to_cpu,
    state_l2_distance,
    to_device_data,
)


class ReGEnUnlearnServer:
    def __init__(self, config: Config):
        self.config = config
        self.global_data = build_global_graph(Path(config.data_dir), config.seed)
        self.clients: List[ReGEnUnlearnClient] = []
        for client_id, (global_ids, data) in enumerate(
            split_clients(self.global_data, config.num_clients, config.seed)
        ):
            self.clients.append(
                ReGEnUnlearnClient(client_id, global_ids, data, config, self.global_data)
            )

        self.global_state = model_state_to_cpu(make_model(config, self.global_data))
        self.history = {"pretrain": [], "unlearn": [], "repair": []}
        self.experiment_summary: Dict[str, object] = {
            "pretrain_final": None,
            "unlearning": None,
            "retrain_baseline": None,
        }

    def _aggregate_metric_rows(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        keys = metrics[0].keys()
        return {f"avg_{key}": float(np.mean([m[key] for m in metrics])) for key in keys}

    def evaluate_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._aggregate_metric_rows([c.evaluate(state_dict) for c in self.clients])

    def _evaluate_selected_clients(
        self, clients: List[ReGEnUnlearnClient], state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return self._aggregate_metric_rows([c.evaluate(state_dict) for c in clients])

    def pretrain(self) -> None:
        for round_idx in range(self.config.federated_rounds):
            states = [client.supervised_train(self.global_state) for client in self.clients]
            self.global_state = average_state_dicts(states)
            if ((round_idx + 1) % self.config.eval_interval == 0) or (
                round_idx + 1 == self.config.federated_rounds
            ):
                metrics = self.evaluate_state(self.global_state)
                metrics["round"] = round_idx + 1
                self.history["pretrain"].append(metrics)
                print(
                    f"[Pretrain] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f}"
                )
            else:
                print(f"[Pretrain] round={round_idx + 1} train_only")
        self.experiment_summary["pretrain_final"] = self.evaluate_state(self.global_state)

    def _target_clients(self) -> List[ReGEnUnlearnClient]:
        target_set = set(self.config.target_client_ids)
        return [client for client in self.clients if client.client_id in target_set]

    def _remaining_clients(self) -> List[ReGEnUnlearnClient]:
        target_set = set(self.config.target_client_ids)
        return [client for client in self.clients if client.client_id not in target_set]

    def _affected_clients(
        self,
        target_clients: List[ReGEnUnlearnClient],
        remaining_clients: List[ReGEnUnlearnClient],
    ) -> tuple[List[int], List[Dict[str, float]]]:
        target_prototypes = [
            client.evaluate_prototype(self.global_state) for client in target_clients
        ]
        rows = []
        affected = []
        for client in remaining_clients:
            prototype = client.evaluate_prototype(self.global_state)
            similarities = [
                float(
                    F.cosine_similarity(
                        prototype.unsqueeze(0), target_proto.unsqueeze(0)
                    ).item()
                )
                for target_proto in target_prototypes
            ]
            score = max(similarities) if similarities else 0.0
            rows.append({"client_id": client.client_id, "similarity": score})
            if score >= self.config.affected_threshold:
                affected.append(client.client_id)
        rows.sort(key=lambda item: item["similarity"], reverse=True)
        return affected, rows

    def _build_guide_state(
        self, remaining_clients: List[ReGEnUnlearnClient]
    ) -> Dict[str, torch.Tensor]:
        states = [
            client.supervised_train(self.global_state, local_epochs=self.config.repair_local_epochs)
            for client in remaining_clients
        ]
        return average_state_dicts(states)

    def _run_server_unlearning(
        self,
        guide_state: Dict[str, torch.Tensor],
        artifacts: List[ClientArtifacts],
    ) -> Dict[str, torch.Tensor]:
        old_model = make_model(self.config, self.global_data).to(self.config.device)
        load_model_state(old_model, self.global_state, self.config.device)
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False

        model = make_model(self.config, self.global_data).to(self.config.device)
        load_model_state(model, self.global_state, self.config.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        for epoch in range(self.config.unlearn_epochs):
            optimizer.zero_grad()
            total_ascent = None
            total_retain = None
            for artifact in artifacts:
                graph = to_device_data(artifact.enhanced_graph, self.config.device)
                logits, _ = model(graph)
                ascent = -masked_cross_entropy(logits, graph.y, graph.train_mask)
                with torch.no_grad():
                    old_logits, _ = old_model(graph)
                retain = F.mse_loss(logits[graph.val_mask], old_logits[graph.val_mask]) if graph.val_mask.sum() > 0 else ascent * 0.0
                total_ascent = ascent if total_ascent is None else total_ascent + ascent
                total_retain = retain if total_retain is None else total_retain + retain

            total_ascent = total_ascent / max(1, len(artifacts))
            total_retain = total_retain / max(1, len(artifacts))
            loss_reg = state_l2_distance(model, guide_state)
            loss = (
                self.config.lambda_ascend * total_ascent
                + self.config.lambda_reg * loss_reg
                + self.config.lambda_retain * total_retain
            )
            loss.backward()
            optimizer.step()

            snapshot = self.evaluate_state(model_state_to_cpu(model))
            snapshot["epoch"] = epoch + 1
            self.history["unlearn"].append(snapshot)
            print(
                f"[Unlearn] epoch={epoch + 1} val={snapshot['avg_val_acc']:.4f} test={snapshot['avg_test_acc']:.4f}"
            )

        return model_state_to_cpu(model)

    def run_unlearning(self) -> None:
        target_clients = self._target_clients()
        remaining_clients = self._remaining_clients()
        if not target_clients:
            raise ValueError("No target clients selected for unlearning.")

        before_metrics = self.evaluate_state(self.global_state)
        remaining_prototypes = [
            client.evaluate_prototype(self.global_state) for client in remaining_clients
        ]
        artifacts = [
            client.build_unlearning_artifacts(self.global_state, remaining_prototypes)
            for client in target_clients
        ]
        guide_state = self._build_guide_state(remaining_clients)
        affected_ids, similarity_rows = self._affected_clients(target_clients, remaining_clients)
        current_state = self._run_server_unlearning(guide_state, artifacts)

        repair_clients = [
            client for client in remaining_clients if client.client_id in set(affected_ids)
        ]
        if not repair_clients:
            repair_clients = remaining_clients

        for round_idx in range(self.config.repair_rounds):
            updated_states = []
            repair_ids = {client.client_id for client in repair_clients}
            for client in remaining_clients:
                if client.client_id in repair_ids:
                    updated_states.append(
                        client.supervised_train(
                            current_state, local_epochs=self.config.repair_local_epochs
                        )
                    )
                else:
                    updated_states.append(current_state)
            current_state = average_state_dicts(updated_states)
            metrics = self._evaluate_selected_clients(remaining_clients, current_state)
            metrics["round"] = round_idx + 1
            self.history["repair"].append(metrics)
            print(
                f"[Repair] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f}"
            )

        self.global_state = current_state
        after_metrics = self._evaluate_selected_clients(remaining_clients, self.global_state)

        self.experiment_summary["unlearning"] = {
            "target_client_ids": list(self.config.target_client_ids),
            "before_global_metrics": before_metrics,
            "after_remaining_metrics": after_metrics,
            "metric_delta": {
                f"{key}_delta": after_metrics.get(key, 0.0) - before_metrics.get(key, 0.0)
                for key in before_metrics.keys()
                if key in after_metrics
            },
            "affected_client_ids": affected_ids,
            "prototype_similarities": similarity_rows,
            "target_artifacts": [artifact.info for artifact in artifacts],
        }

    def run_retrain_baseline(self) -> None:
        retained_clients = self._remaining_clients()
        retrain_state = model_state_to_cpu(make_model(self.config, self.global_data))
        retrain_history = []
        for round_idx in range(self.config.federated_rounds):
            states = [client.supervised_train(retrain_state) for client in retained_clients]
            retrain_state = average_state_dicts(states)
            if ((round_idx + 1) % self.config.eval_interval == 0) or (
                round_idx + 1 == self.config.federated_rounds
            ):
                metrics = self._evaluate_selected_clients(retained_clients, retrain_state)
                metrics["round"] = round_idx + 1
                retrain_history.append(metrics)
                print(
                    f"[Retrain] round={round_idx + 1} val={metrics['avg_val_acc']:.4f} test={metrics['avg_test_acc']:.4f}"
                )
        self.global_state = retrain_state
        self.clients = retained_clients
        self.history["pretrain"] = retrain_history
        self.experiment_summary["retrain_baseline"] = {
            "removed_client_ids": list(self.config.target_client_ids),
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
