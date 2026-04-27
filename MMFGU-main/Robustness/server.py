from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mmfgu.client import FederatedClient
from mmfgu.config import Config
from mmfgu.data import build_global_graph, split_clients
from mmfgu.model import make_model
from mmfgu.training_utils import build_probe_graphs
from mmfgu.utils import model_state_to_cpu


class RobustnessMMFGUServer:
    def __init__(self, config: Config):
        self.config = config
        print(f"[ServerInit] Loading global graph from {config.data_dir}")
        self.global_data = build_global_graph(
            Path(config.data_dir), config.seed, config.task
        )
        print(
            f"[ServerInit] Global graph ready: num_nodes={self.global_data.num_nodes} "
            f"num_edges={int(self.global_data.edge_index.size(1))}"
        )
        self.clients: List[FederatedClient] = []
        print("[ServerInit] Splitting clients")
        for client_id, (global_ids, data) in enumerate(
            split_clients(self.global_data, config.num_clients, config.seed)
        ):
            self.clients.append(
                FederatedClient(client_id, global_ids, data, config, self.global_data)
            )
        print(f"[ServerInit] Client objects ready: count={len(self.clients)}")

        self.global_state = model_state_to_cpu(make_model(config, self.global_data))
        print("[ServerInit] Global model initialized")
        self.history = {"pretrain": [], "purge": []}

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

    def _should_eval_round(self, round_idx: int, total_rounds: int) -> bool:
        interval = max(1, self.config.eval_interval)
        return ((round_idx + 1) % interval == 0) or (round_idx + 1 == total_rounds)

    def _main_metric_names(self) -> tuple[str, str]:
        return "avg_val_acc", "avg_test_acc"

    def evaluate_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._aggregate_metric_rows([client.evaluate(state_dict) for client in self.clients])

    def _evaluate_selected_clients(
        self, clients: List[FederatedClient], state_dict: Dict[str, torch.Tensor]
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
                val_key, test_key = self._main_metric_names()
                print(
                    f"[Pretrain] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}"
                )
            else:
                print(f"[Pretrain] round={round_idx + 1} train_only")

    def _affected_clients_multi(
        self, target_ids: List[int], target_prototypes: List[torch.Tensor]
    ) -> tuple[List[int], List[Dict[str, float]]]:
        affected = []
        rows = []
        target_set = set(target_ids)
        for client in self.clients:
            if client.client_id in target_set or not client.history:
                continue
            prototype = client.history[-1]["prototype"]
            similarity = max(
                float(
                    F.cosine_similarity(
                        target_proto.unsqueeze(0), prototype.unsqueeze(0)
                    ).item()
                )
                for target_proto in target_prototypes
            )
            rows.append({"client_id": client.client_id, "similarity": similarity})
            if similarity > self.config.prototype_threshold:
                affected.append(client.client_id)
        rows.sort(key=lambda row: row["similarity"], reverse=True)
        return affected, rows

    def _build_multi_client_probes(self, target_ids: List[int]) -> List[object]:
        probes = []
        per_client_count = max(1, self.config.probe_count // max(1, len(target_ids)))
        for client_id in target_ids:
            requester = self.clients[client_id]
            requester_train_nodes = requester.data.train_mask.nonzero(as_tuple=False).view(-1)
            probes.extend(
                build_probe_graphs(
                    requester.data,
                    requester_train_nodes.cpu(),
                    per_client_count,
                    self.config.seed + client_id,
                )
            )
        return probes[: self.config.probe_count]

    def run_client_unlearning_for_targets(self, target_ids: List[int]) -> Dict[str, object]:
        target_set = set(target_ids)
        remaining_clients = [
            client for client in self.clients if client.client_id not in target_set
        ]
        before_metrics = self._evaluate_selected_clients(remaining_clients, self.global_state)

        target_prototypes = [
            self.clients[client_id].evaluate_prototype(self.global_state)
            for client_id in target_ids
        ]
        affected, similarity_rows = self._affected_clients_multi(target_ids, target_prototypes)
        probes = self._build_multi_client_probes(target_ids)
        noise_teacher_state = model_state_to_cpu(make_model(self.config, self.global_data))

        current_state = copy.deepcopy(self.global_state)
        purge_history = []
        for round_idx in range(self.config.purge_rounds):
            all_states = []
            for client in remaining_clients:
                if client.client_id in affected:
                    all_states.append(
                        client.purge_train(current_state, noise_teacher_state, probes)
                    )
                else:
                    all_states.append(current_state)
            current_state = self._average_state_dicts(all_states)
            if self._should_eval_round(round_idx, self.config.purge_rounds):
                metrics = self._evaluate_selected_clients(remaining_clients, current_state)
                metrics["round"] = round_idx + 1
                purge_history.append(metrics)
                val_key, test_key = self._main_metric_names()
                print(
                    f"[Purge] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f} affected={len(affected)}"
                )
            else:
                print(f"[Purge] round={round_idx + 1} train_only")

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
            "affected_client_ids": affected,
            "affected_client_count": len(affected),
            "prototype_similarities": similarity_rows,
            "probe_stats": {"probe_count": len(probes)},
            "purge_history": purge_history,
        }
