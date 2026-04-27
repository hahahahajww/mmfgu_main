from __future__ import annotations

import copy
import time
from typing import Dict, List

import torch

from FedEraser.fed_eraser import FedEraserRunner


class RobustnessFedEraserRunner(FedEraserRunner):
    def reconstruct_without_clients(self, forget_ids: List[int]) -> dict[str, object]:
        forget_set = set(forget_ids)
        retained_clients = [
            client for client in self.server.clients if client.client_id not in forget_set
        ]
        before_metrics = self.server._evaluate_clients_state(
            retained_clients, self.server.global_state
        )

        reconstruction_state = {k: v.clone() for k, v in self.initial_state.items()}
        start_time = time.time()
        total_retained_rounds = len(self.retained_rounds)
        reconstruction_history = []

        for retained_idx, retained in enumerate(self.retained_rounds):
            calibrated_deltas = []
            weights = []
            for client in retained_clients:
                calibration_state = self._train_client_epochs(
                    client, reconstruction_state, self.calibration_local_epochs
                )
                calibration_delta = self._state_delta(calibration_state, reconstruction_state)
                historical_delta = retained["client_deltas"][client.client_id]
                calibrated_deltas.append(
                    self._calibrate_delta(historical_delta, calibration_delta)
                )
                weights.append(self._client_weight(client))

            aggregated_delta = self._aggregate_weighted_deltas(calibrated_deltas, weights)
            reconstruction_state = self._apply_delta(reconstruction_state, aggregated_delta)
            if self.server._should_eval_round(retained_idx, total_retained_rounds):
                metrics = self.server._evaluate_clients_state(
                    retained_clients, reconstruction_state
                )
                metrics["round"] = retained["round"]
                reconstruction_history.append(metrics)
                val_key, test_key = self.server._main_metric_names()
                print(
                    f"[FedEraser Robustness] round={retained['round']} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f} targets={len(forget_ids)}"
                )
            else:
                print(
                    f"[FedEraser Robustness] round={retained['round']} reconstruct_only"
                )

        elapsed = time.time() - start_time
        after_metrics = self.server._evaluate_clients_state(
            retained_clients, reconstruction_state
        )
        return {
            "target_client_ids": list(forget_ids),
            "target_client_count": len(forget_ids),
            "retain_interval": self.retain_interval,
            "calibration_ratio": self.calibration_ratio,
            "calibration_local_epochs": self.calibration_local_epochs,
            "retained_round_count": len(self.retained_rounds),
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                key + "_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "reconstruction_seconds": elapsed,
            "reconstruction_history": reconstruction_history,
        }
