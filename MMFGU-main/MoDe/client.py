from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from mmfgu.model import MMFGUModel, make_model
from mmfgu.training_utils import accuracy, masked_cross_entropy
from mmfgu.utils import load_model_state, model_state_to_cpu, to_device_data

from .config import Config


class MoDeClient:
    def __init__(
        self,
        client_id: int,
        global_ids: torch.Tensor,
        data: Data,
        config: Config,
        global_template: Data,
    ):
        self.client_id = client_id
        self.global_ids = global_ids
        self.data = data
        self.config = config
        self.global_template = global_template
        self._device_data_cache: Dict[str, Data] = {}

    def new_model(self) -> MMFGUModel:
        return make_model(self.config, self.global_template).to(self.config.device)

    def device_data(self) -> Data:
        device = self.config.device
        if device not in self._device_data_cache:
            self._device_data_cache[device] = to_device_data(self.data, device)
        return self._device_data_cache[device]

    def supervised_train(
        self,
        global_state: Dict[str, torch.Tensor],
        local_epochs: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        model = self.new_model()
        load_model_state(model, global_state, self.config.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        data = self.device_data()
        train_epochs = self.config.local_epochs if local_epochs is None else local_epochs

        model.train()
        for _ in range(train_epochs):
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = masked_cross_entropy(logits, data.y, data.train_mask)
            loss.backward()
            optimizer.step()

        return model_state_to_cpu(model)

    def guided_train(
        self,
        global_state: Dict[str, torch.Tensor],
        degraded_state: Dict[str, torch.Tensor],
        local_epochs: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        model = self.new_model()
        degraded_model = self.new_model()
        load_model_state(model, global_state, self.config.device)
        load_model_state(degraded_model, degraded_state, self.config.device)

        degraded_model.eval()
        for param in degraded_model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        data = self.device_data()
        train_epochs = self.config.local_epochs if local_epochs is None else local_epochs

        model.train()
        for _ in range(train_epochs):
            optimizer.zero_grad()
            logits, _ = model(data)
            with torch.no_grad():
                teacher_logits, _ = degraded_model(data)
                pseudo_labels = teacher_logits.argmax(dim=-1)
            loss = masked_cross_entropy(logits, pseudo_labels, data.train_mask)
            loss.backward()
            optimizer.step()

        return model_state_to_cpu(model)

    def evaluate(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        model = self.new_model()
        load_model_state(model, state_dict, self.config.device)
        model.eval()

        data = self.device_data()
        with torch.no_grad():
            logits, _ = model(data)

        return {
            "train_acc": accuracy(logits, data.y, data.train_mask),
            "val_acc": accuracy(logits, data.y, data.val_mask),
            "test_acc": accuracy(logits, data.y, data.test_mask),
        }

    def evaluate_pseudo_agreement(
        self,
        global_state: Dict[str, torch.Tensor],
        degraded_state: Dict[str, torch.Tensor],
    ) -> float:
        model = self.new_model()
        degraded_model = self.new_model()
        load_model_state(model, global_state, self.config.device)
        load_model_state(degraded_model, degraded_state, self.config.device)
        model.eval()
        degraded_model.eval()
        data = self.device_data()
        with torch.no_grad():
            logits, _ = model(data)
            degraded_logits, _ = degraded_model(data)
        mask = data.train_mask
        if int(mask.sum().item()) == 0:
            return 0.0
        preds = logits[mask].argmax(dim=-1)
        pseudo = degraded_logits[mask].argmax(dim=-1)
        return float((preds == pseudo).float().mean().item())
