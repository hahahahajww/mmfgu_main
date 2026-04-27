from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
from torch_geometric.data import Data

from .config import Config
from .model import ReGEnUnlearnModel, make_model
from .modules import RFPSampler, build_prompt_enhanced_graph, optimize_sampler, prototype_from_model
from .utils import accuracy, load_model_state, masked_cross_entropy, model_state_to_cpu, to_device_data


@dataclass
class ClientArtifacts:
    enhanced_graph: Data
    sampled_nodes: torch.Tensor
    info: dict[str, object]


class ReGEnUnlearnClient:
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

    def new_model(self) -> ReGEnUnlearnModel:
        return make_model(self.config, self.global_template).to(self.config.device)

    def supervised_train(
        self,
        global_state: Dict[str, torch.Tensor],
        local_epochs: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        model = self.new_model()
        load_model_state(model, global_state, self.config.device)
        data = to_device_data(self.data, self.config.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        model.train()
        for _ in range(local_epochs or self.config.local_epochs):
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = masked_cross_entropy(logits, data.y, data.train_mask)
            loss.backward()
            optimizer.step()
        return model_state_to_cpu(model)

    def evaluate(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        model = self.new_model()
        load_model_state(model, state_dict, self.config.device)
        model.eval()
        data = to_device_data(self.data, self.config.device)
        with torch.no_grad():
            logits, _ = model(data)
        return {
            "train_acc": accuracy(logits, data.y, data.train_mask),
            "val_acc": accuracy(logits, data.y, data.val_mask),
            "test_acc": accuracy(logits, data.y, data.test_mask),
        }

    def evaluate_prototype(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.new_model()
        load_model_state(model, state_dict, self.config.device)
        return prototype_from_model(model, self.data, self.config.device)

    def build_retain_client(self) -> "ReGEnUnlearnClient":
        return ReGEnUnlearnClient(
            client_id=self.client_id,
            global_ids=self.global_ids.clone(),
            data=copy.deepcopy(self.data),
            config=self.config,
            global_template=self.global_template,
        )

    def build_unlearning_artifacts(
        self,
        global_state: Dict[str, torch.Tensor],
        remaining_prototypes: list[torch.Tensor],
    ) -> ClientArtifacts:
        sampler = RFPSampler(
            feature_dim=self.data.image_x.size(1) + self.data.text_x.size(1),
            hidden_dim=self.config.sampler_hidden_dim,
        )
        sampling = optimize_sampler(
            sampler=sampler,
            config=self.config,
            client_data=self.data,
            model_factory=self.new_model,
            global_state=global_state,
            remaining_prototypes=remaining_prototypes,
        )
        model = self.new_model()
        load_model_state(model, global_state, self.config.device)
        enhanced_graph, prompt_info = build_prompt_enhanced_graph(
            self.config, sampling.sampled_graph, model
        )
        info = {
            "client_id": self.client_id,
            "selected_node_count": int(sampling.selected_nodes.numel()),
            "sampled_subgraph_nodes": int(sampling.sampled_graph.num_nodes),
            "sampler_reward": sampling.reward,
            "overlap_proxy": sampling.overlap_proxy,
            "utility_after_sampler_ascent": sampling.utility_after_ascent,
            "prompt": prompt_info,
        }
        return ClientArtifacts(
            enhanced_graph=enhanced_graph,
            sampled_nodes=sampling.selected_nodes,
            info=info,
        )
