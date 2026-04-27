from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from .config import Config
from .model import ReGEnUnlearnModel
from .utils import accuracy, load_model_state, masked_cross_entropy, to_device_data


@dataclass
class SamplingResult:
    selected_nodes: torch.Tensor
    sampled_graph: Data
    reward: float
    overlap_proxy: float
    utility_after_ascent: float


class RFPSampler(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def score(self, data: Data) -> torch.Tensor:
        degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
        degree = torch.log1p(degree).unsqueeze(-1)
        feats = torch.cat([data.image_x, data.text_x, degree], dim=-1)
        return self.scorer(feats).squeeze(-1)

    def sample_mask(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(data)
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        sampled = torch.bernoulli(probs)
        if int(sampled.sum().item()) == 0:
            sampled[torch.argmax(probs)] = 1.0
        return sampled.bool(), probs


def _induced_subgraph(data: Data, chosen_nodes: torch.Tensor, hops: int) -> Data:
    if chosen_nodes.numel() == 0:
        chosen_nodes = torch.tensor([0], dtype=torch.long)
    subset, edge_index, _, _ = k_hop_subgraph(
        chosen_nodes.unique(), hops, data.edge_index, relabel_nodes=True
    )
    local = Data(
        image_x=data.image_x[subset].clone(),
        text_x=data.text_x[subset].clone(),
        edge_index=edge_index.clone(),
        y=data.y[subset].clone(),
        train_mask=data.train_mask[subset].clone(),
        val_mask=data.val_mask[subset].clone(),
        test_mask=data.test_mask[subset].clone(),
        num_nodes=subset.numel(),
        original_node_ids=subset.clone(),
    )
    return local


def optimize_sampler(
    sampler: RFPSampler,
    config: Config,
    client_data: Data,
    model_factory,
    global_state: dict[str, torch.Tensor],
    remaining_prototypes: list[torch.Tensor],
) -> SamplingResult:
    optimizer = torch.optim.Adam(sampler.parameters(), lr=config.sampler_lr)
    base_model: ReGEnUnlearnModel = model_factory()
    load_model_state(base_model, global_state, config.device)
    base_model.eval()
    base_device_data = to_device_data(client_data, config.device)
    with torch.no_grad():
        logits, cache = base_model(base_device_data)
        acc0 = accuracy(logits, base_device_data.y, base_device_data.val_mask)
        relation_h = cache["relation_h"].detach().cpu()

    best: SamplingResult | None = None
    for step in range(config.sampler_steps):
        optimizer.zero_grad()
        mask, probs = sampler.sample_mask(client_data)
        selected = mask.nonzero(as_tuple=False).view(-1)
        target_k = max(1, int(config.sampling_rate * client_data.num_nodes))
        if selected.numel() > target_k:
            top_idx = torch.topk(probs[selected], k=target_k).indices
            selected = selected[top_idx]
        elif selected.numel() < target_k:
            filler = torch.topk(probs, k=target_k).indices
            selected = torch.unique(torch.cat([selected, filler], dim=0))[:target_k]

        sampled_graph = _induced_subgraph(client_data, selected.cpu(), config.sampler_hops)

        overlap_proxy = 0.0
        if remaining_prototypes and selected.numel() > 0:
            sampled_proto = relation_h[selected.cpu()].mean(dim=0, keepdim=True)
            sims = [
                float(F.cosine_similarity(sampled_proto, proto.view(1, -1), dim=-1).item())
                for proto in remaining_prototypes
            ]
            overlap_proxy = max(sims)

        temp_model: ReGEnUnlearnModel = model_factory()
        load_model_state(temp_model, global_state, config.device)
        temp_model.train()
        temp_opt = torch.optim.Adam(temp_model.parameters(), lr=config.lr)
        sub_device = to_device_data(sampled_graph, config.device)
        temp_opt.zero_grad()
        sub_logits, _ = temp_model(sub_device)
        ascent = -masked_cross_entropy(sub_logits, sub_device.y, sub_device.train_mask)
        ascent.backward()
        temp_opt.step()

        temp_model.eval()
        with torch.no_grad():
            logits1, _ = temp_model(base_device_data)
            acc1 = accuracy(logits1, base_device_data.y, base_device_data.val_mask)

        reward = 1.0 / (max(0.0, acc0 - acc1) + 1.0) - config.overlap_penalty_weight * overlap_proxy
        log_prob = (
            torch.log(probs[mask]).sum() + torch.log1p(-probs[~mask]).sum()
            if (~mask).any()
            else torch.log(probs[mask]).sum()
        )
        loss = -reward * log_prob
        loss.backward()
        optimizer.step()

        candidate = SamplingResult(
            selected_nodes=selected.cpu(),
            sampled_graph=sampled_graph,
            reward=float(reward),
            overlap_proxy=float(overlap_proxy),
            utility_after_ascent=float(acc1),
        )
        if best is None or candidate.reward > best.reward:
            best = candidate

    assert best is not None
    return best


def build_prompt_enhanced_graph(
    config: Config,
    target_graph: Data,
    model: ReGEnUnlearnModel,
) -> tuple[Data, dict[str, float]]:
    device_graph = to_device_data(target_graph, next(model.parameters()).device.type)
    model.eval()
    with torch.no_grad():
        _, cache = model(device_graph)
    relation_h = cache["relation_h"].detach().cpu()

    token_count = min(config.prompt_token_count, target_graph.num_nodes)
    token_scores = relation_h.norm(dim=-1)
    token_ids = torch.topk(token_scores, k=max(1, token_count)).indices
    tokens = torch.cat(
        [target_graph.image_x[token_ids], target_graph.text_x[token_ids]], dim=-1
    )
    tokens = F.normalize(tokens, dim=-1)
    sim = torch.sigmoid(tokens @ tokens.t())
    adj = (sim > config.prompt_similarity_threshold).float()
    adj.fill_diagonal_(1.0)
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    h = tokens.clone()
    for _ in range(config.prompt_message_passing_steps):
        h = adj @ h / deg
    prompt_vector = h.mean(dim=0)

    enhanced = Data()
    for key, value in target_graph.to_dict().items():
        enhanced[key] = value.clone() if torch.is_tensor(value) else value

    insert_count = max(1, int(config.prompt_insertion_ratio * target_graph.num_nodes))
    insert_ids = torch.topk(token_scores, k=min(insert_count, target_graph.num_nodes)).indices
    split = target_graph.image_x.size(1)
    enhanced.image_x[insert_ids] = enhanced.image_x[insert_ids] + prompt_vector[:split]
    enhanced.text_x[insert_ids] = enhanced.text_x[insert_ids] + prompt_vector[split:]
    return enhanced, {
        "token_count": int(token_ids.numel()),
        "insert_count": int(insert_ids.numel()),
        "prompt_norm": float(prompt_vector.norm().item()),
    }


def prototype_from_model(model: ReGEnUnlearnModel, data: Data, device: str) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, cache = model(to_device_data(data, device))
    return cache["relation_h"].mean(dim=0).detach().cpu()
