from mmfgu.training_utils import accuracy, masked_cross_entropy
from mmfgu.utils import load_model_state, model_state_to_cpu, set_seed, to_device_data

import torch


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in states[0]:
        stacked = torch.stack([state[key].float() for state in states], dim=0)
        result[key] = stacked.mean(dim=0)
    return result


def state_l2_distance(
    model: torch.nn.Module, guide_state: dict[str, torch.Tensor]
) -> torch.Tensor:
    total = None
    for name, param in model.named_parameters():
        diff = param - guide_state[name].to(param.device)
        value = diff.pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total.sqrt()


__all__ = [
    "accuracy",
    "masked_cross_entropy",
    "load_model_state",
    "model_state_to_cpu",
    "set_seed",
    "to_device_data",
    "average_state_dicts",
    "state_l2_distance",
]
