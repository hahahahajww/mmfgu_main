# 导入必要的库
import copy  # 用于深拷贝对象
import os
import random  # 用于生成随机数
from typing import Dict  # 类型提示

import numpy as np  # NumPy 库，用于数值计算
import torch  # PyTorch 核心库
from torch_geometric.data import Data  # PyTorch Geometric 的数据结构


# -----------------------------
# 这一部分是跨模块通用的基础工具函数。
# 基本都是“哪里都可能用到的小功能”。
# -----------------------------


def set_seed(seed: int) -> None:
    """固定随机种子，尽量让实验结果可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUDA 矩阵乘法的确定性要求在导入后设置环境变量也通常可生效。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 固定 Python 内置随机数种子
    random.seed(seed)
    # 固定 NumPy 随机数种子
    np.random.seed(seed)
    # 固定 PyTorch 随机数种子
    torch.manual_seed(seed)
    # 固定所有 CUDA 设备的随机数种子
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # 老版本 PyTorch 或个别环境下不支持时，至少保留上面的固定项。
        pass


def to_device_data(data: Data, device: str) -> Data:
    """把 PyG 的 Data 对象整体搬到指定设备上。

    这里不会原地修改原对象，而是复制出一个新的 Data。
    这样做更安全，避免不同流程互相污染数据。
    """
    # 创建新的 Data 对象
    copied = Data()
    # 遍历原 Data 对象的所有属性
    for key, value in data.to_dict().items():
        # 如果是张量，移至指定设备；否则深拷贝
        copied[key] = (
            value.to(device) if torch.is_tensor(value) else copy.deepcopy(value)
        )
    # 返回新的 Data 对象
    return copied


def model_state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """把模型参数提取成 CPU 上的 state_dict。
    联邦学习里客户端和服务器经常交换参数，
    统一转成 CPU tensor 更方便保存和聚合。
    """
    # 提取模型状态字典，并将所有参数移至 CPU
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], device: str
) -> None:
    """把 state_dict 加载回模型，并移动到指定设备。"""
    # 将状态字典中的参数移至指定设备，然后加载到模型
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
