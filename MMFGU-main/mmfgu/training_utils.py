# 导入必要的库
import random  # 用于生成随机数
from typing import Dict, List, Optional  # 类型提示

import torch  # PyTorch 核心库
import torch.nn.functional as F  # PyTorch 函数库
from torch_geometric.data import Data  # PyTorch Geometric 的数据结构
from torch_geometric.utils import k_hop_subgraph  # 用于提取 k 跳子图

from .model import MMFGUModel  # 导入模型类
# -----------------------------
# 这一部分是训练过程中常用的小工具函数。
# 它们不直接负责“联邦流程调度”，
# 但会被 client.py / server.py 频繁调用。
# -----------------------------

def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """带 mask 的交叉熵。

    只有 mask=True 的节点会参与损失计算。
    在这里主要用于只在 train_mask 上训练。
    """
    # 检查是否有节点需要计算损失
    if mask.sum() == 0:
        # 如果没有节点需要计算，返回 0
        return logits.sum() * 0.0
    # 计算带掩码的交叉熵损失
    return F.cross_entropy(logits[mask], labels[mask])

def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """带 mask 的准确率。"""
    # 检查是否有节点需要计算准确率
    if mask.sum() == 0:
        # 如果没有节点需要计算，返回 0
        return 0.0
    # 计算预测结果
    pred = logits[mask].argmax(dim=-1)
    # 计算准确率
    return (pred == labels[mask]).float().mean().item()

#算原型
def relation_prototype(model: MMFGUModel, data: Data) -> torch.Tensor:
    """计算客户端当前的 relation prototype。

    这里的做法很直接：
    把所有节点的 relation 表征求平均，作为当前客户端的“中心表示”。
    服务器后面会用它来判断哪些客户端彼此相似。
    """
    # 设置模型为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 前向传播，获取缓存
        _, cache = model(data)  #把客户端的局部图 data 输入模型
    # 计算所有节点的 relation 表征的平均值
    return cache["relation_h"].mean(dim=0).detach().cpu()
      #return cache["relation_h"].cpu().mean(dim=0).detach()


def average_relation_activation(
    model: MMFGUModel, data: Data, node_ids: Optional[torch.Tensor] = None
) -> float:
    """统计 relation 表征的平均激活强度。

    这里用的是向量范数的平均值。
    如果传入 node_ids，就只统计指定节点。
    常用于观察被遗忘节点在遗忘前后是否被“弱化”。
    """
    # 设置模型为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 前向传播，获取缓存
        _, cache = model(data)
        # 获取 relation 表征
        relation_h = cache["relation_h"]
        # 如果指定了节点 ID，只统计这些节点
        if node_ids is not None and node_ids.numel() > 0:
            relation_h = relation_h[node_ids]
        # 计算向量范数的平均值
        return float(relation_h.norm(dim=-1).mean().item())


def probe_alignment_mse(
    model_a: MMFGUModel, model_b: MMFGUModel, probes: List[Data], device: str
) -> float:
    """比较两个模型在 probe 图上的输出差异。

    返回的是所有 probe 上 logits 的平均 MSE。
    值越大，说明两个模型在这些 probe 上差异越明显。
    """
    # 检查是否有 probe
    if not probes:
        return 0.0

    total = 0.0
    # 设置模型为评估模式
    model_a.eval()
    model_b.eval()
    # 遍历所有 probe
    for probe in probes:
        # 将 probe 移至指定设备
        probe = probe.to(device)
        # 禁用梯度计算
        with torch.no_grad():
            # 前向传播，获取 logits
            logits_a, _ = model_a(probe)
            logits_b, _ = model_b(probe)
        # 计算均方误差并累加
        total += F.mse_loss(logits_a, logits_b).item()
    # 返回平均均方误差
    return float(total / len(probes))


def parameter_group_norm(
    local_state: Dict[str, torch.Tensor], #客户端本地训练后的参数
    old_state: Dict[str, torch.Tensor],  #训练前的参数
    prefix: str, #只看参数里面以这个前缀开头的
) -> float:
    """计算一组参数的更新幅度。

    结果可以理解成：该模块这次本地训练改动了多少。
    """
    total = 0.0
    # 遍历所有参数
    for key, value in local_state.items():
        # 检查参数是否属于指定组
        if key.startswith(prefix):
            # 计算参数差异
            diff = value.float() - old_state[key].float()
            # 计算差异的平方和
            total += diff.pow(2).sum().item()
    # 返回均方根
    return float(total**0.5)


def sample_mismatch_nodes(
    num_nodes: int, nodes: torch.Tensor, seed: int
) -> torch.Tensor:
    # num_nodes：总共有多少个节点（比如1000）
    # nodes：你要处理的节点列表（比如[5, 3, 8]）
    """为每个目标节点随机采一个“不等于自己”的错配节点。"""
    # 初始化随机数生成器
    rng = random.Random(seed)
    result = []
    # 遍历每个目标节点
    for node in nodes.tolist():
        # 随机选择一个候选节点
        candidate = rng.randrange(num_nodes)
        # 确保候选节点不等于目标节点
        if num_nodes > 1:
            while candidate == node:
                candidate = rng.randrange(num_nodes)
        # 添加到结果列表
        result.append(candidate)
    # 转换为张量并返回
    return torch.tensor(result, dtype=torch.long)
#  tensor([8, 7]) 只存错配的对方节点


def boundary_nodes(
    edge_index: torch.Tensor, target_nodes: torch.Tensor
) -> torch.Tensor:
    """找出目标节点的一阶边界邻居。

    这些节点虽然不是被遗忘节点本身，
    但因为和它们直接相连，也容易受到影响。
    """
    # 转换目标节点为集合，方便查找
    target_set = set(target_nodes.tolist())
    collected = set()  # 存储边界节点
    # 获取边的源节点和目标节点
    row, col = edge_index
    # 遍历所有边
    for src, dst in zip(row.tolist(), col.tolist()):
        # 如果源节点是目标节点，添加目标节点到边界节点
        if src in target_set:
            collected.add(dst)
        # 如果目标节点是目标节点，添加源节点到边界节点
        if dst in target_set:
            collected.add(src)
    collected.difference_update(target_set)
    # 转换为张量并返回
    return torch.tensor(sorted(collected), dtype=torch.long)


# 一个客户端中所有待遗忘节点会一次性输入这个函数。
def build_probe_graphs(
    client_data: Data, forget_nodes: torch.Tensor, probe_count: int, seed: int
) -> List[Data]:
    """围绕被遗忘节点构造 probe 子图。

    probe 的作用：
    - 模拟局部结构扰动
    - 观察模型对这些扰动是否敏感
    - 选出最敏感的 probe，再做一轮修复

    当前实现里做了两种简单扰动：
    1. 替换中心节点的文本特征
    2. 删除一条边，并给文本加一点噪声
    """
    # 初始化随机数生成器
    rng = random.Random(seed)
    probes = []  # 存储 probe 图

    # 如果没有待遗忘节点，就不需要构造 probe
    if forget_nodes.numel() == 0:
        return probes

    # 一共构造 probe_count 个局部 probe 图
    for idx in range(probe_count):
        # 轮流以待遗忘节点作为中心节点，避免 probe 全都围绕同一个点
        center = int(forget_nodes[idx % forget_nodes.numel()].item())

        # 提取中心节点的一跳子图，作为局部 probe 的基础结构
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            center, 1, client_data.edge_index, relabel_nodes=True
        )

        # 用这个局部子图构造一个独立的 probe 样本
        if hasattr(client_data, "x"):
            probe = Data(
                x=client_data.x[subset].clone(),
                edge_index=sub_edge_index.clone(),
                y=client_data.y[subset].clone(),
                num_nodes=subset.numel(),
            )
        else:
            probe = Data(
                image_x=client_data.image_x[subset].clone(),  # 图像特征
                text_x=client_data.text_x[subset].clone(),  # 文本特征
                edge_index=sub_edge_index.clone(),  # 边索引
                y=client_data.y[subset].clone(),  # 标签
                num_nodes=subset.numel(),  # 节点数量
            )
        # 这个局部子图的文本图像对是一致的
        # 找到中心节点在子图中的局部编号，后面要对它做定向扰动
        local_center = int((subset == center).nonzero(as_tuple=False)[0].item())

        if idx % 2 == 0:
            # 第一类 probe：保留图结构，只把中心节点文本特征替换成别的节点文本，
            # 模拟“图文关系被错配”的情况
            # 随机选择一个替换节点
            replacement = rng.randrange(client_data.num_nodes)
            # 确保替换节点不等于中心节点
            if client_data.num_nodes > 1:
                while replacement == center:
                    replacement = rng.randrange(client_data.num_nodes)
            # 替换文本特征
            if hasattr(client_data, "x"):
                probe.x[local_center] = client_data.x[replacement].clone()
            else:
                probe.text_x[local_center] = client_data.text_x[replacement].clone()
        else:
            # 第二类 probe：对局部结构和文本同时做轻微扰动，
            # 模拟 relation 已沿图传播留下残留后的局部不稳定情况
            # 如果边数大于 1，随机删除一条边
            if probe.edge_index.size(1) > 1:
                keep = torch.ones(probe.edge_index.size(1), dtype=torch.bool)

                # edge_index = [
                #     [src1, src2, src3, ...],
                #     [dst1, dst2, dst3, ...]
                # ]
                # 随机删掉一条边（随机选择一列为false），制造局部结构扰动
                keep[rng.randrange(probe.edge_index.size(1))] = False
                probe.edge_index = probe.edge_index[:, keep]

            # 给中心节点文本特征加小噪声，制造轻微语义扰动
            if hasattr(client_data, "x"):
                probe.x[local_center] = (
                    probe.x[local_center]
                    + torch.randn_like(probe.x[local_center]) * 0.05
                )
            else:
                probe.text_x[local_center] = (
                    probe.text_x[local_center]
                    + torch.randn_like(probe.text_x[local_center]) * 0.05
                )

        # 添加到 probe 列表
        probes.append(probe)

    return probes
