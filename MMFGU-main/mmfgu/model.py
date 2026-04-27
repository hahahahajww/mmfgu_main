# 导入必要的库
from typing import Dict, Optional, Tuple  # 类型提示

import torch  # PyTorch 核心库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 函数库
from torch_geometric.data import Data  # PyTorch Geometric 的数据结构
from torch_geometric.nn import GATConv, GCNConv, SAGEConv  # 图神经网络卷积层

from .config import Config  # 导入配置类


# 融合类
class RelationFusion(nn.Module):
    """把图像模态和文本模态融合成 relation 表征。
    直接使用预提取的 CLIP 特征（768维），不做额外编码。
    """

    def __init__(self, clip_dim: int, hidden_dim: int, dropout: float):
        """初始化关系融合模块。"""
        super().__init__()  # 调用父类初始化
        # 创建融合网络
        self.net = nn.Sequential(
            nn.Linear(clip_dim * 2, hidden_dim),  # 线性层，输入为图像和文本特征的拼接
            nn.ReLU(),  # ReLU 激活函数
            nn.Dropout(dropout),  # Dropout 层，防止过拟合
            nn.Linear(hidden_dim, hidden_dim),  # 线性层，输出隐藏维度
        )

    def forward(self, image_h: torch.Tensor, text_h: torch.Tensor) -> torch.Tensor:
        """前向传播，融合图像和文本特征。"""
        # 拼接图像和文本特征，然后通过融合网络
        return self.net(torch.cat([image_h, text_h], dim=-1))


class FusedFeatureEncoder(nn.Module):
    """把单路融合特征投影到图传播所需的隐藏空间。"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphModule(nn.Module):
    """图传播模块。
    作用是把 relation 表征沿图结构传播，让节点吸收邻居信息。
    """

    def __init__(self, hidden_dim: int, gnn_type: str, dropout: float):
        """初始化图传播模块。"""
        super().__init__()  # 调用父类初始化

        # 根据指定的 GNN 类型创建卷积层
        if gnn_type == "gcn":
            # 使用 GCN 卷积层
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == "gat":
            # 使用 GAT 卷积层，多头注意力
            self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        else:
            # 默认使用 GraphSAGE
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # 保存 dropout 率
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播，在图上传播特征。"""
        # 第一层卷积
        x = self.conv1(x, edge_index)
        # ReLU 激活
        x = F.relu(x)
        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第二层卷积
        return self.conv2(x, edge_index)


class MMFGUModel(nn.Module):
    """完整模型：直接使用预提取 CLIP 特征 + 融合 + 图传播 + 分类。"""

    def __init__(
        self,
        clip_dim: int,  # CLIP 特征维度
        hidden_dim: int,  # 隐藏层维度
        num_classes: int,  # 类别数量
        dropout: float,  # Dropout 率
        gnn_type: str,  # GNN 类型
        fused_input_dim: Optional[int] = None,
    ):
        """初始化完整模型。"""
        super().__init__()  # 调用父类初始化
        # 保存 CLIP 特征维度
        self.clip_dim = clip_dim
        self.fused_input_dim = fused_input_dim
        # 初始化关系融合模块
        self.relation_fusion = RelationFusion(clip_dim, hidden_dim, dropout)
        self.fused_encoder = (
            FusedFeatureEncoder(fused_input_dim, hidden_dim, dropout)
            if fused_input_dim is not None
            else None
        )
        # 初始化图传播模块
        self.graph_module = GraphModule(hidden_dim, gnn_type, dropout)
        # 初始化分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播，执行完整的模型计算。"""
        if hasattr(data, "x") and self.fused_encoder is not None:
            fused_h = data.x
            relation_h = self.fused_encoder(fused_h)
            image_h = fused_h
            text_h = fused_h
        else:
            # 直接使用预提取的 CLIP 特征（768维）
            image_h = data.image_x  # 图像特征
            text_h = data.text_x  # 文本特征
            # 融合成 relation 表征
            relation_h = self.relation_fusion(image_h, text_h)

        # 在图上传播
        propagated_h = self.graph_module(relation_h, data.edge_index)

        # 节点分类
        logits = self.classifier(propagated_h)

        # 返回分类结果和中间特征
        return logits, {
            "image_h": image_h,  # 图像特征
            "text_h": text_h,  # 文本特征
            "relation_h": relation_h,  # 关系表征
            "propagated_h": propagated_h,  # 传播后的特征
        }

    def encode(self, data: Data) -> Dict[str, torch.Tensor]:
        """返回中间表征，供链接预测等任务复用。"""
        _, cache = self(data)
        return cache

    def score_pairs(
        self, node_h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """用点积给节点对打分。"""
        return (node_h[src] * node_h[dst]).sum(dim=-1)


def make_model(config: Config, global_data: Data) -> MMFGUModel:
    """根据预提取的 CLIP 特征维度创建模型。"""
    has_fused_x = hasattr(global_data, "x")
    clip_dim = global_data.image_x.size(1) if hasattr(global_data, "image_x") else 1
    fused_input_dim = global_data.x.size(1) if has_fused_x else None
    # 创建并返回模型实例
    return MMFGUModel(
        clip_dim=clip_dim,  # CLIP 特征维度
        hidden_dim=config.hidden_dim,  # 隐藏层维度
        num_classes=int(global_data.y.max().item() + 1),  # 类别数量
        dropout=config.dropout,  # Dropout 率
        gnn_type=config.gnn_type,  # GNN 类型
        fused_input_dim=fused_input_dim,
    )
