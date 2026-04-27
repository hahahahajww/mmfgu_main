# 导入必要的库
import copy  # 用于深拷贝对象
from typing import Dict, List, Tuple, Any  # 类型提示

import numpy as np
import torch  # PyTorch 核心库
from torch_geometric.data import Data  # PyTorch Geometric 的数据结构
from torch_geometric.utils import subgraph  # 用于子图操作

from .config import Config  # 导入配置类
from .model import MMFGUModel, make_model  # 导入模型相关类和函数
from .training_utils import (
    accuracy,  # 计算准确率
    average_relation_activation,  # 计算关系激活的平均值
    boundary_nodes,  # 计算边界节点
    build_probe_graphs,  # 构建探针图
    masked_cross_entropy,  # 带掩码的交叉熵损失
    parameter_group_norm,  # 计算参数组的范数
    probe_alignment_mse,  # 计算探针对齐的均方误差
    relation_prototype,  # 计算关系原型
    sample_mismatch_nodes,  # 采样不匹配的节点
)
from .utils import load_model_state, model_state_to_cpu, to_device_data  # 工具函数


class FederatedClient:
    """联邦学习里的客户端。
    每个客户端持有：
    - 自己的局部子图
    - 自己的本地训练流程
    - 本地遗忘流程
    - 净化训练流程
    """

    def __init__(
        self,
        client_id: int,  # 客户端ID
        global_ids: torch.Tensor,  # 全局节点ID
        data: Data,  # 客户端本地数据
        config: Config,  # 配置对象
        global_template: Data,  # 全局数据模板
    ):
        # 初始化客户端属性
        self.client_id = client_id  # 客户端ID
        self.global_ids = global_ids  # 全局节点ID
        self.data = data  # 客户端本地数据
        self.config = config  # 配置对象
        self.global_template = global_template  # 全局数据模板

        # 这里记录客户端在训练过程中的一些历史信息，
        # 例如 prototype、模块更新幅度等。
        self.history: List[Dict[str, torch.Tensor]] = []  # 历史记录列表
        self._device_data_cache: Dict[str, Data] = {}

    def new_model(self) -> MMFGUModel:
        """每次需要训练/评估时，重新创建一个同结构模型。"""
        # 创建新模型并移至指定设备
        return make_model(self.config, self.global_template).to(self.config.device)

    def has_lp_split(self) -> bool:
        return hasattr(self.data, "lp_train_source_node")

    def _clone_local_data(self) -> Data:
        local = Data()
        for key, value in self.data.to_dict().items():
            local[key] = (
                value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            )
        return local

    def device_data(self) -> Data:
        """缓存本地数据到目标设备，避免每轮重复搬运。"""
        device = self.config.device
        if device not in self._device_data_cache:
            self._device_data_cache[device] = to_device_data(self.data, device)
        return self._device_data_cache[device]

    def sample_forget_edges(self) -> torch.Tensor:
        train_edges = self.data.lp_train_source_node
        if train_edges.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        num_forget = max(1, int(train_edges.numel() * self.config.forget_ratio))
        generator = torch.Generator().manual_seed(self.config.seed + self.client_id)
        perm = torch.randperm(train_edges.numel(), generator=generator)[:num_forget]
        return perm

    def _link_prediction_loss(
        self, model: MMFGUModel, data: Data, src: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        if src.numel() == 0:
            return sum(p.sum() for p in model.parameters()) * 0.0
        cache = model.encode(data)
        node_h = cache["propagated_h"]
        batch_size = max(1, self.config.batch_size)
        losses = []
        for start in range(0, src.size(0), batch_size):
            batch_src = src[start : start + batch_size]
            batch_pos = pos[start : start + batch_size]
            neg = torch.randint(0, data.num_nodes, batch_pos.shape, device=src.device)
            pos_score = model.score_pairs(node_h, batch_src, batch_pos)
            neg_score = model.score_pairs(node_h, batch_src, neg)
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.cat([pos_score, neg_score], dim=0),
                    torch.cat(
                        [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
                    ),
                )
            )
        return torch.stack(losses).mean()

    def _edge_repr(
        self, node_h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        return node_h[src] * node_h[dst]

    def _batched_auc_from_scores(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ) -> float:
        if pos_score.numel() == 0 or neg_score.numel() == 0:
            return 0.0
        pos_expand = pos_score.unsqueeze(1)
        auc = (
            (pos_expand > neg_score).float()
            + 0.5 * (pos_expand == neg_score).float()
        ).mean(dim=1)
        return float(auc.mean().item())

    def _evaluate_lp_part(
        self, model: MMFGUModel, data: Data, part: str
    ) -> dict[str, float]:
        src_all = data[f"lp_{part}_source_node"]
        pos_all = data[f"lp_{part}_target_node"]
        if src_all.numel() == 0:
            return {"mrr": 0.0, "hits@3": 0.0, "hits@10": 0.0, "auc_roc": 0.0}
        cache = model.encode(data)
        node_h = cache["propagated_h"]
        batch_size = max(1, self.config.batch_size)
        if part == "train":
            auc_parts = []
            rank_parts = []
            for start in range(0, src_all.size(0), batch_size):
                src = src_all[start : start + batch_size]
                pos = pos_all[start : start + batch_size]
                neg = torch.randint(0, data.num_nodes, pos.shape, device=node_h.device)
                pos_score = model.score_pairs(node_h, src, pos)
                neg_score = model.score_pairs(node_h, src, neg)
                rank_parts.append(
                    1 + (neg_score >= pos_score).float()
                )
                auc_parts.append(self._batched_auc_from_scores(pos_score, neg_score.unsqueeze(1)))
            rank = torch.cat(rank_parts, dim=0) if rank_parts else torch.empty(0)
            return {
                "mrr": float((1.0 / rank).mean().item()) if rank.numel() > 0 else 0.0,
                "hits@3": float((rank <= 3).float().mean().item()) if rank.numel() > 0 else 0.0,
                "hits@10": float((rank <= 10).float().mean().item()) if rank.numel() > 0 else 0.0,
                "auc_roc": float(np.mean(auc_parts)) if auc_parts else 0.0,
            }

        neg_all = data[f"lp_{part}_target_node_neg"]
        ranks = []
        auc_scores = []
        for start in range(0, src_all.size(0), batch_size):
            src = src_all[start : start + batch_size]
            pos = pos_all[start : start + batch_size]
            neg = neg_all[start : start + batch_size]
            pos_score = model.score_pairs(node_h, src, pos)
            expanded_src = src.unsqueeze(1).expand_as(neg)
            neg_score = model.score_pairs(
                node_h, expanded_src.reshape(-1), neg.reshape(-1)
            ).view_as(neg)
            ranks.append(1 + (neg_score >= pos_score.unsqueeze(1)).sum(dim=1).float())
            auc_scores.append(self._batched_auc_from_scores(pos_score, neg_score))
        rank = torch.cat(ranks, dim=0) if ranks else torch.empty(0)
        return {
            "mrr": float((1.0 / rank).mean().item()) if rank.numel() > 0 else 0.0,
            "hits@3": float((rank <= 3).float().mean().item()) if rank.numel() > 0 else 0.0,
            "hits@10": float((rank <= 10).float().mean().item()) if rank.numel() > 0 else 0.0,
            "auc_roc": float(np.mean(auc_scores)) if auc_scores else 0.0,
        }

    def sample_forget_nodes(self) -> torch.Tensor:
        """按固定随机种子从训练集里采样遗忘样本，保证 unlearning/retrain 一致。"""
        if self.config.task == "link_prediction":
            return self.sample_forget_edges()
        # 获取训练节点
        train_nodes = self.data.train_mask.nonzero(as_tuple=False).view(-1)
        # tensor([0, 1, 4, 7, 9, ...])
        # 如果没有训练节点，返回空张量
        if train_nodes.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        # 计算需要遗忘的节点数量
        num_forget = max(1, int(train_nodes.numel() * self.config.forget_ratio))
        # 设置随机种子，确保采样一致性
        generator = torch.Generator().manual_seed(self.config.seed + self.client_id)
        # 随机排列并选择前 num_forget 个节点
        perm = torch.randperm(train_nodes.numel(), generator=generator)[:num_forget]
        return train_nodes[perm]  # 从中随机挑出来的【遗忘节点】

    def build_retain_client(self, forget_nodes: torch.Tensor) -> "FederatedClient":
        """构造物理删除训练样本后的 retrain 客户端。

        不删除节点、不删除边；
        只把 forget_nodes 从训练集 D_train 中移除。
        """
        local = self._clone_local_data()
        if self.config.task == "link_prediction":
            keep_mask = torch.ones(local.lp_train_source_node.size(0), dtype=torch.bool)
            keep_mask[forget_nodes] = False
            local.lp_train_source_node = local.lp_train_source_node[keep_mask]
            local.lp_train_target_node = local.lp_train_target_node[keep_mask]
        else:
            local.train_mask[forget_nodes] = False

        # 返回新的客户端实例
        return FederatedClient(
            self.client_id,  # 客户端ID
            self.global_ids.clone(),  # 复制全局节点ID
            local,  # 新的数据
            self.config,  # 配置对象
            self.global_template,  # 全局数据模板
        )

    def supervised_train(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """客户端本地监督训练。
        流程很标准：
        1. 用服务器下发的全局参数初始化模型
        2. 在本地训练集上做若干轮训练
        3. 返回本地更新后的参数给服务器做聚合
        """
        # 创建新模型
        # 先新建一个空的同结构模型
        # 再把服务器当前的
        # global_state
        # 参数加载进去
        model = self.new_model()
        # 加载全局模型状态
        load_model_state(model, global_state, self.config.device)

        # 初始化优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        # 将数据移至指定设备  客户端本地的图
        data = self.device_data()

        # 开始训练
        model.train()
        # 本地训练指定轮数
        for _ in range(self.config.local_epochs):
            optimizer.zero_grad()
            if self.config.task == "link_prediction":
                loss = self._link_prediction_loss(
                    model, data, data.lp_train_source_node, data.lp_train_target_node
                )
            else:
                logits, _ = model(data)
                loss = masked_cross_entropy(logits, data.y, data.train_mask)
            loss.backward()
            optimizer.step()

        # 将模型状态移至CPU
        state = model_state_to_cpu(model)

        # 记录一些后面会用到的信息：
        # - prototype：客户端当前 relation 空间的整体中心
        # - relation_update_norm / graph_update_norm：参数更新强度
        self.history.append(
            {
                "prototype": relation_prototype(model, data),  # 计算关系原型
                "relation_update_norm": torch.tensor(
                    parameter_group_norm(
                        state, global_state, "relation_fusion"
                    )  # 计算关系融合模块的更新范数
                ),
                "graph_update_norm": torch.tensor(
                    parameter_group_norm(
                        state, global_state, "graph_module"
                    )  # 计算图模块的更新范数
                ),
                # 本地训练后，relation_fusion改了多少
                # 本地训练后，graph_module改了多少
                # 也就是两个模块的参数更新幅度。
            }
        )
        # 返回更新后的模型状态
        # state = 客户端本地训练完后的模型参数字典  model.state_dict()
        return state

    def evaluate(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """用给定参数在本客户端上评估 train/val/test 精度。"""
        # 创建新模型
        model = self.new_model()
        # 加载模型状态
        load_model_state(model, state_dict, self.config.device)
        # 设置为评估模式
        model.eval()

        # 将数据移至指定设备
        data = self.device_data()
        with torch.no_grad():
            if self.config.task == "link_prediction":
                val_metrics = self._evaluate_lp_part(model, data, "valid")
                test_metrics = self._evaluate_lp_part(model, data, "test")
                return {
                    "val_mrr": val_metrics["mrr"],
                    "val_hits@3": val_metrics["hits@3"],
                    "val_hits@10": val_metrics["hits@10"],
                    "val_auc_roc": val_metrics["auc_roc"],
                    "test_mrr": test_metrics["mrr"],
                    "test_hits@3": test_metrics["hits@3"],
                    "test_hits@10": test_metrics["hits@10"],
                    "test_auc_roc": test_metrics["auc_roc"],
                }

            logits, _ = model(data)

        return {
            "train_acc": accuracy(logits, data.y, data.train_mask),
            "val_acc": accuracy(logits, data.y, data.val_mask),
            "test_acc": accuracy(logits, data.y, data.test_mask),
        }

    def evaluate_relation_stats(
        self, state_dict: Dict[str, torch.Tensor], node_ids: torch.Tensor
    ) -> Dict[str, float]:
        """评估指定节点上的 relation 表征强度。

        这里主要用来观察：
        被遗忘节点在遗忘前后，relation 激活是否下降。
        """
        # 创建新模型
        model = self.new_model()
        # 加载模型状态
        load_model_state(model, state_dict, self.config.device)
        # 将数据移至指定设备
        data = self.device_data()
        # 将节点ID移至指定设备
        node_ids = node_ids.to(self.config.device)
        # 返回关系激活统计
        return {
            "relation_activation_mean": average_relation_activation(
                model,
                data,
                node_ids,  # 计算指定节点的平均关系激活
            ),
        }

    def evaluate_prototype(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算给定模型参数在本客户端上的 relation prototype。"""
        # 创建新模型
        model = self.new_model()
        # 加载模型状态
        load_model_state(model, state_dict, self.config.device)
        # 将数据移至指定设备
        data = self.device_data()
        # 返回关系原型
        return relation_prototype(model, data)

    def evaluate_probe_alignment(
        self,
        state_dict: Dict[str, torch.Tensor],  # 学生模型状态
        teacher_state: Dict[str, torch.Tensor],  # 教师模型状态
        probes: List[Data],  # 探针图列表
    ) -> float:
        """比较两个模型在 probe 图上的输出差异。"""
        # 创建学生模型
        model = self.new_model()
        # 创建教师模型
        teacher = self.new_model()
        # 加载学生模型状态
        load_model_state(model, state_dict, self.config.device)
        # 加载教师模型状态
        load_model_state(teacher, teacher_state, self.config.device)
        # 将探针图移至指定设备
        probe_devices = [to_device_data(probe, self.config.device) for probe in probes]
        # 返回探针对齐的均方误差
        return probe_alignment_mse(model, teacher, probe_devices, self.config.device)

    def _local_unlearn_link_prediction(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[Data], Dict[str, Any]]:
        old_model = self.new_model()
        new_model = self.new_model()
        load_model_state(old_model, global_state, self.config.device)
        load_model_state(new_model, global_state, self.config.device)
        for param in old_model.parameters():
            param.requires_grad = False

        data = self.device_data()
        old_model.eval()
        with torch.no_grad():
            old_cache = old_model.encode(data)

        optimizer = torch.optim.Adam(
            new_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        forget_edges = self.sample_forget_edges().to(self.config.device)
        keep_mask = torch.ones(
            data.lp_train_source_node.size(0),
            dtype=torch.bool,
            device=self.config.device,
        )
        keep_mask[forget_edges] = False
        retain_src = data.lp_train_source_node[keep_mask]
        retain_dst = data.lp_train_target_node[keep_mask]
        forget_src = data.lp_train_source_node[forget_edges]
        forget_dst = data.lp_train_target_node[forget_edges]
        if forget_edges.numel() == 0:
            state = model_state_to_cpu(new_model)
            return (
                state,
                [],
                {
                    "forget_edge_count": 0,
                    "forget_edge_ids": [],
                    "forget_node_count": 0,
                    "forget_node_ids": [],
                    "retain_edge_count": int(retain_src.numel()),
                    "boundary_node_count": 0,
                    "selected_probe_count": 0,
                    "relation_activation_before": 0.0,
                    "relation_activation_after": 0.0,
                },
            )
        forget_nodes = torch.unique(torch.cat([forget_src, forget_dst], dim=0))
        bd_nodes = boundary_nodes(data.edge_index.cpu(), forget_nodes.cpu()).to(
            self.config.device
        )
        relation_before = average_relation_activation(old_model, data, forget_nodes)

        for epoch in range(self.config.unlearn_local_epochs):
            optimizer.zero_grad()
            new_cache = new_model.encode(data)
            mismatch = sample_mismatch_nodes(
                data.num_nodes, forget_dst.cpu(), self.config.seed + epoch
            ).to(self.config.device)

            forget_relation = self._edge_repr(
                new_cache["propagated_h"], forget_src, forget_dst
            )
            mismatch_relation = self._edge_repr(
                new_cache["propagated_h"], forget_src, mismatch
            ).detach()
            loss_dec = torch.nn.functional.mse_loss(forget_relation, mismatch_relation)

            anchor_new = self._edge_repr(
                new_cache["propagated_h"], forget_src, mismatch
            )
            anchor_old = self._edge_repr(
                old_cache["propagated_h"], forget_src, mismatch
            )
            loss_anchor = torch.nn.functional.mse_loss(anchor_new, anchor_old)

            if retain_src.numel() > 0:
                retain_new = self._edge_repr(
                    new_cache["propagated_h"], retain_src, retain_dst
                )
                retain_old = self._edge_repr(
                    old_cache["propagated_h"], retain_src, retain_dst
                )
                loss_mm = torch.nn.functional.mse_loss(retain_new, retain_old)
            else:
                loss_mm = loss_dec * 0.0

            if bd_nodes.numel() > 0:
                loss_bd = torch.nn.functional.mse_loss(
                    new_cache["propagated_h"][bd_nodes],
                    old_cache["propagated_h"][bd_nodes],
                )
            else:
                loss_bd = loss_dec * 0.0

            loss = (
                self.config.alpha_dec * loss_dec
                + self.config.alpha_anchor * loss_anchor
                + self.config.beta_mm * loss_mm
                + self.config.delta_bd * loss_bd
            )
            loss.backward()
            optimizer.step()

        probes = build_probe_graphs(
            self.data,
            forget_nodes.detach().cpu(),
            self.config.probe_count,
            self.config.seed + self.client_id,
        )
        state = model_state_to_cpu(new_model)
        relation_after = average_relation_activation(new_model, data, forget_nodes)
        unlearn_info = {
            "forget_edge_count": int(forget_edges.numel()),
            "forget_edge_ids": forget_edges.detach().cpu().tolist(),
            "forget_node_count": int(forget_nodes.numel()),
            "forget_node_ids": forget_nodes.detach().cpu().tolist(),
            "retain_edge_count": int(retain_src.numel()),
            "boundary_node_count": int(bd_nodes.numel()),
            "selected_probe_count": min(len(probes), self.config.probe_topk),
            "relation_activation_before": relation_before,
            "relation_activation_after": relation_after,
        }
        return state, probes[: self.config.probe_topk], unlearn_info

    def _purge_train_link_prediction(
        self,
        global_state: Dict[str, torch.Tensor],
        teacher_state: Dict[str, torch.Tensor],
        probes: List[Data],
    ) -> Dict[str, torch.Tensor]:
        model = self.new_model()
        teacher = self.new_model()
        load_model_state(model, global_state, self.config.device)
        load_model_state(teacher, teacher_state, self.config.device)
        for p in teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        data = self.device_data()

        for _ in range(self.config.purge_local_epochs):
            optimizer.zero_grad()
            loss_pos = self._link_prediction_loss(
                model, data, data.lp_train_source_node, data.lp_train_target_node
            )
            loss_neg = loss_pos * 0.0
            if probes:
                for probe in probes:
                    probe_device = to_device_data(probe, self.config.device)
                    student_cache = model.encode(probe_device)
                    with torch.no_grad():
                        teacher_cache = teacher.encode(probe_device)
                    loss_neg = loss_neg + torch.nn.functional.mse_loss(
                        student_cache["propagated_h"], teacher_cache["propagated_h"]
                    )
                loss_neg = loss_neg / len(probes)
            loss = loss_pos + self.config.lambda_neg * loss_neg
            loss.backward()
            optimizer.step()

        return model_state_to_cpu(model)

    def local_unlearn(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[Data], Dict[str, Any]]:
        """客户端本地执行遗忘。

        这个阶段是整套方法里最关键的一步：
        1. 从当前全局模型出发
        2. 随机抽一部分本地节点作为“要遗忘的目标”
        3. 用多项损失约束去破坏这些节点的旧 relation
        4. 再用 probe 做一轮局部修复

        返回值包括：
        - 遗忘后的本地参数
        - 选中的 probe 图
        - 新的 prototype
        - 一些遗忘统计信息
        """
        if self.config.task == "link_prediction":
            return self._local_unlearn_link_prediction(global_state)

        # 创建旧模型（遗忘前）
        old_model = self.new_model()
        # 创建新模型（遗忘后）
        new_model = self.new_model()
        # 加载全局模型状态到旧模型
        load_model_state(old_model, global_state, self.config.device)
        # 加载全局模型状态到新模型
        load_model_state(new_model, global_state, self.config.device)

        # old_model 作为“遗忘前老师模型”固定不动，只用来提供参照。
        for param in old_model.parameters():
            param.requires_grad = False

        # old_model冻结，不更新
        # new_model会在
        # loss_dec + loss_anchor + loss_mm + loss_bd下不断更新

        # 分类头这里先冻结，重点先更新前面的多模态 relation 部分。
        # 冻结分类头是为了让模型在遗忘时优先改relation
        # 表征，而不是靠最后分类层临时调输出掩盖旧知识。
        for param in new_model.classifier.parameters():
            param.requires_grad = False

        # 将数据移至指定设备
        data = self.device_data()
        # 设置旧模型为评估模式
        old_model.eval()
        # 禁用梯度计算
        with torch.no_grad():
            # 前向传播，获取旧模型的缓存
            _, old_cache = old_model(data)

        # 初始化优化器，只优化需要梯度的参数
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, new_model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # 按固定规则采样遗忘节点，确保和 retrain baseline 使用同一批数据。
        forget_nodes = self.sample_forget_nodes().to(self.config.device)
        # forget_nodes = tensor([5, 8])
        # retain_nodes：没有被遗忘的节点
        retain_nodes = torch.tensor(
            [i for i in range(data.num_nodes) if i not in set(forget_nodes.tolist())],
            dtype=torch.long,
            device=self.config.device,
        )

        # bd_nodes：遗忘节点边界上的邻居节点
        bd_nodes = boundary_nodes(data.edge_index.cpu(), forget_nodes.cpu()).to(
            self.config.device
        )
        # tensor([1, 3, 4, 9, 10])

        # 记录遗忘前 relation 激活，后面用于实验统计
        relation_before = average_relation_activation(old_model, data, forget_nodes)
        # 观察遗忘节点上的relation
        # 表征强度在遗忘前后有没有下降，用来辅助判断本地relationunlearning是否真的生效。

        # 执行本地遗忘训练
        for epoch in range(self.config.unlearn_local_epochs):
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播，获取新模型的缓存
            _, new_cache = new_model(data)

            # mismatch 节点：给遗忘节点重新配一个“不匹配”的文本目标
            mismatch = sample_mismatch_nodes(
                data.num_nodes, forget_nodes.cpu(), self.config.seed + epoch
            ).to(self.config.device)

            # 显式构造本轮会用到的 relation，对三项损失统一写法。
            forget_relation = new_model.relation_fusion(
                new_cache["image_h"][forget_nodes], new_cache["text_h"][forget_nodes]
            )
            mismatch_relation = new_model.relation_fusion(
                new_cache["image_h"][forget_nodes], new_cache["text_h"][mismatch]
            ).detach()
            # L_dec：让原本的 relation 和错配 relation 更接近，相当于打散旧关系
            loss_dec = torch.nn.functional.mse_loss(forget_relation, mismatch_relation)

            anchor_new = new_model.relation_fusion(
                new_cache["image_h"][forget_nodes], new_cache["text_h"][mismatch]
            )
            anchor_old = old_model.relation_fusion(
                old_cache["image_h"][forget_nodes], old_cache["text_h"][mismatch]
            )
            # L_anchor：让“新错配关系”不要偏离旧模型太离谱
            loss_anchor = torch.nn.functional.mse_loss(anchor_new, anchor_old)

            retain_relation_new = (
                new_model.relation_fusion(
                    new_cache["image_h"][retain_nodes],
                    new_cache["text_h"][retain_nodes],
                )
                if retain_nodes.numel() > 0
                else None
            )
            retain_relation_old = (
                old_model.relation_fusion(
                    old_cache["image_h"][retain_nodes],
                    old_cache["text_h"][retain_nodes],
                )
                if retain_nodes.numel() > 0
                else None
            )
            # L_mm：保留节点的 relation 不要被破坏太多
            loss_mm = (
                torch.nn.functional.mse_loss(retain_relation_new, retain_relation_old)
                if retain_nodes.numel()
                > 0  # 避免在 retain_nodes 为空时对空张量算损失，并用一个安全的零损失替代它。
                else loss_dec * 0.0
            )

            # L_bd：边界节点的图传播表示尽量稳定
            loss_bd = (
                torch.nn.functional.mse_loss(
                    new_cache["propagated_h"][bd_nodes],
                    old_cache["propagated_h"][bd_nodes],
                )
                if bd_nodes.numel() > 0
                else loss_dec * 0.0
            )

            # 总损失
            loss = (
                self.config.alpha_dec * loss_dec
                + self.config.alpha_anchor * loss_anchor
                + self.config.beta_mm * loss_mm
                + self.config.delta_bd * loss_bd
            )
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        # 构建 probe 图，用来检测遗忘后哪些局部结构最敏感
        probes = build_probe_graphs(
            self.data,
            forget_nodes.cpu(),
            self.config.probe_count,
            self.config.seed + self.client_id,
        )
        # [Data(...), Data(...), Data(...), ...] 每个Data是image_x，text_x，edge_index，y，num_nodes
        scored = []
        # 计算每个探针图的差异
        for probe in probes:
            probe_device = to_device_data(probe, self.config.device)
            with torch.no_grad():
                old_logits, _ = old_model(probe_device)
                new_logits, _ = new_model(probe_device)
            # 计算均方误差并记录
            scored.append(
                (torch.nn.functional.mse_loss(old_logits, new_logits).item(), probe)
            )
            # 计算「旧模型输出」和「新模型输出」的差异大小（MSE 损失），把这个差异分数和对应的探针一起存起来。

        # 选出差异最大的 top-k probe
        scored.sort(key=lambda x: x[0], reverse=True)
        selected_probes = [probe for _, probe in scored[: self.config.probe_topk]]

        # 再做一轮轻量修复，让模型在这些敏感 probe 上更稳定一些。
        stage2_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, new_model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        # 执行探针修复训练
        for _ in range(self.config.probe_epochs):
            if not selected_probes:
                break
            # 清零梯度
            stage2_optimizer.zero_grad()
            total = 0.0
            # 遍历所有选中的探针图
            for probe in selected_probes:
                probe_device = to_device_data(probe, self.config.device)
                # 前向传播
                logits, _ = new_model(probe_device)

                # 显式构造 t_k：这里采用旧模型在 probe 图上的输出，
                # 把它视作“错配/扰动关系下应有的目标输出”。
                with torch.no_grad():
                    target_logits, _ = old_model(probe_device)

                # 计算 L_probe = ||f'(G_probe) - t_k||^2
                total = total + torch.nn.functional.mse_loss(logits, target_logits)
            # 平均损失
            total = total / len(selected_probes)
            # 反向传播
            total.backward()
            # 更新参数
            stage2_optimizer.step()

        # 将模型状态移至CPU
        state = model_state_to_cpu(new_model)
        # 计算遗忘后的关系激活
        relation_after = average_relation_activation(new_model, data, forget_nodes)

        # 构建遗忘信息字典
        unlearn_info = {
            "forget_node_count": int(forget_nodes.numel()),  # 遗忘节点数量
            "forget_node_ids": forget_nodes.detach().cpu().tolist(),  # 遗忘节点ID
            "retain_node_count": int(retain_nodes.numel()),  # 保留节点数量
            "boundary_node_count": int(bd_nodes.numel()),  # 边界节点数量
            "selected_probe_count": len(selected_probes),  # 选中的探针数量
            "relation_activation_before": relation_before,  # 遗忘前的关系激活
            "relation_activation_after": relation_after,  # 遗忘后的关系激活
        }
        # 返回遗忘后的模型状态、选中的探针图和遗忘信息
        return state, selected_probes, unlearn_info

    def purge_train(
        self,  # 当前受影响客户端 (j)
        global_state: Dict[str, torch.Tensor],  # 全局模型状态
        teacher_state: Dict[str, torch.Tensor],  # 教师模型状态（遗忘后的模型）
        probes: List[Data],  # 探针图列表
    ) -> Dict[str, torch.Tensor]:
        """受影响客户端执行联邦净化训练。

        直观理解：
        - 继续保住自己的节点分类能力（loss_pos）
        - 同时在 probe 上向 requester 的遗忘后模型靠拢（loss_neg）
        """
        if self.config.task == "link_prediction":
            return self._purge_train_link_prediction(
                global_state, teacher_state, probes
            )

        # 创建新模型
        model = self.new_model()
        # 创建教师模型
        teacher = self.new_model()
        # 加载全局模型状态
        load_model_state(model, global_state, self.config.device)
        # 加载教师模型状态
        load_model_state(teacher, teacher_state, self.config.device)

        # 固定教师模型参数
        for p in teacher.parameters():
            p.requires_grad = False

        # 初始化优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        # 将数据移至指定设备
        data = self.device_data()

        # 执行净化训练
        for _ in range(self.config.purge_local_epochs):
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播 得到的预测值
            logits, _ = model(data)  # 客户端自己的本地图数据

            # 正样本：正常节点分类损失
            loss_pos = masked_cross_entropy(
                logits, data.y, data.train_mask
            )  # 只对 train_mask=True 的节点算交叉熵损失

            # 负样本项：在 probe 上向遗忘后的 teacher 对齐
            loss_neg = logits.sum() * 0.0
            if probes:
                for probe in probes:
                    probe_device = to_device_data(probe, self.config.device)
                    # 学生模型前向传播
                    student_logits, _ = model(probe_device)
                    # 教师模型前向传播（禁用梯度）
                    with torch.no_grad():
                        teacher_logits, _ = teacher(probe_device)
                    # 计算均方误差
                    loss_neg = loss_neg + torch.nn.functional.mse_loss(
                        student_logits, teacher_logits
                    )
                # 平均损失
                loss_neg = loss_neg / len(probes)

            # 总损失
            loss = loss_pos + self.config.lambda_neg * loss_neg
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        # 将模型状态移至CPU并返回
        return model_state_to_cpu(model)
