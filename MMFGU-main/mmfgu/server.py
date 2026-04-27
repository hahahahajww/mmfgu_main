# 导入必要的库
import copy  # 用于深拷贝对象
import json  # 用于 JSON 序列化和反序列化
from dataclasses import asdict  # 用于将 dataclass 转换为字典
from pathlib import Path  # 用于处理文件路径
from typing import Dict, List, Sequence  # 类型提示

import numpy as np  # NumPy 库，用于数值计算
import torch  # PyTorch 核心库
import torch.nn.functional as F  # PyTorch 函数库

from .client import FederatedClient  # 导入客户端类
from .config import Config  # 导入配置类
from .data import build_global_graph, split_clients  # 导入数据处理函数
from .model import make_model  # 导入模型创建函数
from .training_utils import build_probe_graphs  # 导入 probe 构造函数
from .utils import model_state_to_cpu  # 导入工具函数


class FederatedServer:
    """联邦学习里的服务器端。

    服务器负责：
    1. 构建全局图并切分客户端
    2. 初始化全局模型
    3. 做联邦预训练聚合
    4. 在遗忘请求到来后协调后续 purge 过程
    5. 保存实验输出
    """

    def __init__(self, config: Config):
        """初始化服务器。"""
        # 保存配置
        self.config = config
        # 全局图只在服务器这里统一构建一次。随机划分节点的训练集，验证集用到随机种子
        self.global_data = build_global_graph(
            Path(config.data_dir), config.seed, config.task
        )

        # 把全局图切成多个客户端子图。  # 存放所有客户端对象，列表里每个元素都是一个 FederatedClient
        self.clients: List[FederatedClient] = []  # 客户端列表
        # 遍历切分后的客户端数据
        for client_id, (global_ids, data) in enumerate(
            split_clients(self.global_data, config.num_clients, config.seed)
        ):
            # client_id = 0
            # global_ids = tensor([0, 1, 2])
            # data = 客户端0的本地小图
            # 创建客户端实例并添加到列表
            self.clients.append(
                FederatedClient(client_id, global_ids, data, config, self.global_data)
            )

        # 初始化全局模型参数。 随机
        # relation_fusion的MLP参数
        # graph_module的GNN参数
        # classifier的线性分类头参数
        self.global_state = model_state_to_cpu(make_model(config, self.global_data))

        # history 保存按轮记录的指标。
        self.history = {"pretrain": [], "purge": []}  # 训练历史

        # experiment_summary 保存最终实验摘要。
        self.experiment_summary: Dict[str, object] = {
            "pretrain_final": None,  # 预训练最终结果
            "unlearning": None,  # 遗忘结果
            "retrain_baseline": None,  # 重训练基线结果
            "client_unlearning": None,  # 客户端遗忘结果
            "client_retrain_baseline": None,  # 客户端重训练基线结果
        }

    def _aggregate_metric_rows(
        self, metrics: List[Dict[str, float]], prefix: str = "avg_"
    ) -> Dict[str, float]:
        if not metrics:
            return {}
        keys = metrics[0].keys()
        return {prefix + key: float(np.mean([m[key] for m in metrics])) for key in keys}

    def _main_metric_names(self) -> tuple[str, str]:
        if self.config.task == "link_prediction":
            return "avg_val_auc_roc", "avg_test_auc_roc"
        return "avg_val_acc", "avg_test_acc"

    def _average_state_dicts(
        self, states: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """对多个客户端参数做逐项平均。

        这是服务器端执行的标准 FedAvg 聚合。
        """
        result = {}
        num_states = len(states)

        # 遍历模型里的每一个参数名，例如 classifier.weight、graph_module.conv1.bias
        for key in states[0]:
            # 先用第一个客户端的参数初始化累加器，避免从全 0 张量手动建形状
            avg = states[0][key].float().clone()

            # 把其余客户端同名参数依次累加
            for state in states[1:]:
                avg += state[key].float()

            # 最后除以客户端数量，得到该参数的联邦平均值
            result[key] = avg / num_states

        return result

    def _should_eval_round(self, round_idx: int, total_rounds: int) -> bool:
        interval = max(1, self.config.eval_interval)
        return ((round_idx + 1) % interval == 0) or (round_idx + 1 == total_rounds)

    def evaluate_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """把一个模型参数拿到所有客户端上做平均评估。"""
        metrics = [client.evaluate(state_dict) for client in self.clients]
        return self._aggregate_metric_rows(metrics)

    def evaluate_client_metrics(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> List[Dict[str, float]]:
        """记录每个客户端单独的最终指标。"""
        rows = []  # 存储每个客户端的指标
        # 遍历所有客户端
        for client in self.clients:
            # 评估客户端
            metrics = client.evaluate(state_dict)
            # 添加客户端 ID 和指标
            rows.append({"client_id": client.client_id, **metrics})
        return rows

    def pretrain(self) -> None:
        """执行联邦预训练。
        每一轮流程：
        1. 服务器把当前全局参数发给所有客户端
        2. 客户端各自本地训练
        3. 服务器把客户端参数做平均（FedAvg）
        4. 记录当前轮的平均指标
        """
        # 执行指定轮数的联邦训练
        for round_idx in range(self.config.federated_rounds):
            # 收集所有客户端的训练结果
            states = [
                client.supervised_train(self.global_state) for client in self.clients
            ]
            # 聚合客户端参数（FedAvg）
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

        # 记录预训练最终结果
        self.experiment_summary["pretrain_final"] = self.evaluate_state(
            self.global_state
        )

    def run_retrain_baseline(self) -> None:
        """在同一批遗忘节点上做从头训练的 retrain 金标准。"""
        retrain_clients = []  # 重训练客户端列表
        forget_info = None  # 遗忘信息
        # 遍历所有客户端
        for client in self.clients:
            # 如果是需要遗忘的客户端
            if client.client_id == self.config.forget_client_id:
                # 采样遗忘节点
                forget_nodes = client.sample_forget_nodes()
                # 构建保留客户端（物理删除遗忘节点）
                retrain_client = client.build_retain_client(forget_nodes)
                # 记录遗忘信息
                forget_info = {
                    "client_id": client.client_id,  # 客户端 ID
                    "forget_pair_count": int(forget_nodes.numel()),  # 遗忘节点数量
                    "forget_node_ids": forget_nodes.tolist(),  # 遗忘节点 ID
                    "retrain_mode": "physical_train_set_removal",  # 重训练模式
                    "retain_train_count": int(
                        retrain_client.data.train_mask.sum().item()  # 保留的训练节点数量
                    ),
                }
                # 添加到重训练客户端列表
                retrain_clients.append(retrain_client)
            else:
                # 其他客户端直接添加
                retrain_clients.append(client)

        # 初始化重训练模型状态
        retrain_state = model_state_to_cpu(make_model(self.config, self.global_data))
        retrain_history = []  # 重训练历史
        # 执行指定轮数的联邦训练
        for round_idx in range(self.config.federated_rounds):
            # 收集所有客户端的训练结果
            states = [
                client.supervised_train(retrain_state) for client in retrain_clients
            ]
            # 聚合客户端参数
            retrain_state = self._average_state_dicts(states)
            if self._should_eval_round(round_idx, self.config.federated_rounds):
                metrics = self._evaluate_clients_state(retrain_clients, retrain_state)
                metrics["round"] = round_idx + 1
                retrain_history.append(metrics)
                val_key, test_key = self._main_metric_names()
                print(
                    f"[Retrain] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}"
                )
            else:
                print(f"[Retrain] round={round_idx + 1} train_only")

        # 更新全局状态和客户端列表
        self.global_state = retrain_state
        self.clients = retrain_clients
        self.history["pretrain"] = retrain_history
        # 记录重训练基线结果
        self.experiment_summary["retrain_baseline"] = {
            "forget_request": forget_info,  # 遗忘请求信息
            "final_metrics": self._evaluate_clients_state(
                retrain_clients,
                retrain_state,  # 最终指标
            ),
        }

    def run_client_unlearning(self) -> None:
        """执行客户端遗忘。

        与 relation unlearning 不同，这里目标客户端直接退出：
        1. 用退出客户端的遗忘前 prototype 识别受影响客户端
        2. 用退出客户端局部图构造 probes
        3. 用随机初始化噪声模型作为 purge teacher
        4. 只在剩余客户端上继续聚合训练
        """

        requester = self.clients[self.config.forget_client_id]  # 遗忘客户端
        remaining_clients = [  # 剩下
            client
            for client in self.clients
            if client.client_id != self.config.forget_client_id
        ]

        before_metrics = self._evaluate_clients_state(
            remaining_clients, self.global_state
        )
        print("Before client unlearning metrics:")
        print(json.dumps(before_metrics, indent=2))

        requester_old_prototype = requester.evaluate_prototype(self.global_state)
        affected, similarity_scores = self.affected_clients(
            self.config.forget_client_id, requester_old_prototype
        )
        print("Prototype similarities:")
        for row in similarity_scores:
            print(f"  client={row['client_id']} similarity={row['similarity']:.4f}")
        print("Affected clients:", affected)

        requester_train_nodes = requester.data.train_mask.nonzero(as_tuple=False).view(
            -1
        )
        if self.config.task == "link_prediction":
            requester_train_nodes = torch.unique(
                torch.cat(
                    [
                        requester.data.lp_train_source_node,
                        requester.data.lp_train_target_node,
                    ],
                    dim=0,
                )
            )
        probes = build_probe_graphs(  # 探针图（伪图）来源：直接用被遗忘客户端自己的本地图裁剪 / 构造
            requester.data,
            requester_train_nodes.cpu(),
            self.config.probe_count,
            self.config.seed + self.config.forget_client_id,
        )

        # 客户端遗忘没有 requester 本地遗忘模型，这里用随机初始化模型作为“已退出客户端”的噪声教师。
        noise_teacher_state = model_state_to_cpu(
            make_model(self.config, self.global_data)
        )

        current_state = copy.deepcopy(self.global_state)
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
                metrics = self._evaluate_clients_state(remaining_clients, current_state)
                metrics["round"] = round_idx + 1
                self.history["purge"].append(metrics)
                val_key, test_key = self._main_metric_names()
                print(
                    f"[ClientPurge] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}"
                )
            else:
                print(f"[ClientPurge] round={round_idx + 1} train_only")

        self.global_state = current_state
        self.clients = remaining_clients

        after_metrics = self.evaluate_state(self.global_state)
        affected_client_metrics = [
            client.evaluate(self.global_state)
            for client in self.clients
            if client.client_id in affected
        ]

        self.experiment_summary["client_unlearning"] = {
            "removed_client_id": self.config.forget_client_id,
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                key + "_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "affected_client_ids": affected,
            "affected_client_count": len(affected),
            "prototype_similarities": similarity_scores,
            "probe_stats": {
                "probe_count": len(probes),
            },
            "affected_client_final_mean_test_metric": float(
                np.mean(
                    [
                        m[
                            "test_auc_roc"
                            if self.config.task == "link_prediction"
                            else "test_acc"
                        ]
                        for m in affected_client_metrics
                    ]
                )
            )
            if affected_client_metrics
            else 0.0,
        }

    def run_client_retrain_baseline(self) -> None:
        """客户端遗忘对应的 retrain baseline。

        直接把目标客户端从联邦系统中移除，
        然后只在剩余客户端上从头联邦训练。
        """

        retained_clients = [
            client
            for client in self.clients
            if client.client_id != self.config.forget_client_id
        ]

        retrain_state = model_state_to_cpu(make_model(self.config, self.global_data))
        retrain_history = []
        for round_idx in range(self.config.federated_rounds):
            states = [
                client.supervised_train(retrain_state) for client in retained_clients
            ]
            retrain_state = self._average_state_dicts(states)
            if self._should_eval_round(round_idx, self.config.federated_rounds):
                metrics = self._evaluate_clients_state(retained_clients, retrain_state)
                metrics["round"] = round_idx + 1
                retrain_history.append(metrics)
                val_key, test_key = self._main_metric_names()
                print(
                    f"[ClientRetrain] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}"
                )
            else:
                print(f"[ClientRetrain] round={round_idx + 1} train_only")

        self.global_state = retrain_state
        self.clients = retained_clients
        self.history["pretrain"] = retrain_history
        self.experiment_summary["client_retrain_baseline"] = {
            "removed_client_id": self.config.forget_client_id,
            "remaining_client_count": len(retained_clients),
            "final_metrics": self._evaluate_clients_state(
                retained_clients, retrain_state
            ),
        }

    def _evaluate_clients_state(
        self, clients: List[FederatedClient], state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """对指定客户端列表评估模型，用于 retrain baseline。"""
        metrics = [client.evaluate(state_dict) for client in clients]
        return self._aggregate_metric_rows(metrics)

    def affected_clients(
        self, requester_id: int, requester_prototype: torch.Tensor
    ) -> tuple[List[int], List[Dict[str, float]]]:
        """根据 prototype 相似度找出“受影响客户端”，并返回相似度诊断信息。

        直观理解：
        如果别的客户端和 requester 的 relation 空间很接近，
        那么 requester 的遗忘请求可能也会影响到它们。
        """
        affected = []  # 受影响客户端列表
        scores = []  # 相似度分数列表
        # 遍历所有客户端
        for client in self.clients:
            # 跳过请求者自己和没有历史记录的客户端
            if client.client_id == requester_id or not client.history:
                continue
            # 获取客户端的原型 （最后一次的）
            prototype = client.history[-1]["prototype"]
            # 计算余弦相似度
            score = F.cosine_similarity(
                requester_prototype.unsqueeze(0), prototype.unsqueeze(0)
            ).item()
            # 添加到分数列表
            scores.append({"client_id": client.client_id, "similarity": score})
            # 如果相似度超过阈值，添加到受影响客户端列表
            if score > self.config.prototype_threshold:
                affected.append(client.client_id)
        # 按相似度降序排序
        scores.sort(key=lambda row: row["similarity"], reverse=True)
        return affected, scores

    def run_unlearning(self) -> None:
        """执行完整遗忘流程。
        流程分成三段：
        1. requester 客户端先本地遗忘
        2. 服务器找出受影响客户端
        3. 这些客户端做 purge 训练，再由服务器聚合
        """
        # 获取请求者客户端
        requester = self.clients[self.config.forget_client_id]
        print(f"遗忘请求客户端 ID = {self.config.forget_client_id}")

        # 先记下遗忘前全局指标，后面做前后对比。
        before_metrics = self.evaluate_state(self.global_state)
        print("Before unlearning metrics:")
        print(json.dumps(before_metrics, indent=2))
        # 获取请求者的旧原型
        # 联邦预训练最后聚合得到的全局模型，在 requester 本地重新计算出来的遗忘前 prototype。
        requester_old_prototype = requester.evaluate_prototype(self.global_state)

        # requester 先独立做本地遗忘。
        requester_state, probes, unlearn_info = requester.local_unlearn(
            self.global_state
        )

        # 服务器根据 prototype 相似度找受影响客户端。
        affected, similarity_scores = self.affected_clients(
            self.config.forget_client_id, requester_old_prototype
        )
        print("Prototype similarities:")
        # 打印相似度分数
        for row in similarity_scores:
            print(f"  client={row['client_id']} similarity={row['similarity']:.4f}")
        print("Affected clients:", affected)

        # 额外记录 requester 自己在遗忘节点上的 relation 变化。
        # 转换遗忘节点 ID 为张量
        requester_forget_ids = torch.tensor(
            unlearn_info["forget_node_ids"], dtype=torch.long
        )
        # 评估遗忘前的关系激活
        requester_relation_before = requester.evaluate_relation_stats(
            self.global_state, requester_forget_ids
        )
        # 评估本地遗忘后的关系激活
        requester_relation_after_local = requester.evaluate_relation_stats(
            requester_state, requester_forget_ids
        )
        # 评估探针对齐
        requester_probe_alignment = requester.evaluate_probe_alignment(
            self.global_state, requester_state, probes
        )

        """
        记录 requester 在遗忘节点上的 relation 变化，以及 probe 对遗忘前后模型的区分能力，用来分析本地遗忘是否真的生效。
        深拷贝全局状态 → 作为每轮净化的基准
        遍历所有客户端
        遗忘客户端 (requester)：直接用遗忘后的模型，不训练
        受影响客户端：执行 purge_train 净化蒸馏
        未受影响客户端：保持全局模型不动，不做任何训练
        收集所有状态 → 用于下一轮聚合
        """

        # 深拷贝当前全局状态
        current_state = copy.deepcopy(self.global_state)
        # 执行指定轮数的净化训练
        for round_idx in range(self.config.purge_rounds):
            all_states = []  # 所有客户端的模型状态
            # 遍历所有客户端
            for client in self.clients:
                if client.client_id == self.config.forget_client_id:
                    # requester 客户端直接使用自己的遗忘后模型
                    all_states.append(requester_state)
                elif client.client_id in affected:
                    # 受影响客户端执行 purge 训练
                    all_states.append(
                        client.purge_train(current_state, requester_state, probes)
                    )
                else:
                    # 未受影响客户端保持当前参数不动
                    all_states.append(current_state)

            # 聚合客户端参数
            current_state = self._average_state_dicts(all_states)
            # 评估聚合后的模型
            if self._should_eval_round(round_idx, self.config.purge_rounds):
                metrics = self.evaluate_state(current_state)
                metrics["round"] = round_idx + 1
                self.history["purge"].append(metrics)
                val_key, test_key = self._main_metric_names()
                print(
                    f"[Purge] round={round_idx + 1} val={metrics[val_key]:.4f} test={metrics[test_key]:.4f}"
                )
            else:
                print(f"[Purge] round={round_idx + 1} train_only")

        # 更新全局状态
        self.global_state = current_state

        # 再记录遗忘后的最终指标。
        after_metrics = self.evaluate_state(self.global_state)
        # 评估最终的关系激活
        requester_relation_after_final = requester.evaluate_relation_stats(
            self.global_state, requester_forget_ids
        )
        # 评估受影响客户端的指标
        affected_client_metrics = [
            client.evaluate(self.global_state)
            for client in self.clients
            if client.client_id in affected
        ]

        # 记录遗忘结果摘要
        self.experiment_summary["unlearning"] = {
            "before_global_metrics": before_metrics,
            "after_global_metrics": after_metrics,
            "metric_delta": {
                key + "_delta": after_metrics[key] - before_metrics[key]
                for key in before_metrics
            },
            "forget_request": {
                "client_id": self.config.forget_client_id,
                **unlearn_info,
            },
            "affected_client_ids": affected,
            "affected_client_count": len(affected),
            "prototype_similarities": similarity_scores,
            "requester_relation_stats": {
                "before_local_unlearning": requester_relation_before,
                "after_local_unlearning": requester_relation_after_local,
                "after_final_purge": requester_relation_after_final,
            },
            "probe_stats": {
                "probe_count": len(probes),
                "pre_vs_local_unlearn_alignment_mse": requester_probe_alignment,
            },
            "affected_client_final_mean_test_metric": float(
                np.mean(
                    [
                        m[
                            "test_auc_roc"
                            if self.config.task == "link_prediction"
                            else "test_acc"
                        ]
                        for m in affected_client_metrics
                    ]
                )
            )
            if affected_client_metrics
            else 0.0,
        }

    def save_outputs(self) -> None:
        """只保存最基本的实验结果。"""

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.global_state, out_dir / "final_global_model.pt")

        with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.experiment_summary, f, indent=2)
