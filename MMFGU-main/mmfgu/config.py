# 导入必要的库
from dataclasses import dataclass  # 用于创建数据类
import argparse  # 用于解析命令行参数
import sys
import torch  # PyTorch 核心库


@dataclass
class Config:
    """集中放实验配置。

    你可以把它理解成“整套实验的参数总表”。
    运行时命令行参数会覆盖这里的默认值。
    """

    # 数据目录
    data_dir: str = r"E:\MMFGU\datasets\ele-fashion"  # 数据集所在路径

    # 任务类型
    task: str = "node_classification"  # node_classification 或 link_prediction

    # 联邦学习：客户端数量
    num_clients: int = 10  # 参与联邦学习的客户端数量

    # 模型结构参数
    hidden_dim: int = 256  # 隐藏层维度
    dropout: float = 0.2  # Dropout 率
    gnn_type: str = "sage"  # GNN 类型，可选值：sage、gcn、gat

    # 优化器参数
    lr: float = 1e-3  # 学习率
    weight_decay: float = 1e-4  # 权重衰减
    batch_size: int = 4096  # 链接预测训练/评估批大小

    # 预训练阶段：联邦轮数、每轮本地训练轮数
    federated_rounds: int = 100  # 联邦预训练轮数
    local_epochs: int = 5  # 每轮联邦训练中客户端本地训练的轮数
    eval_interval: int = 1  # 每隔多少轮做一次完整评估

    # 本地遗忘阶段：本地训练轮数
    unlearn_local_epochs: int = 10  # 本地遗忘训练的轮数

    # probe 修复阶段：probe 训练轮数、生成数量、保留 top-k
    probe_epochs: int = 3  # Probe 修复训练的轮数
    probe_count: int = 50  # 生成的 Probe 数量
    probe_topk: int = 10  # 保留的 top-k Probe 数量

    # 联邦净化阶段：服务器轮数、每个受影响客户端本地训练轮数
    purge_rounds: int = 5  # 联邦净化训练的轮数
    purge_local_epochs: int = 2  # 每轮净化训练中客户端本地训练的轮数

    # 遗忘请求参数
    forget_client_id: int = 0  # 需要执行遗忘的客户端 ID
    forget_ratio: float = 0.2  # 遗忘数据的比例

    # 根据 prototype 相似度判断“受影响客户端”
    prototype_threshold: float = 0.35  # 原型相似度阈值

    # 各个损失项的权重
    lambda_neg: float = 1.0  # 负样本损失权重
    alpha_dec: float = 1.0  # 关系衰减损失权重
    alpha_anchor: float = 1.0  # 锚点损失权重
    beta_mm: float = 1.0  # 保留节点损失权重
    delta_bd: float = 0.5  # 边界节点损失权重

    # 其他基础配置
    seed: int = 42  # 随机种子
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 设备选择
    output_dir: str = "outputs_formal_experiment"  # 输出目录

    # 是否在预训练后继续执行遗忘流程
    run_unlearning: bool = False  # 是否执行遗忘流程
    run_retrain_baseline: bool = False  # 是否执行重训练基线


def parse_args() -> Config:
    """把命令行参数解析成 Config。

    例如：
    python run_mmfgu.py --num-clients 5 --run-unlearning
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Federated multimodal node classification and relation unlearning"
    )
    # 添加命令行参数
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)  # 数据目录
    parser.add_argument(
        "--task",
        type=str,
        default=Config.task,
        choices=["node_classification", "link_prediction"],
    )
    parser.add_argument(
        "--num-clients", type=int, default=Config.num_clients
    )  # 客户端数量
    parser.add_argument(
        "--federated-rounds", type=int, default=Config.federated_rounds
    )  # 联邦轮数
    parser.add_argument(
        "--local-epochs", type=int, default=Config.local_epochs
    )  # 本地训练轮数
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument(
        "--unlearn-local-epochs",
        type=int,
        default=Config.unlearn_local_epochs,  # 本地遗忘训练轮数
    )
    parser.add_argument(
        "--probe-epochs", type=int, default=Config.probe_epochs
    )  # Probe 训练轮数
    parser.add_argument(
        "--probe-count", type=int, default=Config.probe_count
    )  # Probe 数量
    parser.add_argument(
        "--probe-topk", type=int, default=Config.probe_topk
    )  # 保留的 top-k Probe 数量
    parser.add_argument(
        "--purge-rounds", type=int, default=Config.purge_rounds
    )  # 净化轮数
    parser.add_argument(
        "--purge-local-epochs",
        type=int,
        default=Config.purge_local_epochs,  # 净化本地训练轮数
    )
    parser.add_argument(
        "--forget-client-id", type=int, default=Config.forget_client_id
    )  # 遗忘客户端 ID
    parser.add_argument(
        "--forget-ratio", type=float, default=Config.forget_ratio
    )  # 遗忘比例
    parser.add_argument(
        "--prototype-threshold",
        type=float,
        default=Config.prototype_threshold,  # 原型相似度阈值
    )
    parser.add_argument(
        "--lambda-neg", type=float, default=Config.lambda_neg
    )  # 负样本损失权重
    parser.add_argument("--alpha-dec", type=float, default=Config.alpha_dec)
    parser.add_argument("--alpha-anchor", type=float, default=Config.alpha_anchor)
    parser.add_argument("--beta-mm", type=float, default=Config.beta_mm)
    parser.add_argument("--delta-bd", type=float, default=Config.delta_bd)
    parser.add_argument(
        "--hidden-dim", type=int, default=Config.hidden_dim
    )  # 隐藏层维度
    parser.add_argument("--dropout", type=float, default=Config.dropout)  # Dropout 率
    parser.add_argument(
        "--gnn-type",
        type=str,
        default=Config.gnn_type,
        choices=["sage", "gcn", "gat"],  # GNN 类型
    )
    parser.add_argument("--lr", type=float, default=Config.lr)  # 学习率
    parser.add_argument(
        "--weight-decay", type=float, default=Config.weight_decay
    )  # 权重衰减
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seed", type=int, default=Config.seed)  # 随机种子
    parser.add_argument("--device", type=str, default=Config.device)  # 设备
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)  # 输出目录
    parser.add_argument("--run-unlearning", action="store_true")  # 是否执行遗忘
    parser.add_argument(
        "--run-retrain-baseline", action="store_true"
    )  # 是否执行重训练基线
    # 解析命令行参数
    args = parser.parse_args()

    argv = set(sys.argv[1:])
    if args.task == "link_prediction" and "--federated-rounds" not in argv:
        federated_rounds = 100
    else:
        federated_rounds = args.federated_rounds
    if args.task == "link_prediction" and "--eval-interval" not in argv:
        eval_interval = 5
    else:
        eval_interval = max(1, args.eval_interval)
    if args.task == "link_prediction" and "--local-epochs" not in argv:
        local_epochs = 5
    else:
        local_epochs = args.local_epochs

    # 创建并返回 Config 对象
    return Config(
        data_dir=args.data_dir,  # 数据目录
        task=args.task,
        num_clients=args.num_clients,  # 客户端数量
        federated_rounds=federated_rounds,  # 链接预测固定 100，节点分类按参数走
        local_epochs=local_epochs,  # 本地训练轮数
        eval_interval=eval_interval,
        unlearn_local_epochs=args.unlearn_local_epochs,  # 本地遗忘训练轮数
        probe_epochs=args.probe_epochs,  # Probe 训练轮数
        probe_count=args.probe_count,  # Probe 数量
        probe_topk=args.probe_topk,  # 保留的 top-k Probe 数量
        purge_rounds=args.purge_rounds,  # 净化轮数
        purge_local_epochs=args.purge_local_epochs,  # 净化本地训练轮数
        forget_client_id=args.forget_client_id,  # 遗忘客户端 ID
        forget_ratio=args.forget_ratio,  # 遗忘比例
        prototype_threshold=args.prototype_threshold,  # 原型相似度阈值
        lambda_neg=args.lambda_neg,  # 负样本损失权重
        alpha_dec=args.alpha_dec,
        alpha_anchor=args.alpha_anchor,
        beta_mm=args.beta_mm,
        delta_bd=args.delta_bd,
        hidden_dim=args.hidden_dim,  # 隐藏层维度
        dropout=args.dropout,  # Dropout 率
        gnn_type=args.gnn_type,  # GNN 类型
        lr=args.lr,  # 学习率
        weight_decay=args.weight_decay,  # 权重衰减
        batch_size=args.batch_size,
        seed=args.seed,  # 随机种子
        device=args.device,  # 设备
        output_dir=args.output_dir,  # 输出目录
        run_unlearning=args.run_unlearning,  # 是否执行遗忘
        run_retrain_baseline=args.run_retrain_baseline,  # 是否执行重训练基线
    )
