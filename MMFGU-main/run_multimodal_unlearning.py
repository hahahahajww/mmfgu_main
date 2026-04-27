import json
import sys

from mmfgu.config import parse_args
from mmfgu.server import FederatedServer
from mmfgu.utils import set_seed


def apply_multimodal_unlearning_defaults(config):
    argv = set(sys.argv[1:])

    # 用更保守的统一默认值，减轻 purge 过强带来的精度回撤。
    if "--prototype-threshold" not in argv:
        config.prototype_threshold = 0.65
    if "--purge-rounds" not in argv:
        config.purge_rounds = 6
    if "--purge-local-epochs" not in argv:
        config.purge_local_epochs = 2
    if "--lambda-neg" not in argv:
        config.lambda_neg = 0.15

    return config


def main() -> None:
    """多模态遗忘实验入口。

    整体执行顺序很简单：
    1. 读取配置
    2. 固定随机种子
    3. 创建服务器
    4. 先做联邦预训练
    5. 如果指定了 --run-unlearning，就继续做遗忘流程
    6. 保存结果并打印最终指标
    """

    config = apply_multimodal_unlearning_defaults(parse_args())
    set_seed(config.seed)

    print("Running formal experiment configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "unlearn_local_epochs": config.unlearn_local_epochs,
                "probe_epochs": config.probe_epochs,
                "probe_count": config.probe_count,
                "probe_topk": config.probe_topk,
                "purge_rounds": config.purge_rounds,
                "purge_local_epochs": config.purge_local_epochs,
                "forget_client_id": config.forget_client_id,
                "forget_ratio": config.forget_ratio,
                "output_dir": config.output_dir,
                "run_unlearning": config.run_unlearning,
            },
            indent=2,
        )
    )

    server = FederatedServer(config)

    # 阶段 1：联邦预训练
    server.pretrain()

    # 阶段 2：如果需要，再继续做遗忘与净化
    if config.run_unlearning:
        server.run_unlearning()

    # 阶段 3：保存结果
    server.save_outputs()

    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
