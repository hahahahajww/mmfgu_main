import json
import sys
from pathlib import Path

from mmfgu.config import parse_args
from mmfgu.server import FederatedServer
from mmfgu.utils import set_seed


def apply_dataset_client_unlearning_defaults(config):
    """只给客户端遗忘入口加统一默认值。

    不会覆盖用户在命令行里显式传入的参数。
    这样既不影响模态遗忘，也不改全局默认配置。
    """

    argv = set(sys.argv[1:])

    # 用更保守的统一默认值，避免客户端 purge 过强导致整体精度掉得比 retrain 更多。
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
    """客户端遗忘独立入口。

    流程：
    1. 正常联邦预训练
    2. 目标客户端退出
    3. 对受影响客户端执行 purge
    4. 保存结果并打印最终指标
    """

    config = apply_dataset_client_unlearning_defaults(parse_args())
    set_seed(config.seed)

    print("Running client unlearning configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "purge_rounds": config.purge_rounds,
                "purge_local_epochs": config.purge_local_epochs,
                "forget_client_id": config.forget_client_id,
                "prototype_threshold": config.prototype_threshold,
                "lambda_neg": config.lambda_neg,
                "output_dir": config.output_dir,
            },
            indent=2,
        )
    )

    server = FederatedServer(config)
    server.pretrain()
    server.run_client_unlearning()
    server.save_outputs()

    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
