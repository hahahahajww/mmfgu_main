import json

from mmfgu.config import parse_args
from mmfgu.server import FederatedServer
from mmfgu.utils import set_seed


def main() -> None:
    """多模态遗忘对应的 retrain baseline 入口。

    这里不走普通预训练/遗忘流程，
    而是直接在物理删除训练样本后的数据上从头训练，
    作为 gold retrain baseline。
    """

    config = parse_args()
    set_seed(config.seed)

    print("Running retrain baseline configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "forget_client_id": config.forget_client_id,
                "forget_ratio": config.forget_ratio,
                "output_dir": config.output_dir,
            },
            indent=2,
        )
    )

    server = FederatedServer(config)
    server.run_retrain_baseline()
    server.save_outputs()

    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
