import json

from mmfgu.config import parse_args
from mmfgu.server import FederatedServer
from mmfgu.utils import set_seed


def main() -> None:
    """客户端遗忘对应的 retrain baseline 独立入口。"""

    config = parse_args()
    set_seed(config.seed)

    print("Running client retrain baseline configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "forget_client_id": config.forget_client_id,
                "output_dir": config.output_dir,
            },
            indent=2,
        )
    )

    server = FederatedServer(config)
    server.run_client_retrain_baseline()
    server.save_outputs()

    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
