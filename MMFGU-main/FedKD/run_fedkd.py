from __future__ import annotations

import json

from FedKD.config import parse_args
from FedKD.server import FedKDServer
from mmfgu.utils import set_seed


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    print("Running FedKD configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "forget_client_id": config.forget_client_id,
                "distill_epochs": config.distill_epochs,
                "distill_lr": config.distill_lr,
                "distill_temperature": config.distill_temperature,
                "output_dir": config.output_dir,
            },
            indent=2,
        )
    )

    server = FedKDServer(config)
    server.pretrain()
    server.run_client_unlearning()
    server.save_outputs()

    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
