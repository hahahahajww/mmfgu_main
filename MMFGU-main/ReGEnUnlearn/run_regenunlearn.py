from __future__ import annotations

import json

from ReGEnUnlearn.config import parse_args
from ReGEnUnlearn.server import ReGEnUnlearnServer
from ReGEnUnlearn.utils import set_seed


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    print("Running ReGEnUnlearn configuration:")
    print(json.dumps({
        "data_dir": config.data_dir,
        "num_clients": config.num_clients,
        "target_client_ids": list(config.target_client_ids),
        "federated_rounds": config.federated_rounds,
        "local_epochs": config.local_epochs,
        "sampling_rate": config.sampling_rate,
        "sampler_steps": config.sampler_steps,
        "prompt_token_count": config.prompt_token_count,
        "unlearn_epochs": config.unlearn_epochs,
        "repair_rounds": config.repair_rounds,
        "output_dir": config.output_dir,
        "run_unlearning": config.run_unlearning,
    }, indent=2))

    server = ReGEnUnlearnServer(config)
    server.pretrain()
    if config.run_unlearning:
        server.run_unlearning()
    server.save_outputs()
    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
