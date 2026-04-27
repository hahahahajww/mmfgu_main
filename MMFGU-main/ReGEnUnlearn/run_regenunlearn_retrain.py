from __future__ import annotations

import json

from ReGEnUnlearn.config import parse_args
from ReGEnUnlearn.server import ReGEnUnlearnServer
from ReGEnUnlearn.utils import set_seed


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    server = ReGEnUnlearnServer(config)
    server.run_retrain_baseline()
    server.save_outputs()
    print("Retrain final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
