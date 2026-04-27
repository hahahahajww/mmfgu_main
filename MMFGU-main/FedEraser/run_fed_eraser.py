import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmfgu.config import parse_args
from mmfgu.utils import set_seed

from FedEraser.fed_eraser import FedEraserRunner


def parse_fed_eraser_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--retain-interval", type=int, default=2)
    parser.add_argument("--calibration-ratio", type=float, default=0.5)
    extra, remaining = parser.parse_known_args(sys.argv[1:])
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *remaining]
    try:
        config = parse_args()
    finally:
        sys.argv = original_argv
    if "--eval-interval" not in set(original_argv[1:]):
        config.eval_interval = 10
    return config, extra


def main() -> None:
    config, extra = parse_fed_eraser_args()
    set_seed(config.seed)

    print("Running FedEraser configuration:")
    print(
        json.dumps(
            {
                "data_dir": config.data_dir,
                "task": config.task,
                "num_clients": config.num_clients,
                "federated_rounds": config.federated_rounds,
                "local_epochs": config.local_epochs,
                "forget_client_id": config.forget_client_id,
                "retain_interval": extra.retain_interval,
                "calibration_ratio": extra.calibration_ratio,
                "output_dir": config.output_dir,
            },
            indent=2,
        )
    )

    runner = FedEraserRunner(
        config=config,
        retain_interval=extra.retain_interval,
        calibration_ratio=extra.calibration_ratio,
    )
    runner.pretrain_with_retention()
    runner.reconstruct_without_client()
    runner.save_outputs()

    print("Final metrics:")
    print(json.dumps(runner.summary["fed_eraser"], indent=2))


if __name__ == "__main__":
    main()
