import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FUSED.config import parse_args
from FUSED.fused import run


def main() -> None:
    config = parse_args()
    print("Running FUSED configuration:")
    print(json.dumps(config.asdict(), indent=2))

    runner = run(config)
    print("Final metrics:")
    print(json.dumps(runner.summary["fused"], indent=2))


if __name__ == "__main__":
    main()
