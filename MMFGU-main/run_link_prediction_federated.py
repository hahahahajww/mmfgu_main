import json
import sys
from typing import Optional

from mmfgu.config import parse_args
from mmfgu.server import FederatedServer
from mmfgu.utils import set_seed


def ensure_arg(flag: str, value: Optional[str] = None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def main() -> None:
    """联邦链接预测专用入口。

    默认固定为 link prediction，并给出一套可直接运行的默认值；
    用户仍然可以在命令行显式覆盖这些参数。
    """

    ensure_arg("--task", "link_prediction")
    ensure_arg("--data-dir", r"E:\MMFGU\datasets\Sports")
    ensure_arg("--output-dir", r"E:\MMFGU\outputs_lp_unlearning")

    config = parse_args()
    set_seed(config.seed)

    print("Running federated link prediction configuration:")
    print(
        json.dumps(
            {
                "task": config.task,
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
                "alpha_dec": config.alpha_dec,
                "alpha_anchor": config.alpha_anchor,
                "beta_mm": config.beta_mm,
                "delta_bd": config.delta_bd,
                "output_dir": config.output_dir,
                "run_unlearning": config.run_unlearning,
            },
            indent=2,
        )
    )

    server = FederatedServer(config)
    server.pretrain()

    if config.run_unlearning:
        server.run_unlearning()

    server.save_outputs()

    print("Final metrics:")
    print(json.dumps(server.evaluate_state(server.global_state), indent=2))


if __name__ == "__main__":
    main()
