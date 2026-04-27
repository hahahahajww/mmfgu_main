"""`mmfgu` 包的对外导出入口。

通常外部只需要这几个核心对象：
- `Config`：实验配置
- `parse_args`：命令行参数解析
- `FederatedServer`：服务器主流程
- `set_seed`：固定随机种子
"""
from .config import Config, parse_args
from .server import FederatedServer
from .utils import set_seed
