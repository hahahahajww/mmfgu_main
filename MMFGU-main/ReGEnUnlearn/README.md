# ReGEnUnlearn

这是一个基于论文 `Subgraph Federated Unlearning` 的多模态联邦图遗忘实现，按你当前工程的数据格式做了适配，代码全部放在 `ReGEnUnlearn/` 下，便于和现有方法做对比实验。

## 当前实现范围

- 多模态输入：`image_features/*.npy` + `text_features/*.npy`
- 图任务：节点分类
- 联邦训练：Louvain 子图划分 + FedAvg
- 遗忘核心：
  - `RFPS` 风格的强化采样器，学习选择更低干扰的遗忘子图
  - `PGPKD` 风格的参数自由 prompt 聚合与插入
  - 服务器侧梯度上升遗忘 + remaining clients repair
- 对比基线：`retrain from scratch`

## 目录说明

- `config.py`: 实验配置
- `data.py`: 数据读取与客户端子图划分
- `model.py`: 多模态 GNN
- `modules.py`: RFPS 与 prompt 蒸馏模块
- `client.py`: 客户端训练与遗忘工件生成
- `server.py`: 联邦训练、遗忘、重训练 baseline
- `run_regenunlearn.py`: 主实验入口
- `run_regenunlearn_retrain.py`: retrain baseline 入口

## 数据格式

数据目录需要和你当前项目保持一致，例如：

```text
datasets/Toys/
  Toys.csv
  image_features/*.npy
  text_features/*.npy
  optional graph pt file
```

## 运行方式

先跑预训练 + 遗忘：

```bash
python -m ReGEnUnlearn.run_regenunlearn \
  --data-dir E:\MMFGU\datasets\Toys \
  --num-clients 10 \
  --target-client-ids 0,1 \
  --federated-rounds 80 \
  --run-unlearning \
  --output-dir ReGEnUnlearn/outputs/two_client_unlearn
```

跑 retrain baseline：

```bash
python -m ReGEnUnlearn.run_regenunlearn_retrain \
  --data-dir E:\MMFGU\datasets\Toys \
  --num-clients 10 \
  --target-client-ids 0,1 \
  --federated-rounds 80 \
  --output-dir ReGEnUnlearn/outputs/two_client_retrain
```

## 输出文件

- `final_global_model.pt`
- `training_history.json`
- `config.json`
- `experiment_summary.json`

## 说明

这版实现是“论文思想 + 你现有多模态工程适配”的可运行版本，不是论文作者原始代码逐行复现。重点是保证：

1. 可以直接接到你当前数据格式上
2. 可以做 ReGEnUnlearn vs retrain 的对比
3. 关键模块和论文结构一一对应

如果你下一步要，我可以继续在这个文件夹里补：

1. `FedProx` / `FedPub` 风格聚合对比
2. `ASR` 回门触发验证
3. 多 seed 批量对比脚本
4. 更严格的 non-IID 划分
