# Robu

`run_booksnc_client_robustness.py` 用现有 `Robustness/run_mmfgu_client_robustness.py` 框架，直接在 `datasets/books-nc` 上跑 MMFGU 的客户端遗忘鲁棒性实验。

默认设置：

- 数据集：`E:\MMFGU\datasets\books-nc`
- 任务：`node_classification`
- 模型：`GCN`
- 客户端数：`10`

运行示例：

```bash
python Robu/run_booksnc_client_robustness.py --device cpu --federated-rounds 5 --purge-rounds 2
```

输出默认写到：`Robu/outputs/books_nc_mmfgu_gcn`

预先离线保存 `books-nc` 的 Louvain 客户端划分：

```bash
python Robu/save_booksnc_louvain_partitions.py --num-clients 10 --seeds 42 43 44
```
