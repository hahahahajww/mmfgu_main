# 导入必要的库
import ast  # 用于安全解析字符串
from pathlib import Path  # 用于处理文件路径
from typing import List, Tuple, Optional  # 类型提示

import networkx as nx  # 网络分析库
import numpy as np  # NumPy 库，用于数值计算
import pandas as pd  # Pandas 库，用于数据处理
import torch  # PyTorch 核心库
from networkx.algorithms.community import louvain_communities  # Louvain 社区发现算法
from sklearn.model_selection import train_test_split  # 用于数据集划分
from torch_geometric.data import Data  # PyTorch Geometric 的数据结构
from torch_geometric.utils import subgraph  # 用于子图操作


# 输入："[1,2,3]"
# 输出：[1, 2, 3]
def safe_parse_list(value) -> List[int]:
    """把 csv 里像 '[1, 2, 3]' 这样的字符串安全转成列表。"""
    # 检查值是否为 NaN
    if pd.isna(value):
        return []
    try:
        # 尝试解析字符串
        parsed = ast.literal_eval(str(value))
    except Exception:
        # 解析失败返回空列表
        return []
    # 检查解析结果是否为列表
    if not isinstance(parsed, list):
        return []

    # 尝试将列表中的元素转换为整数
    result = []
    for item in parsed:
        try:
            result.append(int(item))
        except Exception:
            # 转换失败跳过该元素
            pass
    return result


LABEL_CANDIDATES = [
    "label",
    "labels",
    "category",
    "class",
    "y",
    "target",
    "TYPE",
    "SCHOOL",
    "TIMEFRAME",
    "AUTHOR",
]
EDGE_GROUP_CANDIDATES = [
    ["also_buy", "also_view"],
    ["also_posted"],
    ["neighbors"],
    ["edge_list"],
]


def _build_label_locality_edges(
    labels: torch.Tensor, num_nodes: int, max_neighbors: int = 8
) -> torch.Tensor:
    """当数据缺少真实边时，为链接预测构造一个轻量伪图。

    规则：在同标签节点内部，按原始顺序给每个节点连接后续若干个节点，
    从而得到稀疏、可复现且规模可控的无向边。
    """

    edges = set()
    for label in torch.unique(labels).tolist():
        members = (labels == int(label)).nonzero(as_tuple=False).view(-1).tolist()
        if len(members) < 2:
            continue
        for idx, src in enumerate(members):
            upper = min(len(members), idx + 1 + max_neighbors)
            for dst in members[idx + 1 : upper]:
                if src == dst:
                    continue
                edges.add((src, dst))
                edges.add((dst, src))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def _sample_negative_targets(
    src_nodes: torch.Tensor,
    pos_nodes: torch.Tensor,
    allowed_nodes: torch.Tensor,
    edge_pairs: set[tuple[int, int]],
    num_neg: int,
    seed: int,
) -> torch.Tensor:
    """为链接预测评估采样负目标节点。"""

    generator = torch.Generator().manual_seed(seed)
    allowed_list = allowed_nodes.tolist()
    result = []
    for row_idx, (src, pos) in enumerate(zip(src_nodes.tolist(), pos_nodes.tolist())):
        picked = []
        picked_set = set()
        max_tries = max(100, num_neg * 20)
        tries = 0
        while len(picked) < num_neg and tries < max_tries:
            candidate = allowed_list[
                int(
                    torch.randint(
                        0, len(allowed_list), (1,), generator=generator
                    ).item()
                )
            ]
            tries += 1
            if candidate == pos or candidate == src:
                continue
            if (src, candidate) in edge_pairs:
                continue
            if candidate in picked_set:
                continue
            picked.append(candidate)
            picked_set.add(candidate)

        if not picked:
            for candidate in allowed_list:
                if (
                    candidate != pos
                    and candidate != src
                    and (src, candidate) not in edge_pairs
                ):
                    picked.append(candidate)
                    if len(picked) == num_neg:
                        break

        if not picked:
            picked.append(allowed_list[0] if allowed_list else 0)

        while len(picked) < num_neg:
            picked.append(picked[-1])
        result.append(picked[:num_neg])

    return torch.tensor(result, dtype=torch.long)


def _build_lp_split_from_file(
    data_dir: Path, num_nodes: int
) -> Optional[dict[str, dict[str, torch.Tensor]]]:
    split_path = data_dir / "lp-edge-split.pt"
    if not split_path.exists():
        return None
    split = torch.load(split_path, weights_only=False)
    expected_parts = {"train", "valid", "test"}
    if not isinstance(split, dict) or set(split.keys()) != expected_parts:
        return None

    sanitized: dict[str, dict[str, torch.Tensor]] = {}
    for part in ["train", "valid", "test"]:
        payload = split[part]
        src = payload["source_node"].long()
        dst = payload["target_node"].long()
        keep_mask = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)

        clean_payload = {
            "source_node": src[keep_mask].contiguous(),
            "target_node": dst[keep_mask].contiguous(),
        }

        if "target_node_neg" in payload:
            neg = payload["target_node_neg"].long()
            neg_keep_mask = ((neg >= 0) & (neg < num_nodes)).all(dim=1)
            keep_mask = keep_mask & neg_keep_mask
            clean_payload["source_node"] = src[keep_mask].contiguous()
            clean_payload["target_node"] = dst[keep_mask].contiguous()
            clean_payload["target_node_neg"] = neg[keep_mask].contiguous()

        filtered = int((~keep_mask).sum().item())
        if filtered > 0:
            print(f"Filtered {filtered} invalid LP {part} edges from {split_path.name}")
        sanitized[part] = clean_payload

    return sanitized


def _build_lp_split_from_edges(
    edge_index: torch.Tensor, num_nodes: int, seed: int
) -> dict[str, dict[str, torch.Tensor]]:
    """没有预制 lp-edge-split.pt 时，从图边自动构造一个链接预测划分。"""

    undirected_edges = sorted(
        {
            (min(int(src), int(dst)), max(int(src), int(dst)))
            for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist())
            if int(src) != int(dst)
        }
    )
    if not undirected_edges:
        empty = torch.empty(0, dtype=torch.long)
        return {
            "train": {"source_node": empty, "target_node": empty},
            "valid": {
                "source_node": empty,
                "target_node": empty,
                "target_node_neg": torch.empty((0, 1), dtype=torch.long),
            },
            "test": {
                "source_node": empty,
                "target_node": empty,
                "target_node_neg": torch.empty((0, 1), dtype=torch.long),
            },
        }

    edge_array = np.array(undirected_edges, dtype=np.int64)
    edge_ids = np.arange(len(edge_array))
    train_ids, temp_ids = train_test_split(
        edge_ids, test_size=0.2, random_state=seed, shuffle=True
    )
    valid_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=seed, shuffle=True
    )

    all_pairs = set(undirected_edges)

    def make_split(indices: np.ndarray, with_neg: bool, split_seed: int) -> dict[str, torch.Tensor]:
        subset = edge_array[indices]
        src = torch.tensor(subset[:, 0], dtype=torch.long)
        dst = torch.tensor(subset[:, 1], dtype=torch.long)
        payload = {"source_node": src, "target_node": dst}
        if with_neg:
            payload["target_node_neg"] = _sample_negative_targets(
                src,
                dst,
                torch.arange(num_nodes, dtype=torch.long),
                all_pairs,
                num_neg=150,
                seed=split_seed,
            )
        return payload

    return {
        "train": make_split(train_ids, False, seed),
        "valid": make_split(valid_ids, True, seed + 1),
        "test": make_split(test_ids, True, seed + 2),
    }


def _attach_lp_split(
    global_data: Data, split: dict[str, dict[str, torch.Tensor]]
) -> None:
    for part, payload in split.items():
        global_data[f"lp_{part}_source_node"] = payload["source_node"].long()
        global_data[f"lp_{part}_target_node"] = payload["target_node"].long()
        if "target_node_neg" in payload:
            global_data[f"lp_{part}_target_node_neg"] = payload[
                "target_node_neg"
            ].long()


def _localize_lp_split(
    global_data: Data, nodes: torch.Tensor, seed: int
) -> dict[str, torch.Tensor]:
    global_to_local = {int(node): idx for idx, node in enumerate(nodes.tolist())}
    allowed_nodes = nodes.clone().long()
    edge_pairs = set(
        zip(
            global_data.lp_train_source_node.tolist(),
            global_data.lp_train_target_node.tolist(),
        )
    )
    local_split: dict[str, torch.Tensor] = {}

    for part in ["train", "valid", "test"]:
        src_all = global_data[f"lp_{part}_source_node"]
        dst_all = global_data[f"lp_{part}_target_node"]
        keep_mask = torch.tensor(
            [
                int(src) in global_to_local and int(dst) in global_to_local
                for src, dst in zip(src_all.tolist(), dst_all.tolist())
            ],
            dtype=torch.bool,
        )
        local_src_global = src_all[keep_mask]
        local_dst_global = dst_all[keep_mask]
        local_split[f"lp_{part}_source_node"] = torch.tensor(
            [global_to_local[int(v)] for v in local_src_global.tolist()],
            dtype=torch.long,
        )
        local_split[f"lp_{part}_target_node"] = torch.tensor(
            [global_to_local[int(v)] for v in local_dst_global.tolist()],
            dtype=torch.long,
        )

        if part in {"valid", "test"}:
            key = f"lp_{part}_target_node_neg"
            if local_src_global.numel() == 0:
                local_split[key] = torch.empty((0, 1), dtype=torch.long)
                continue
            neg_global = global_data[key][keep_mask]
            neg_rows = []
            for row_idx, row in enumerate(neg_global.tolist()):
                filtered = [
                    global_to_local[int(v)]
                    for v in row
                    if int(v) in global_to_local
                    and int(v) != int(local_dst_global[row_idx].item())
                    and int(v) != int(local_src_global[row_idx].item())
                ]
                neg_rows.append(filtered)

            need_resample = any(len(row) == 0 for row in neg_rows)
            if need_resample:
                sampled = _sample_negative_targets(
                    local_src_global,
                    local_dst_global,
                    allowed_nodes,
                    edge_pairs,
                    num_neg=150,
                    seed=seed + hash(part) % 997,
                )
                local_split[key] = torch.tensor(
                    [
                        [global_to_local[int(v)] for v in row]
                        for row in sampled.tolist()
                    ],
                    dtype=torch.long,
                )
            else:
                min_neg = min(len(row) for row in neg_rows)
                min_neg = max(1, min_neg)
                local_split[key] = torch.tensor(
                    [row[:min_neg] for row in neg_rows], dtype=torch.long
                )

    return local_split


def _pick_single_file(files: List[Path], kind: str, root: Path) -> Path:
    """从候选文件中选择一个文件。"""

    if not files:
        raise FileNotFoundError(f"No {kind} file found in {root}")
    return sorted(files)[0]


def _load_optional_graph_object(data_dir: Path) -> Optional[object]:
    """尝试加载目录里的图对象文件，失败则返回 None。"""

    graph_candidates = []
    seen = set()
    for pattern in ["*PygGraph.pt", "*Graph.pt", "*.pt"]:
        for graph_path in sorted(data_dir.glob(pattern)):
            if graph_path in seen:
                continue
            seen.add(graph_path)
            graph_candidates.append(graph_path)
    for graph_path in graph_candidates:
        try:
            return torch.load(graph_path, weights_only=False)
        except Exception:
            continue
    return None


def _extract_labels(frame: pd.DataFrame, graph_obj: Optional[object]) -> torch.Tensor:
    """优先从图对象读取标签，不行再尝试从 CSV 读取。"""

    if isinstance(graph_obj, Data) and hasattr(graph_obj, "y"):
        return graph_obj.y.long().view(-1)
    if (
        isinstance(graph_obj, dict)
        and "y" in graph_obj
        and torch.is_tensor(graph_obj["y"])
    ):
        return graph_obj["y"].long().view(-1)

    for column in LABEL_CANDIDATES:
        if column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                return torch.tensor(series.to_numpy(), dtype=torch.long)

            # 对字符串类别做自动编码，例如 TYPE / SCHOOL / TIMEFRAME / AUTHOR。
            codes, uniques = pd.factorize(series.fillna("unknown"), sort=True)
            print(f"Label source: column '{column}' with {len(uniques)} classes")
            return torch.tensor(codes, dtype=torch.long)

    raise ValueError(
        f"CSV file does not contain a label column from {LABEL_CANDIDATES}, "
        "and no labels were found in an optional graph .pt file."
    )


def _extract_edge_index_from_graph_obj(
    graph_obj: Optional[object],
) -> Optional[torch.Tensor]:
    """从图对象中提取 edge_index。"""

    if isinstance(graph_obj, Data) and hasattr(graph_obj, "edge_index"):
        return graph_obj.edge_index.long().contiguous()
    if isinstance(graph_obj, dict):
        for key in ["edge_index", "edges", "adjacency"]:
            if key in graph_obj and torch.is_tensor(graph_obj[key]):
                tensor = graph_obj[key].long()
                if tensor.dim() == 2 and tensor.size(0) == 2:
                    return tensor.contiguous()
                if tensor.dim() == 2 and tensor.size(1) == 2:
                    return tensor.t().contiguous()
    return None


def _extract_edge_index(
    frame: pd.DataFrame, graph_obj: Optional[object]
) -> torch.Tensor:
    """优先从图对象读取边，不行再尝试从 CSV 边字段构图。"""

    edge_index = _extract_edge_index_from_graph_obj(graph_obj)
    if edge_index is not None:
        return edge_index

    num_nodes = len(frame)
    for columns in EDGE_GROUP_CANDIDATES:
        if all(column in frame.columns for column in columns):
            edges = set()
            for idx, row in frame.iterrows():
                for name in columns:
                    for dst in safe_parse_list(row[name]):
                        if 0 <= dst < num_nodes and dst != idx:
                            edges.add((idx, dst))
                            edges.add((dst, idx))
            if edges:
                return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # 如果没有边信息，为每个节点创建自环边
    num_nodes = len(frame)
    self_loops = (
        torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long)
        .t()
        .contiguous()
    )
    return self_loops


def _load_books_nc_graph(data_dir: Path) -> Optional[Data]:
    """加载 books-nc 这类以 .pt 文件组织的节点分类数据。"""

    labels_path = data_dir / "labels-w-missing.pt"
    split_path = data_dir / "split.pt"
    edges_path = data_dir / "nc_edges-nodeid.pt"
    fused_path = data_dir / "t5vit_feat.pt"

    required = [labels_path, split_path, edges_path]
    if not all(path.exists() for path in required):
        return None
    if not fused_path.exists():
        return None

    labels = torch.as_tensor(
        torch.load(labels_path, map_location="cpu", weights_only=False),
        dtype=torch.long,
    ).view(-1)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    edges = torch.as_tensor(
        torch.load(edges_path, map_location="cpu", weights_only=False),
        dtype=torch.long,
    )
    if edges.dim() != 2 or edges.size(-1) != 2:
        raise ValueError(f"Unsupported edge format in {edges_path}")
    edge_index = edges.t().contiguous()

    num_nodes = int(labels.numel())
    valid_edge_mask = (
        (edge_index[0] >= 0)
        & (edge_index[0] < num_nodes)
        & (edge_index[1] >= 0)
        & (edge_index[1] < num_nodes)
    )
    edge_index = edge_index[:, valid_edge_mask].contiguous()

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.as_tensor(split["train_idx"], dtype=torch.long)] = True
    val_mask[torch.as_tensor(split["val_idx"], dtype=torch.long)] = True
    test_mask[torch.as_tensor(split["test_idx"], dtype=torch.long)] = True

    fused_x = torch.load(fused_path, map_location="cpu", weights_only=False).float()
    if fused_x.size(0) != num_nodes:
        raise ValueError(
            f"Feature rows in {fused_path.name} do not match labels: "
            f"{fused_x.size(0)} vs {num_nodes}"
        )
    data = Data(
        x=fused_x,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )
    data.source_data_dir = str(data_dir)
    return data


def build_global_graph(
    data_dir: Path, seed: int, task: str = "node_classification"
) -> Data:
    """从数据目录自动识别文件并构建全局图。"""

    pt_graph = _load_books_nc_graph(data_dir)
    if pt_graph is not None:
        return pt_graph

    csv_path = _pick_single_file(list(data_dir.glob("*.csv")), "CSV", data_dir)
    image_path = _pick_single_file(
        list((data_dir / "image_features").glob("*.npy")),
        "image feature .npy",
        data_dir / "image_features",
    )
    text_path = _pick_single_file(
        list((data_dir / "text_features").glob("*.npy")),
        "text feature .npy",
        data_dir / "text_features",
    )
    graph_obj = _load_optional_graph_object(data_dir)

    # 读取数据
    frame = pd.read_csv(csv_path)  # 读取 CSV 文件
    image_x = torch.from_numpy(np.load(image_path)).float()  # 加载图像特征并转换为张量
    text_x = torch.from_numpy(np.load(text_path)).float()  # 加载文本特征并转换为张量
    try:
        y = _extract_labels(frame, graph_obj)  # 加载标签并转换为张量
    except ValueError:
        if task == "link_prediction":
            y = torch.zeros(len(frame), dtype=torch.long)
        else:
            raise

    edge_index = _extract_edge_index(frame, graph_obj)
    num_nodes = len(frame)
    valid_edge_mask = (
        (edge_index[0] >= 0)
        & (edge_index[0] < num_nodes)
        & (edge_index[1] >= 0)
        & (edge_index[1] < num_nodes)
    )
    if int((~valid_edge_mask).sum().item()) > 0:
        print(
            f"Filtered {int((~valid_edge_mask).sum().item())} invalid global edges"
        )
        edge_index = edge_index[:, valid_edge_mask].contiguous()
    if task == "link_prediction":
        nonself_mask = edge_index[0] != edge_index[1]
        if int(nonself_mask.sum().item()) == 0:
            print("LP edge mode fallback: label-locality pseudo edges")
            edge_index = _build_label_locality_edges(y, len(frame))
            if edge_index.numel() == 0:
                edge_index = _extract_edge_index(frame, graph_obj)

    # 优先做分层划分；如果类别太稀导致失败，则退化为普通随机划分。
    node_ids = np.arange(len(frame))  # 节点 ID 数组
    labels = y.numpy()  # 标签数组
    try:
        # 第一次划分：训练集 60%，临时集 40%
        train_idx, temp_idx = train_test_split(
            node_ids, test_size=0.4, random_state=seed, stratify=labels
        )
        # 第二次划分：验证集 20%，测试集 20%
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=labels[temp_idx]
        )
        print("Data split mode: stratified 6:2:2")
    except ValueError as exc:
        print(f"Data split mode fallback: random 6:2:2 ({exc})")
        train_idx, temp_idx = train_test_split(
            node_ids, test_size=0.4, random_state=seed, stratify=None
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=None
        )

    # 创建掩码
    train_mask = torch.zeros(len(frame), dtype=torch.bool)  # 训练掩码
    val_mask = torch.zeros(len(frame), dtype=torch.bool)  # 验证掩码
    test_mask = torch.zeros(len(frame), dtype=torch.bool)  # 测试掩码
    # 设置掩码值  训练时只能看到掩码为True的值
    train_mask[torch.tensor(train_idx)] = True
    val_mask[torch.tensor(val_idx)] = True
    test_mask[torch.tensor(test_idx)] = True

    global_data = Data(
        image_x=image_x,  # 图像特征
        text_x=text_x,  # 文本特征
        edge_index=edge_index,  # 边索引
        y=y,  # 标签
        train_mask=train_mask,  # 训练掩码
        val_mask=val_mask,  # 验证掩码
        test_mask=test_mask,  # 测试掩码
        num_nodes=len(frame),  # 节点数量
    )
    global_data.source_data_dir = str(data_dir)
    lp_split = _build_lp_split_from_file(data_dir, len(frame))
    if lp_split is None and task == "link_prediction":
        print("LP split mode: generated from edge_index 8:1:1")
        lp_split = _build_lp_split_from_edges(edge_index, len(frame), seed)
    if lp_split is not None:
        _attach_lp_split(global_data, lp_split)
    return global_data


def _merge_louvain_communities(
    global_data: Data, num_clients: int, seed: int
) -> List[torch.Tensor]:
    """先做 Louvain 社区发现，再把社区合并成固定数量的客户端。"""
    print(
        f"[ClientSplit] Start Louvain: num_nodes={global_data.num_nodes} "
        f"num_edges={int(global_data.edge_index.size(1))} num_clients={num_clients}"
    )
    # 创建 NetworkX 图
    graph = nx.Graph()
    graph.add_nodes_from(range(global_data.num_nodes))  # 添加所有节点

    # 提取边并转换为无向边
    row, col = global_data.edge_index.cpu()  # 获取边的行和列
    undirected_edges = {
        (min(int(src), int(dst)), max(int(src), int(dst)))
        for src, dst in zip(row.tolist(), col.tolist())
        if int(src) != int(dst)  # 排除自环
    }
    graph.add_edges_from(undirected_edges)  # 添加边
    print(f"[ClientSplit] NetworkX graph ready: undirected_edges={len(undirected_edges)}")

    # 执行 Louvain 社区发现
    communities = louvain_communities(graph, seed=seed)
    print(f"[ClientSplit] Louvain finished: communities={len(communities)}")
    # 按社区大小降序排序
    communities = sorted(
        (sorted(list(nodes)) for nodes in communities), key=len, reverse=True
    )

    # 用贪心装箱把 Louvain 社区合并成指定数量的客户端，尽量保持每个客户端规模接近。
    bins: List[List[int]] = [[] for _ in range(num_clients)]  # 客户端容器
    bin_sizes = [0 for _ in range(num_clients)]  # 客户端大小
    # 遍历每个社区
    for community in communities:
        # 找到当前最小的客户端
        target_idx = min(range(num_clients), key=lambda idx: bin_sizes[idx])
        # 将社区添加到该客户端
        bins[target_idx].extend(community)
        # 更新客户端大小
        bin_sizes[target_idx] += len(community)

    # 如果社区数量少于客户端数，补救性地把最大客户端继续切开，避免出现空客户端。只要还有客户端的数量是空的
    while any(len(nodes) == 0 for nodes in bins):
        # 找到空客户端
        empty_idx = next(idx for idx, nodes in enumerate(bins) if len(nodes) == 0)
        # 找到最大的客户端，在所有客户端里，找出【节点最多、最大的那个客户端】的编号！
        source_idx = max(range(num_clients), key=lambda idx: len(bins[idx]))
        # 获取最大客户端的节点并排序
        source_nodes = sorted(bins[source_idx])
        # 计算分割点
        split_point = len(source_nodes) // 2
        # 分割客户端
        bins[source_idx] = source_nodes[:split_point]
        bins[empty_idx] = source_nodes[split_point:]

    # 转换为张量并返回
    return [torch.tensor(sorted(nodes), dtype=torch.long) for nodes in bins]


"""
[
    tensor([0, 1, 2, 5, 7, 8, ...]),  # 客户端 0 的节点
    tensor([3, 4, 6, 9, 10, ...]),    # 客户端 1 的节点
    tensor([12, 15, 20, ...])         # 客户端 2 的节点
]
"""


def split_clients(
    global_data: Data, num_clients: int, seed: int
) -> List[Tuple[torch.Tensor, Data]]:
    """把全局图切分成多个客户端子图。

    当前采用 Louvain 社区划分：
    - 先根据全局图发现社区结构
    - 再把社区合并成 num_clients 个客户端
    - 每个客户端保留自己的局部边
    """
    print("[ClientSplit] Begin client partitioning")
    source_data_dir = Path(getattr(global_data, "source_data_dir", ""))
    partition_path = (
        source_data_dir
        / "client_partitions"
        / f"louvain_numclients{num_clients}_seed{seed}.pt"
    )
    if source_data_dir.name == "books-nc" and partition_path.exists():
        print(f"[ClientSplit] Loading cached partition: {partition_path}")
        payload = torch.load(partition_path, map_location="cpu", weights_only=False)
        parts = [torch.as_tensor(nodes, dtype=torch.long) for nodes in payload["client_nodes"]]
    else:
        # 使用 Louvain 社区发现并合并社区
        parts = _merge_louvain_communities(global_data, num_clients, seed)
    clients = []  # 客户端列表

    # 遍历每个客户端的节点，将全局图标号变成从0开始的标号，每个客户端
    # 假设客户端节点：[5, 8, 10]
    # tensor([[0, 1, 0],  # 源节点
    #         [1, 2, 2]])  # 目标节点
    for nodes in parts:
        # 对节点排序
        nodes = nodes.sort().values
        # 提取子图边索引
        edge_index, _ = subgraph(nodes, global_data.edge_index, relabel_nodes=True)
        local_kwargs = {
            "edge_index": edge_index,
            "y": global_data.y[nodes].clone(),
            "train_mask": global_data.train_mask[nodes].clone(),
            "val_mask": global_data.val_mask[nodes].clone(),
            "test_mask": global_data.test_mask[nodes].clone(),
            "num_nodes": nodes.numel(),
        }
        if hasattr(global_data, "x"):
            local_kwargs["x"] = global_data.x[nodes].clone()
        else:
            local_kwargs["image_x"] = global_data.image_x[nodes].clone()
            local_kwargs["text_x"] = global_data.text_x[nodes].clone()
        local = Data(**local_kwargs)
        if hasattr(global_data, "lp_train_source_node"):
            local_lp = _localize_lp_split(global_data, nodes, seed)
            for key, value in local_lp.items():
                local[key] = value
        # 添加到客户端列表
        clients.append((nodes, local))

    print(
        "[ClientSplit] Finished client partitioning: "
        + ", ".join(str(int(nodes.numel())) for nodes, _ in clients)
    )

    return clients


# clients = [
#     ( 客户端0的全局节点ID, 客户端0的本地小图 ),
#     ( 客户端1的全局节点ID, 客户端1的本地小图 ),
#     ( 客户端2的全局节点ID, 客户端2的本地小图 ),
#     ...
# ]
