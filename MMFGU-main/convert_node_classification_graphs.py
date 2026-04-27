from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(r"E:\MMFGU\datasets")
EDGE_GROUP_CANDIDATES = [
    ["also_buy", "also_view"],
    ["also_posted"],
    ["neighbors"],
    ["edge_list"],
]
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


def safe_parse_list(value) -> list[int]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    result: list[int] = []
    for item in parsed:
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def extract_labels(frame: pd.DataFrame) -> torch.Tensor | None:
    for column in LABEL_CANDIDATES:
        if column not in frame.columns:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            return torch.tensor(series.to_numpy(), dtype=torch.long)
        codes, _ = pd.factorize(series.fillna("unknown"), sort=True)
        return torch.tensor(codes, dtype=torch.long)
    return None


def build_edges_from_csv(frame: pd.DataFrame) -> torch.Tensor | None:
    num_nodes = len(frame)
    for columns in EDGE_GROUP_CANDIDATES:
        if not all(column in frame.columns for column in columns):
            continue
        edges: set[tuple[int, int]] = set()
        for idx, row in frame.iterrows():
            for name in columns:
                for dst in safe_parse_list(row[name]):
                    if 0 <= dst < num_nodes and dst != idx:
                        edges.add((idx, dst))
                        edges.add((dst, idx))
        if edges:
            return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return None


def build_self_loops(num_nodes: int) -> torch.Tensor:
    return torch.arange(num_nodes, dtype=torch.long).repeat(2, 1).contiguous()


def convert_from_dgl_graph(graph_path: Path) -> torch.Tensor:
    from dgl.data.utils import load_graphs

    graphs, _ = load_graphs(str(graph_path))
    graph = graphs[0]
    src, dst = graph.edges()
    return torch.stack([src.long(), dst.long()], dim=0).contiguous()


def filter_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    valid_mask = (
        (edge_index[0] >= 0)
        & (edge_index[0] < num_nodes)
        & (edge_index[1] >= 0)
        & (edge_index[1] < num_nodes)
    )
    return edge_index[:, valid_mask].contiguous()


def convert_dataset(name: str) -> None:
    data_dir = ROOT / name
    csv_path = next(data_dir.glob("*.csv"))
    frame = pd.read_csv(csv_path)
    num_nodes = len(frame)
    y = extract_labels(frame)

    edge_index = None
    source = None

    graph_candidates = [
        path for path in sorted(data_dir.glob("*Graph.pt")) if not path.name.endswith("PygGraph.pt")
    ]
    graph_path = next(iter(graph_candidates), None)
    if graph_path is not None:
        try:
            edge_index = convert_from_dgl_graph(graph_path)
            source = f"dgl:{graph_path.name}"
        except Exception as exc:
            print(f"{name}: failed to convert {graph_path.name}: {type(exc).__name__}: {exc}")

    if edge_index is None:
        edge_index = build_edges_from_csv(frame)
        if edge_index is not None:
            source = "csv"

    if edge_index is None:
        edge_index = build_self_loops(num_nodes)
        source = "self_loop"

    edge_index = filter_edge_index(edge_index, num_nodes)

    graph_basename = name
    if graph_path is not None:
        graph_basename = graph_path.stem.replace("Graph", "") or name
    output_path = data_dir / f"{graph_basename}PygGraph.pt"
    payload = {
        "edge_index": edge_index,
        "num_nodes": num_nodes,
        "source": source,
    }
    if y is not None:
        payload["y"] = y
    torch.save(payload, output_path)

    nonself_edges = int((edge_index[0] != edge_index[1]).sum().item())
    print(
        f"{name}: saved {output_path.name} | num_nodes={num_nodes} | "
        f"num_edges={edge_index.size(1)} | nonself_edges={nonself_edges} | source={source}"
    )


def main() -> None:
    for data_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        if not any(data_dir.glob("*.csv")):
            continue
        convert_dataset(data_dir.name)


if __name__ == "__main__":
    main()
