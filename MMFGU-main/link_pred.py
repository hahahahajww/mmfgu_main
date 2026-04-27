import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mmfgu.config import Config
from mmfgu.data import build_global_graph
from mmfgu.model import make_model
from mmfgu.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal link prediction test")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn-type", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-neg", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="lp_test_outputs")
    return parser.parse_args()


def compute_edge_logits(node_embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src_h = node_embedding[edge_index[0]]
    dst_h = node_embedding[edge_index[1]]
    return (src_h * dst_h).sum(dim=1)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=args.data_dir,
        task="link_prediction",
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


def sanitize_edge_index(data) -> None:
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.long()
    valid_mask = (
        (edge_index[0] >= 0)
        & (edge_index[0] < num_nodes)
        & (edge_index[1] >= 0)
        & (edge_index[1] < num_nodes)
    )
    if int((~valid_mask).sum().item()) > 0:
        data.edge_index = edge_index[:, valid_mask].contiguous()
        print(f"Filtered {int((~valid_mask).sum().item())} invalid graph edges")


def sanitize_lp_split(data) -> None:
    num_nodes = int(data.num_nodes)
    for part in ["train", "valid", "test"]:
        src = getattr(data, f"lp_{part}_source_node").long()
        dst = getattr(data, f"lp_{part}_target_node").long()
        valid_mask = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)

        if part in {"valid", "test"}:
            neg = getattr(data, f"lp_{part}_target_node_neg").long()
            neg_mask = ((neg >= 0) & (neg < num_nodes)).all(dim=1)
            valid_mask = valid_mask & neg_mask
            setattr(data, f"lp_{part}_target_node_neg", neg[valid_mask].contiguous())

        filtered = int((~valid_mask).sum().item())
        if filtered > 0:
            print(f"Filtered {filtered} invalid {part} split edges")
        setattr(data, f"lp_{part}_source_node", src[valid_mask].contiguous())
        setattr(data, f"lp_{part}_target_node", dst[valid_mask].contiguous())


def stack_edges_and_labels(pos_src, pos_dst, neg_dst=None):
    pos_edge_index = torch.stack([pos_src, pos_dst], dim=0)
    pos_label = torch.ones(pos_src.size(0), dtype=torch.float)
    if neg_dst is None:
        return pos_edge_index, pos_label

    expanded_src = pos_src.unsqueeze(1).expand_as(neg_dst).reshape(-1)
    neg_edge_index = torch.stack([expanded_src, neg_dst.reshape(-1)], dim=0)
    neg_label = torch.zeros(expanded_src.size(0), dtype=torch.float)
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([pos_label, neg_label], dim=0)
    return edge_index, edge_label


def sample_train_negatives(num_nodes: int, pos_src: torch.Tensor, num_neg: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (pos_src.size(0), num_neg), generator=generator)


def encode_nodes(model, data, device: str) -> torch.Tensor:
    _, cache = model(data.to(device))
    return cache["propagated_h"]


def train(model, data, args) -> dict:
    device = args.device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_src = data.lp_train_source_node.cpu()
    train_dst = data.lp_train_target_node.cpu()
    best_valid_mrr = -1.0
    best_metrics = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_src.size(0))
        total_loss = 0.0

        for start in range(0, train_src.size(0), args.batch_size):
            idx = perm[start : start + args.batch_size]
            src = train_src[idx]
            dst = train_dst[idx]
            neg_dst = sample_train_negatives(
                data.num_nodes, src, args.num_neg, args.seed + epoch + start
            )
            edge_index, edge_label = stack_edges_and_labels(src, dst, neg_dst)

            node_embedding = encode_nodes(model, data, device)
            logits = compute_edge_logits(node_embedding, edge_index.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, edge_label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * idx.numel()

        metrics = evaluate(model, data, device, args.batch_size)
        valid_mrr = metrics["valid"]["mrr"]
        print(
            f"[LP] epoch={epoch} loss={total_loss / train_src.size(0):.4f} "
            f"valid_mrr={valid_mrr:.4f} valid_hits3={metrics['valid']['hits@3']:.4f} "
            f"valid_hits10={metrics['valid']['hits@10']:.4f}"
        )
        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_metrics


@torch.no_grad()
def evaluate(model, data, device: str, batch_size: int = 512) -> dict:
    model.eval()
    node_embedding = encode_nodes(model, data, device)
    metrics = {}

    for part in ["valid", "test"]:
        src_all = getattr(data, f"lp_{part}_source_node").cpu()
        pos_all = getattr(data, f"lp_{part}_target_node").cpu()
        neg_all = getattr(data, f"lp_{part}_target_node_neg").cpu()
        ranks = []
        auc_scores = []

        for start in range(0, src_all.size(0), batch_size):
            src = src_all[start : start + batch_size].to(device)
            pos_dst = pos_all[start : start + batch_size].to(device)
            neg_dst = neg_all[start : start + batch_size].to(device)

            pos_edge_index = torch.stack([src, pos_dst], dim=0)
            pos_score = compute_edge_logits(node_embedding, pos_edge_index)

            expanded_src = src.unsqueeze(1).expand_as(neg_dst)
            neg_edge_index = torch.stack([expanded_src.reshape(-1), neg_dst.reshape(-1)], dim=0)
            neg_score = compute_edge_logits(node_embedding, neg_edge_index).view_as(neg_dst)

            ranks.append(1 + (neg_score >= pos_score.unsqueeze(1)).sum(dim=1).float())

            pos_np = pos_score.detach().cpu().numpy()
            neg_np = neg_score.detach().cpu().numpy()
            for i in range(len(pos_np)):
                labels = np.concatenate([np.ones(1, dtype=np.int64), np.zeros(len(neg_np[i]), dtype=np.int64)])
                scores = np.concatenate([[pos_np[i]], neg_np[i]])
                auc_scores.append(roc_auc_score(labels, scores))

        rank = torch.cat(ranks, dim=0)
        metrics[part] = {
            "mrr": float((1.0 / rank).mean().item()),
            "hits@3": float((rank <= 3).float().mean().item()),
            "hits@10": float((rank <= 10).float().mean().item()),
            "auc_roc": float(np.mean(auc_scores)) if auc_scores else 0.0,
        }

    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    data = build_global_graph(Path(args.data_dir), args.seed, task="link_prediction")
    sanitize_edge_index(data)
    sanitize_lp_split(data)
    model = make_model(config, data).to(args.device)
    best_metrics = train(model, data, args)

    summary = {
        "data_dir": args.data_dir,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_neg": args.num_neg,
        "best_valid": best_metrics["valid"],
        "test": best_metrics["test"],
    }
    (output_dir / "lp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Final link prediction metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
