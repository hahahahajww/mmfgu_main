import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mmfgu.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal multimodal link prediction")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="lp_outputs")
    return parser.parse_args()


class LinkPredictor(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def encode(self, image_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        image_h = self.image_proj(image_x)
        text_h = self.text_proj(text_x)
        return self.fusion(torch.cat([image_h, text_h], dim=-1))

    def score_pairs(
        self, node_emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        return (node_emb[src] * node_emb[dst]).sum(dim=-1)


def load_data(data_dir: Path):
    split = torch.load(data_dir / "lp-edge-split.pt", weights_only=False)
    image_path = next((data_dir / "image_features").glob("*.npy"))
    text_path = next((data_dir / "text_features").glob("*.npy"))
    image_x = torch.from_numpy(np.load(image_path)).float()
    text_x = torch.from_numpy(np.load(text_path)).float()

    num_nodes = (
        max(
            int(split[part]["source_node"].max()) for part in ["train", "valid", "test"]
        )
        + 1
    )
    if image_x.size(0) < num_nodes or text_x.size(0) < num_nodes:
        clip_path = data_dir / "clip_feat.pt"
        if clip_path.exists():
            clip_x = torch.load(clip_path, weights_only=False).float()
            if clip_x.size(0) >= num_nodes:
                print(
                    "Falling back to clip_feat.pt because modality rows do not match node ids"
                )
                return clip_x[:num_nodes], clip_x[:num_nodes], split
        raise ValueError("Feature rows do not cover all node ids in lp-edge-split.pt")

    return image_x[:num_nodes], text_x[:num_nodes], split


@torch.no_grad()
def evaluate(model, image_x, text_x, split, device: str, batch_size: int = 512):
    model.eval()
    node_emb = model.encode(image_x.to(device), text_x.to(device))
    metrics = {}

    for part in ["valid", "test"]:
        src_all = split[part]["source_node"]
        pos_all = split[part]["target_node"]
        neg_all = split[part]["target_node_neg"]
        ranks = []
        auc_scores = []

        for start in range(0, src_all.size(0), batch_size):
            src = src_all[start : start + batch_size].to(device)
            pos_dst = pos_all[start : start + batch_size].to(device)
            neg_dst = neg_all[start : start + batch_size].to(device)

            pos_score = model.score_pairs(node_emb, src, pos_dst)
            expanded_src = src.unsqueeze(1).expand_as(neg_dst)
            neg_score = model.score_pairs(
                node_emb, expanded_src.reshape(-1), neg_dst.reshape(-1)
            ).view_as(neg_dst)
            ranks.append(1 + (neg_score >= pos_score.unsqueeze(1)).sum(dim=1).float())

            pos_np = pos_score.detach().cpu().numpy()
            neg_np = neg_score.detach().cpu().numpy()
            for i in range(len(pos_np)):
                labels = np.concatenate(
                    [
                        np.ones(1, dtype=np.int64),
                        np.zeros(len(neg_np[i]), dtype=np.int64),
                    ]
                )
                scores = np.concatenate([[pos_np[i]], neg_np[i]])
                auc_scores.append(roc_auc_score(labels, scores))

        rank = torch.cat(ranks, dim=0)
        mrr = (1.0 / rank).mean().item()
        hits3 = (rank <= 3).float().mean().item()
        hits10 = (rank <= 10).float().mean().item()
        auc = float(np.mean(auc_scores)) if auc_scores else 0.0
        metrics[part] = {
            "mrr": mrr,
            "hits@3": hits3,
            "hits@10": hits10,
            "auc_roc": auc,
        }

    return metrics


def train(model, image_x, text_x, split, args, device: str):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    src_all = split["train"]["source_node"]
    pos_all = split["train"]["target_node"]
    num_nodes = image_x.size(0)
    best_valid_mrr = -1.0
    best_metrics = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(src_all.size(0))
        total_loss = 0.0

        for start in range(0, src_all.size(0), args.batch_size):
            idx = perm[start : start + args.batch_size]
            src = src_all[idx].to(device)
            pos_dst = pos_all[idx].to(device)
            neg_dst = torch.randint(0, num_nodes, pos_dst.shape, device=device)

            node_emb = model.encode(image_x.to(device), text_x.to(device))
            pos_score = model.score_pairs(node_emb, src, pos_dst)
            neg_score = model.score_pairs(node_emb, src, neg_dst)

            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_score, neg_score], dim=0),
                torch.cat(
                    [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
                ),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * idx.numel()

        metrics = evaluate(model, image_x, text_x, split, device)
        valid_mrr = metrics["valid"]["mrr"]
        print(
            f"[LP] epoch={epoch} loss={total_loss / src_all.size(0):.4f} "
            f"valid_mrr={valid_mrr:.4f} valid_hits3={metrics['valid']['hits@3']:.4f} "
            f"valid_hits10={metrics['valid']['hits@10']:.4f}"
        )

        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_metrics = metrics
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return best_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_x, text_x, split = load_data(data_dir)
    model = LinkPredictor(image_x.size(1), text_x.size(1), args.hidden_dim).to(device)
    best_metrics = train(model, image_x, text_x, split, args, device)

    summary = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_valid": best_metrics["valid"],
        "test": best_metrics["test"],
    }
    (output_dir / "lp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Final link prediction metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
