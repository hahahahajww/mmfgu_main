import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mmfgu.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal link prediction with edge unlearning"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--forget-ratio", type=float, default=0.2)
    parser.add_argument("--unlearn-epochs", type=int, default=20)
    parser.add_argument("--unlearn-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-dec", type=float, default=4.0)
    parser.add_argument("--alpha-anchor", type=float, default=0.2)
    parser.add_argument("--beta-mm", type=float, default=0.25)
    parser.add_argument("--delta-bd", type=float, default=0.1)
    parser.add_argument("--output-dir", default="lp_unlearning_outputs")
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


def sample_forget_edges(
    train_split: dict, forget_ratio: float, seed: int
) -> torch.Tensor:
    num_edges = train_split["source_node"].size(0)
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long)

    num_forget = max(1, int(num_edges * forget_ratio))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_edges, generator=generator)[:num_forget]


def build_retain_train_split(train_split: dict, forget_idx: torch.Tensor) -> dict:
    keep_mask = torch.ones(train_split["source_node"].size(0), dtype=torch.bool)
    keep_mask[forget_idx] = False
    return {
        "source_node": train_split["source_node"][keep_mask],
        "target_node": train_split["target_node"][keep_mask],
    }


def sample_mismatch_targets(
    num_nodes: int, target_nodes: torch.Tensor, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mismatch = torch.randint(0, num_nodes, target_nodes.shape, generator=generator)
    if num_nodes > 1:
        same_mask = mismatch == target_nodes.cpu()
        while same_mask.any():
            mismatch[same_mask] = torch.randint(
                0,
                num_nodes,
                (int(same_mask.sum().item()),),
                generator=generator,
            )
            same_mask = mismatch == target_nodes.cpu()
    return mismatch.long()


def boundary_nodes_from_forget_edges(
    train_split: dict, forget_split: dict
) -> torch.Tensor:
    forget_nodes = set(forget_split["source_node"].tolist()) | set(
        forget_split["target_node"].tolist()
    )
    collected = set()
    for src, dst in zip(
        train_split["source_node"].tolist(), train_split["target_node"].tolist()
    ):
        if src in forget_nodes:
            collected.add(dst)
        if dst in forget_nodes:
            collected.add(src)
    collected.difference_update(forget_nodes)
    return torch.tensor(sorted(collected), dtype=torch.long)


def edge_representation(
    node_emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
    return node_emb[src] * node_emb[dst]


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


@torch.no_grad()
def evaluate_forget_scores(
    model,
    image_x: torch.Tensor,
    text_x: torch.Tensor,
    forget_split: dict,
    device: str,
    batch_size: int = 4096,
) -> dict:
    model.eval()
    node_emb = model.encode(image_x.to(device), text_x.to(device))
    src_all = forget_split["source_node"]
    dst_all = forget_split["target_node"]
    scores = []

    for start in range(0, src_all.size(0), batch_size):
        src = src_all[start : start + batch_size].to(device)
        dst = dst_all[start : start + batch_size].to(device)
        batch_scores = model.score_pairs(node_emb, src, dst)
        scores.append(batch_scores.detach().cpu())

    if not scores:
        return {"mean_score": 0.0, "mean_sigmoid": 0.0}

    all_scores = torch.cat(scores).float()
    return {
        "mean_score": float(all_scores.mean().item()),
        "mean_sigmoid": float(all_scores.sigmoid().mean().item()),
    }


def train(model, image_x, text_x, train_split, eval_split, args, device: str):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    src_all = train_split["source_node"]
    pos_all = train_split["target_node"]
    num_nodes = image_x.size(0)
    best_valid_mrr = -1.0
    best_metrics = None
    best_state = None

    image_x_device = image_x.to(device)
    text_x_device = text_x.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(src_all.size(0))
        total_loss = 0.0

        for start in range(0, src_all.size(0), args.batch_size):
            idx = perm[start : start + args.batch_size]
            src = src_all[idx].to(device)
            pos_dst = pos_all[idx].to(device)
            neg_dst = torch.randint(0, num_nodes, pos_dst.shape, device=device)

            node_emb = model.encode(image_x_device, text_x_device)
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

        metrics = evaluate(model, image_x, text_x, eval_split, device)
        valid_mrr = metrics["valid"]["mrr"]
        print(
            f"[LP-Pretrain] epoch={epoch} loss={total_loss / src_all.size(0):.4f} "
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


def unlearn(
    model,
    image_x,
    text_x,
    train_split,
    retain_split,
    forget_split,
    eval_split,
    args,
    device,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.unlearn_lr, weight_decay=args.weight_decay
    )
    num_nodes = image_x.size(0)
    image_x_device = image_x.to(device)
    text_x_device = text_x.to(device)
    old_model = copy.deepcopy(model).to(device)
    for param in old_model.parameters():
        param.requires_grad = False
    old_model.eval()
    retain_src_all = retain_split["source_node"]
    retain_pos_all = retain_split["target_node"]
    forget_src_all = forget_split["source_node"]
    forget_pos_all = forget_split["target_node"]
    bd_nodes = boundary_nodes_from_forget_edges(train_split, forget_split).to(device)

    before_metrics = evaluate(model, image_x, text_x, eval_split, device)
    before_forget_scores = evaluate_forget_scores(
        model, image_x, text_x, forget_split, device, args.batch_size
    )

    best_valid_auc = before_metrics["valid"]["auc_roc"]
    best_metrics = before_metrics
    best_forget_scores = before_forget_scores
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_metrics = before_metrics
    last_forget_scores = before_forget_scores

    if forget_src_all.numel() == 0:
        return {
            "before_metrics": before_metrics,
            "best_metrics": best_metrics,
            "before_forget_scores": before_forget_scores,
            "best_forget_scores": best_forget_scores,
            "last_metrics": last_metrics,
            "last_forget_scores": last_forget_scores,
        }

    for epoch in range(1, args.unlearn_epochs + 1):
        model.train()
        retain_perm = torch.randperm(retain_src_all.size(0))
        total_loss = 0.0
        total_steps = max(
            (retain_src_all.size(0) + args.batch_size - 1) // args.batch_size,
            (forget_src_all.size(0) + args.batch_size - 1) // args.batch_size,
        )

        with torch.no_grad():
            old_node_emb = old_model.encode(image_x_device, text_x_device)

        for step in range(total_steps):
            retain_idx = retain_perm[
                step * args.batch_size : (step + 1) * args.batch_size
            ]

            node_emb = model.encode(image_x_device, text_x_device)
            forget_src = forget_src_all.to(device)
            forget_pos = forget_pos_all.to(device)
            mismatch = sample_mismatch_targets(
                num_nodes, forget_pos_all, args.seed + epoch
            ).to(device)

            forget_relation = edge_representation(node_emb, forget_src, forget_pos)
            mismatch_relation = edge_representation(
                node_emb, forget_src, mismatch
            ).detach()
            loss_dec = F.mse_loss(forget_relation, mismatch_relation)

            anchor_new = edge_representation(node_emb, forget_src, mismatch)
            anchor_old = edge_representation(old_node_emb, forget_src, mismatch)
            loss_anchor = F.mse_loss(anchor_new, anchor_old)

            if retain_idx.numel() > 0:
                retain_src = retain_src_all[retain_idx].to(device)
                retain_pos = retain_pos_all[retain_idx].to(device)
                retain_relation_new = edge_representation(
                    node_emb, retain_src, retain_pos
                )
                retain_relation_old = edge_representation(
                    old_node_emb, retain_src, retain_pos
                )
                loss_mm = F.mse_loss(retain_relation_new, retain_relation_old)
            else:
                loss_mm = loss_dec * 0.0

            if bd_nodes.numel() > 0:
                loss_bd = F.mse_loss(node_emb[bd_nodes], old_node_emb[bd_nodes])
            else:
                loss_bd = loss_dec * 0.0

            loss = (
                args.alpha_dec * loss_dec
                + args.alpha_anchor * loss_anchor
                + args.beta_mm * loss_mm
                + args.delta_bd * loss_bd
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        metrics = evaluate(model, image_x, text_x, eval_split, device)
        forget_scores = evaluate_forget_scores(
            model, image_x, text_x, forget_split, device, args.batch_size
        )
        last_metrics = metrics
        last_forget_scores = forget_scores
        print(
            f"[LP-Unlearn] epoch={epoch} loss={total_loss / total_steps:.4f} "
            f"valid_auc={metrics['valid']['auc_roc']:.4f} "
            f"forget_mean_sigmoid={forget_scores['mean_sigmoid']:.4f}"
        )

        if metrics["valid"]["auc_roc"] >= best_valid_auc:
            best_valid_auc = metrics["valid"]["auc_roc"]
            best_metrics = metrics
            best_forget_scores = forget_scores
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return {
        "before_metrics": before_metrics,
        "best_metrics": best_metrics,
        "before_forget_scores": before_forget_scores,
        "best_forget_scores": best_forget_scores,
        "last_metrics": last_metrics,
        "last_forget_scores": last_forget_scores,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_x, text_x, split = load_data(data_dir)
    forget_idx = sample_forget_edges(split["train"], args.forget_ratio, args.seed)
    retain_train = build_retain_train_split(split["train"], forget_idx)
    forget_split = {
        "source_node": split["train"]["source_node"][forget_idx],
        "target_node": split["train"]["target_node"][forget_idx],
    }

    model = LinkPredictor(image_x.size(1), text_x.size(1), args.hidden_dim).to(device)
    pretrain_metrics = train(
        model, image_x, text_x, split["train"], split, args, device
    )
    pretrained_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    pretrain_forget_scores = evaluate_forget_scores(
        model, image_x, text_x, forget_split, device, args.batch_size
    )

    unlearning_result = unlearn(
        model,
        image_x,
        text_x,
        split["train"],
        retain_train,
        forget_split,
        split,
        args,
        device,
    )
    before_metrics = unlearning_result["before_metrics"]
    after_metrics = unlearning_result["best_metrics"]
    before_forget_scores = unlearning_result["before_forget_scores"]
    after_forget_scores = unlearning_result["best_forget_scores"]

    summary = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "forget_ratio": args.forget_ratio,
        "unlearn_epochs": args.unlearn_epochs,
        "unlearn_lr": args.unlearn_lr,
        "alpha_dec": args.alpha_dec,
        "alpha_anchor": args.alpha_anchor,
        "beta_mm": args.beta_mm,
        "delta_bd": args.delta_bd,
        "forget_request": {
            "forget_edge_count": int(forget_idx.numel()),
            "retain_edge_count": int(retain_train["source_node"].numel()),
            "boundary_node_count": int(
                boundary_nodes_from_forget_edges(split["train"], forget_split).numel()
            ),
            "forget_edge_indices_sample": forget_idx[:20].tolist(),
            "forget_edges_sample": list(
                zip(
                    forget_split["source_node"][:20].tolist(),
                    forget_split["target_node"][:20].tolist(),
                )
            ),
        },
        "pretrain": {
            "best_valid": pretrain_metrics["valid"],
            "test": pretrain_metrics["test"],
            "forget_scores": pretrain_forget_scores,
        },
        "unlearning": {
            "before": {
                "best_valid": before_metrics["valid"],
                "test": before_metrics["test"],
                "forget_scores": before_forget_scores,
            },
            "after": {
                "best_valid": after_metrics["valid"],
                "test": after_metrics["test"],
                "forget_scores": after_forget_scores,
            },
            "last_epoch": {
                "best_valid": unlearning_result["last_metrics"]["valid"],
                "test": unlearning_result["last_metrics"]["test"],
                "forget_scores": unlearning_result["last_forget_scores"],
            },
            "metric_delta": {
                "valid_mrr_delta": after_metrics["valid"]["mrr"]
                - before_metrics["valid"]["mrr"],
                "test_mrr_delta": after_metrics["test"]["mrr"]
                - before_metrics["test"]["mrr"],
                "valid_hits@3_delta": after_metrics["valid"]["hits@3"]
                - before_metrics["valid"]["hits@3"],
                "test_hits@3_delta": after_metrics["test"]["hits@3"]
                - before_metrics["test"]["hits@3"],
                "valid_auc_roc_delta": after_metrics["valid"]["auc_roc"]
                - before_metrics["valid"]["auc_roc"],
                "test_auc_roc_delta": after_metrics["test"]["auc_roc"]
                - before_metrics["test"]["auc_roc"],
                "forget_mean_score_delta": after_forget_scores["mean_score"]
                - before_forget_scores["mean_score"],
                "forget_mean_sigmoid_delta": after_forget_scores["mean_sigmoid"]
                - before_forget_scores["mean_sigmoid"],
            },
        },
    }
    (output_dir / "lp_unlearning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "pretrained_state": pretrained_state,
            "unlearned_state": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "forget_edge_indices": forget_idx.cpu(),
        },
        output_dir / "lp_unlearning_state.pt",
    )
    print("Final link prediction unlearning metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
