"""Train the FND-CLIP binary classifier on the balanced dataset."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MultiGuardDataset
from evaluate import compute_metrics
from models.fnd_clip import FNDCLIP


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for batch in pbar:
        optimizer.zero_grad()
        out = model(
            image=batch["image"].to(device),
            bert_ids=batch["bert_ids"].to(device),
            bert_mask=batch["bert_mask"].to(device),
            clip_pixels=batch["clip_pixels"].to(device),
            clip_ids=batch["clip_ids"].to(device),
            clip_mask=batch["clip_mask"].to(device),
        )
        logits = out["logits"].squeeze(-1)
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="[eval]"):
        out = model(
            image=batch["image"].to(device),
            bert_ids=batch["bert_ids"].to(device),
            bert_mask=batch["bert_mask"].to(device),
            clip_pixels=batch["clip_pixels"].to(device),
            clip_ids=batch["clip_ids"].to(device),
            clip_mask=batch["clip_mask"].to(device),
        )
        logits = out["logits"].squeeze(-1)
        labels = batch["label"].to(device)
        total_loss += criterion(logits, labels).item()
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(labels, probs)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/v1.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    full = MultiGuardDataset(
        csv_path=cfg["data"]["csv_path"],
        train=True,
        bert_model=cfg["models"]["bert"],
        clip_model=cfg["models"]["clip"],
        max_length=cfg["data"].get("max_length", 128),
    )
    val_frac = cfg["data"].get("val_split", 0.15)
    test_frac = cfg["data"].get("test_split", 0.15)
    n = len(full)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(
        full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

    val_set.dataset = MultiGuardDataset(
        cfg["data"]["csv_path"], train=False,
        bert_model=cfg["models"]["bert"],
        clip_model=cfg["models"]["clip"],
        max_length=cfg["data"].get("max_length", 128),
    )

    loader_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_batch,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = FNDCLIP(
        feat_dim=cfg["model"].get("feat_dim", 512),
        num_classes=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.4f} | "
              f"f1={val_metrics['f1']:.4f} | "
              f"auc={val_metrics['auc_roc']:.4f}")

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
            }, out_dir / "best.pt")
            print(f"  -> saved new best (AUC={best_auc:.4f})")

    print("\n=== Final test evaluation ===")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    with open(out_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump({k: float(v) for k, v in test_metrics.items()}, f)


if __name__ == "__main__":
    main()
