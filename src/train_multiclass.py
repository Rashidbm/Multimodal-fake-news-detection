"""Train FND-CLIP in 5-class mode (kept for reference; V1 ships as binary)."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
from models.fnd_clip import FNDCLIP


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MultiClassDataset(Dataset):
    """5-class dataset using scenario as label (0-4)."""

    def __init__(self, df, bert_model="bert-base-uncased",
                 clip_model="openai/clip-vit-base-patch32",
                 max_length=128, train=True):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.bert_tok = BertTokenizer.from_pretrained(bert_model)
        self.clip_proc = CLIPProcessor.from_pretrained(clip_model)
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            pil = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)

        image = self.tf(pil)
        bert = self.bert_tok(str(row["text"]), padding="max_length",
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        clip = self.clip_proc(images=pil, text=str(row["text"]),
                              return_tensors="pt", padding="max_length",
                              truncation=True, max_length=77)

        # scenario 1-5 -> label 0-4
        label = int(row["scenario"]) - 1

        return {
            "image": image,
            "bert_ids": bert["input_ids"].squeeze(0),
            "bert_mask": bert["attention_mask"].squeeze(0),
            "clip_pixels": clip["pixel_values"].squeeze(0),
            "clip_ids": clip["input_ids"].squeeze(0),
            "clip_mask": clip["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def compute_metrics_multiclass(labels, logits, num_classes=5):
    """Multi-class metrics: accuracy, macro F1, macro/per-class P/R, macro AUC-ROC."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score, confusion_matrix)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)

    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    # Macro AUC-ROC (one-vs-rest)
    try:
        ohe = np.eye(num_classes)[labels]
        out["auc_roc_macro"] = roc_auc_score(ohe, probs, average="macro", multi_class="ovr")
        out["auc_roc_weighted"] = roc_auc_score(ohe, probs, average="weighted", multi_class="ovr")
    except Exception as e:
        out["auc_roc_macro"] = float("nan")
        out["auc_roc_weighted"] = float("nan")

    # Per-class metrics
    p_per = precision_score(labels, preds, average=None, zero_division=0, labels=list(range(num_classes)))
    r_per = recall_score(labels, preds, average=None, zero_division=0, labels=list(range(num_classes)))
    f_per = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(num_classes)))
    for i in range(num_classes):
        out[f"scenario_{i+1}_precision"] = float(p_per[i])
        out[f"scenario_{i+1}_recall"] = float(r_per[i])
        out[f"scenario_{i+1}_f1"] = float(f_per[i])

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    out["confusion_matrix"] = cm.tolist()
    return out


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
        loss = criterion(out["logits"], batch["label"].to(device))
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
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="[eval]"):
        out = model(
            image=batch["image"].to(device),
            bert_ids=batch["bert_ids"].to(device),
            bert_mask=batch["bert_mask"].to(device),
            clip_pixels=batch["clip_pixels"].to(device),
            clip_ids=batch["clip_ids"].to(device),
            clip_mask=batch["clip_mask"].to(device),
        )
        labels = batch["label"].to(device)
        total_loss += criterion(out["logits"], labels).item()
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics_multiclass(labels, logits, num_classes=5)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/v1_multiclass.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_csv(cfg["data"]["csv_path"])
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_set = MultiClassDataset(train_df, train=True,
                                  bert_model=cfg["models"]["bert"],
                                  clip_model=cfg["models"]["clip"],
                                  max_length=cfg["data"].get("max_length", 128))
    val_set = MultiClassDataset(val_df, train=False,
                                bert_model=cfg["models"]["bert"],
                                clip_model=cfg["models"]["clip"],
                                max_length=cfg["data"].get("max_length", 128))
    test_set = MultiClassDataset(test_df, train=False,
                                 bert_model=cfg["models"]["bert"],
                                 clip_model=cfg["models"]["clip"],
                                 max_length=cfg["data"].get("max_length", 128))

    loader_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 2),
        collate_fn=collate,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = FNDCLIP(feat_dim=cfg["model"].get("feat_dim", 512),
                    num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.4f} | "
              f"f1_macro={val_metrics['f1_macro']:.4f} | "
              f"auc_macro={val_metrics['auc_roc_macro']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_metrics": val_metrics, "config": cfg,
            }, out_dir / "best.pt")
            print(f"  -> saved best (f1_macro={best_f1:.4f})")

    print("\n=== Final test evaluation ===")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    with open(out_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump({k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in test_metrics.items()}, f)


if __name__ == "__main__":
    main()
